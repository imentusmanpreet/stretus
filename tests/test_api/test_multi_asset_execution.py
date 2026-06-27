"""
tests/test_api/test_multi_asset_execution.py
─────────────────────────────────────────────
Unit tests for the multi-asset (equity_cash + crypto_spot) execution path.

Scope:
  - Canonical-symbol derivation per asset class.
  - ref_data adapter-symbol lookup back-compat shim.
  - InstrumentDefaults asset-class branching.
  - Entry-gate trading-window asset-class branching (24/7 crypto).
  - TradeManager bracket-order venue conventions for crypto vs equity.
  - RiskManager fractional quantity sizing for crypto.
  - BinanceClient kline → DataFrame normalisation (no network — payload mocked).
"""
from __future__ import annotations

import types
from decimal import Decimal

import pandas as pd
import pytest

from app.schemas.execution import (
    AssetClass,
    ExchangeOrderType,
    GatesConfig,
    OrderValidity,
    ProductType,
)
from app.services.execution import ref_data_service as rds
from app.services.execution.entry_gates import evaluate_entry_gates
from app.services.execution.market_data.binance_client import (
    BinanceClient,
    _binance_klines_to_df,
    _plan_for,
    _strip_canonical_crypto_suffix,
)
from app.services.execution.market_data.factory import (
    get_market_data_client,
    reset_market_data_clients,
)
from app.services.execution.market_data.upstox_client import UpstoxClient
from app.services.execution.ref_data_service import InstrumentDefaults
from app.services.execution.risk_manager import RiskInput, RiskManager
from app.services.execution.trade_manager import TradeManager


# ─────────────────────────────────────────────────────────────────────────────
# Canonical-symbol derivation
# ─────────────────────────────────────────────────────────────────────────────

class TestCanonicalSymbol:
    def test_equity_ns_suffix(self) -> None:
        assert rds.to_canonical_symbol("RELIANCE.NS", AssetClass.equity_cash) == "equity:RELIANCE@NSE"

    def test_equity_bse_suffix(self) -> None:
        assert rds.to_canonical_symbol("RELIANCE.BO", AssetClass.equity_cash) == "equity:RELIANCE@BSE"

    def test_equity_bare_ticker_defaults_to_nse(self) -> None:
        assert rds.to_canonical_symbol("reliance", AssetClass.equity_cash) == "equity:RELIANCE@NSE"

    def test_equity_canonical_is_idempotent(self) -> None:
        assert (
            rds.to_canonical_symbol("equity:RELIANCE@NSE", AssetClass.equity_cash)
            == "equity:RELIANCE@NSE"
        )

    def test_crypto_base_quote_form(self) -> None:
        assert (
            rds.to_canonical_symbol("BTC_USDT", AssetClass.crypto_spot)
            == "crypto_spot:BTC_USDT@BINANCE_SPOT"
        )

    def test_crypto_slash_form(self) -> None:
        assert (
            rds.to_canonical_symbol("ETH/USDC", AssetClass.crypto_spot)
            == "crypto_spot:ETH_USDC@BINANCE_SPOT"
        )

    def test_crypto_canonical_is_idempotent(self) -> None:
        assert (
            rds.to_canonical_symbol(
                "crypto_spot:BTC_USDT@BINANCE_SPOT", AssetClass.crypto_spot
            )
            == "crypto_spot:BTC_USDT@BINANCE_SPOT"
        )

    def test_crypto_bare_concatenated_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            rds.to_canonical_symbol("BTCUSDT", AssetClass.crypto_spot)


# ─────────────────────────────────────────────────────────────────────────────
# Default adapter selection
# ─────────────────────────────────────────────────────────────────────────────

class TestDefaultAdapter:
    def test_equity_defaults_to_upstox(self) -> None:
        assert rds.default_adapter_id(AssetClass.equity_cash) == "upstox_rest"

    def test_crypto_defaults_to_binance(self) -> None:
        assert rds.default_adapter_id(AssetClass.crypto_spot) == "binance_rest"


# ─────────────────────────────────────────────────────────────────────────────
# Market-data factory
# ─────────────────────────────────────────────────────────────────────────────

class TestMarketDataFactory:
    def setup_method(self) -> None:
        reset_market_data_clients()

    def _force_settings(self, monkeypatch, policy: str) -> None:
        # Live market data flows through the stretus-backend gateway (InternalMarketDataClient),
        # so the factory AND the gateway client must see the same forced settings — independent of
        # any ambient .env. The gateway client reads historical_data_url at construction time.
        import app.services.execution.market_data.factory as fac
        import app.services.execution.market_data.internal_client as internal
        fake_settings = type("S", (), {
            "equity_market_data_source": policy,
            "historical_data_url": "http://gateway.test",
            "market_data_timeout_seconds": 5.0,
        })()
        monkeypatch.setattr(fac, "get_settings", lambda: fake_settings)
        monkeypatch.setattr(internal, "get_settings", lambda: fake_settings)
        fac.reset_market_data_clients()

    def test_equity_uses_gateway_primary_under_default_resilient_policy(self, monkeypatch) -> None:
        # The resilient policy wraps the live gateway client in a gateway→backtest_feed fallback
        # chain, with the gateway (InternalMarketDataClient) as the primary.
        from app.services.execution.market_data.backtest_feed_client import BacktestFeedClient
        from app.services.execution.market_data.internal_client import InternalMarketDataClient
        from app.services.execution.market_data.resilient_client import ResilientMarketDataClient

        self._force_settings(monkeypatch, "resilient")
        client = get_market_data_client(AssetClass.equity_cash)
        assert isinstance(client, ResilientMarketDataClient)
        assert isinstance(client._clients[0], InternalMarketDataClient)
        assert isinstance(client._clients[1], BacktestFeedClient)

    def test_equity_source_policy_upstox_only(self, monkeypatch) -> None:
        # "upstox" = live broker feed, now reached through the gateway (no fallback wrapping).
        from app.services.execution.market_data.internal_client import InternalMarketDataClient

        self._force_settings(monkeypatch, "upstox")
        assert isinstance(get_market_data_client(AssetClass.equity_cash), InternalMarketDataClient)

    def test_crypto_returns_gateway_client(self, monkeypatch) -> None:
        # Crypto also routes through the gateway; the gateway owns broker (Binance) selection.
        from app.services.execution.market_data.internal_client import InternalMarketDataClient

        self._force_settings(monkeypatch, "resilient")
        client = get_market_data_client(AssetClass.crypto_spot)
        assert isinstance(client, InternalMarketDataClient)

    def test_factory_caches_per_asset_class(self) -> None:
        a = get_market_data_client(AssetClass.crypto_spot)
        b = get_market_data_client(AssetClass.crypto_spot)
        assert a is b


# ─────────────────────────────────────────────────────────────────────────────
# Entry-gate trading window — asset-class aware
# ─────────────────────────────────────────────────────────────────────────────

def _make_df_with_ts(ts_utc: str) -> pd.DataFrame:
    idx = pd.DatetimeIndex([pd.Timestamp(ts_utc, tz="UTC")])
    return pd.DataFrame(
        {"Open": [100.0], "High": [101.0], "Low": [99.0], "Close": [100.5], "Volume": [1000.0]},
        index=idx,
    )


class _StubExecState:
    consecutive_losses = 0
    last_trade_was_loss = None
    bars_since_last_trade = 0


class _StubRuleEngine:
    def evaluate_entry(self, df, block):  # noqa: D401
        return True, []


class TestEntryGateWindow:
    def test_crypto_no_window_means_24x7(self) -> None:
        df = _make_df_with_ts("2026-05-28T03:00:00")  # 03:00 UTC, outside any IST equity window
        gates = GatesConfig()  # no window set
        result = evaluate_entry_gates(
            df=df,
            gates=gates,
            exec_state=_StubExecState(),
            side="BUY",
            rule_engine=_StubRuleEngine(),
            entry_block={"trigger": {}, "filters": []},
            asset_class=AssetClass.crypto_spot,
        )
        assert result.passed is True
        assert any("crypto 24/7" in m for m in result.messages)

    def test_crypto_window_uses_utc(self) -> None:
        # Bar at 02:00 UTC; window 03:00–10:00 UTC → blocked.
        df = _make_df_with_ts("2026-05-28T02:00:00")
        gates = GatesConfig(
            entry_window_start="03:00",
            entry_window_end="10:00",
        )
        result = evaluate_entry_gates(
            df=df, gates=gates, exec_state=_StubExecState(), side="BUY",
            rule_engine=_StubRuleEngine(), entry_block={"trigger": {}, "filters": []},
            asset_class=AssetClass.crypto_spot,
        )
        assert result.passed is False
        assert result.blocked_by == "entry_window"
        assert "UTC" in (result.messages[-1] if result.messages else "")

    def test_equity_window_uses_ist(self) -> None:
        # Bar at 02:00 UTC = 07:30 IST; equity window 09:15–15:30 IST → blocked.
        df = _make_df_with_ts("2026-05-28T02:00:00")
        gates = GatesConfig(
            entry_window_start="09:15",
            entry_window_end="15:30",
        )
        result = evaluate_entry_gates(
            df=df, gates=gates, exec_state=_StubExecState(), side="BUY",
            rule_engine=_StubRuleEngine(), entry_block={"trigger": {}, "filters": []},
            asset_class=AssetClass.equity_cash,
        )
        assert result.passed is False
        assert result.blocked_by == "entry_window"
        assert "IST" in (result.messages[-1] if result.messages else "")


# ─────────────────────────────────────────────────────────────────────────────
# TradeManager — venue conventions per asset class
# ─────────────────────────────────────────────────────────────────────────────

class TestTradeManagerVenueConventions:
    def setup_method(self) -> None:
        self.tm = TradeManager()

    def test_equity_intraday_bracket_uses_mis_limit_sl_m_day(self) -> None:
        b = self.tm.build_bracket_order(
            symbol="RELIANCE.NS", quantity=5, entry_price=2500.0,
            stop_loss_price=2450.0, take_profit_price=2600.0,
            strategy_type="intraday", strategy_id="abc", mode="paper",
        )
        assert b.entry_order.product_type   == ProductType.mis
        assert b.entry_order.order_type     == ExchangeOrderType.limit
        assert b.entry_order.validity       == OrderValidity.day
        assert b.stop_loss_order.order_type == ExchangeOrderType.sl_m
        assert b.entry_order.quantity       == Decimal(5)

    def test_equity_positional_bracket_uses_cnc(self) -> None:
        b = self.tm.build_bracket_order(
            symbol="RELIANCE.NS", quantity=5, entry_price=2500.0,
            stop_loss_price=2450.0, take_profit_price=2600.0,
            strategy_type="positional", strategy_id="abc", mode="paper",
        )
        assert b.entry_order.product_type == ProductType.cnc

    def test_crypto_spot_bracket_uses_spot_stop_loss_gtc(self) -> None:
        b = self.tm.build_bracket_order(
            symbol="BTC_USDT", quantity=Decimal("0.00125"), entry_price=50000.0,
            stop_loss_price=49000.0, take_profit_price=51000.0,
            strategy_type="intraday", strategy_id="xyz", mode="paper",
            asset_class=AssetClass.crypto_spot,
        )
        assert b.entry_order.product_type   == ProductType.spot
        assert b.entry_order.order_type     == ExchangeOrderType.limit
        assert b.entry_order.validity       == OrderValidity.gtc
        assert b.stop_loss_order.order_type == ExchangeOrderType.stop_loss
        # Fractional quantity is preserved losslessly as Decimal.
        assert b.entry_order.quantity == Decimal("0.00125")
        assert b.metadata["asset_class"] == "crypto_spot"

    def test_idempotency_key_is_asset_class_scoped(self) -> None:
        b_eq = self.tm.build_bracket_order(
            symbol="RELIANCE.NS", quantity=5, entry_price=2500.0,
            stop_loss_price=2450.0, take_profit_price=2600.0,
            strategy_type="intraday", strategy_id="same-id",
            mode="paper", bar_datetime="2026-05-28T10:00:00",
        )
        b_cr = self.tm.build_bracket_order(
            symbol="RELIANCE.NS", quantity=Decimal(5), entry_price=2500.0,
            stop_loss_price=2450.0, take_profit_price=2600.0,
            strategy_type="intraday", strategy_id="same-id",
            mode="paper", bar_datetime="2026-05-28T10:00:00",
            asset_class=AssetClass.crypto_spot,
        )
        # Different asset_class → different key — they must not collide on
        # OMS-side dedup even if every other input matches.
        assert b_eq.idempotency_key != b_cr.idempotency_key

    def test_float_quantity_is_rejected(self) -> None:
        with pytest.raises(TypeError):
            self.tm.build_bracket_order(
                symbol="BTC_USDT", quantity=0.00125,  # type: ignore[arg-type]
                entry_price=50000.0, stop_loss_price=49000.0,
                take_profit_price=51000.0,
                strategy_type="intraday", strategy_id="xyz", mode="paper",
                asset_class=AssetClass.crypto_spot,
            )


# ─────────────────────────────────────────────────────────────────────────────
# RiskManager — fractional crypto sizing
# ─────────────────────────────────────────────────────────────────────────────

class TestRiskManagerCryptoSizing:
    def setup_method(self) -> None:
        self.rm = RiskManager()

    def _crypto_inputs(self) -> RiskInput:
        risk_cfg = types.SimpleNamespace(stop_loss_pct=2.0, take_profit_pct=5.0)
        exec_state = types.SimpleNamespace(
            capital=10_000.0,
            max_risk_per_trade_pct=2.0,
            min_trade_value=10.0,
        )
        instr = InstrumentDefaults(
            tick_size=0.01,
            lot_size=1,
            upper_circuit=55000.0,
            lower_circuit=45000.0,
            qty_step_size=0.00001,
            min_notional=10.0,
        )
        return RiskInput(
            entry_price=50000.0,
            risk_config=risk_cfg,
            exec_state=exec_state,
            instrument=instr,
            asset_class=AssetClass.crypto_spot,
        )

    def test_crypto_position_size_is_fractional_decimal(self) -> None:
        out = self.rm.calculate(self._crypto_inputs())
        assert out.ok is True
        assert isinstance(out.position_size, Decimal)
        # 10000 * 2% = 200 risk; SL = 50000 * (1 - 0.02) = 49000;
        # risk/unit = 1000; raw = 0.2; capped at 20% capital = 2000 USDT = 0.04 BTC.
        assert out.position_size == Decimal("0.04000")
        assert out.principal_amount == 2000.0

    def test_crypto_min_notional_blocks_small_trade(self) -> None:
        ri = self._crypto_inputs()
        # Tiny capital → principal under min_notional → ok=False.
        ri.exec_state = types.SimpleNamespace(
            capital=0.5,
            max_risk_per_trade_pct=2.0,
            min_trade_value=0.01,
        )
        out = self.rm.calculate(ri)
        assert out.ok is False

    def test_equity_path_still_returns_decimal_int(self) -> None:
        risk_cfg = types.SimpleNamespace(stop_loss_pct=2.0, take_profit_pct=5.0)
        exec_state = types.SimpleNamespace(
            capital=100_000.0,
            max_risk_per_trade_pct=2.0,
            min_trade_value=500.0,
        )
        instr = InstrumentDefaults(
            tick_size=0.05, lot_size=1,
            upper_circuit=3000.0, lower_circuit=2000.0,
        )
        out = self.rm.calculate(RiskInput(
            entry_price=2500.0, risk_config=risk_cfg, exec_state=exec_state,
            instrument=instr, asset_class=AssetClass.equity_cash,
        ))
        assert out.ok is True
        assert isinstance(out.position_size, Decimal)
        # Decimal is int-valued for equity.
        assert out.position_size == out.position_size.to_integral_value()


# ─────────────────────────────────────────────────────────────────────────────
# Binance kline normaliser
# ─────────────────────────────────────────────────────────────────────────────

class TestBinanceKlineNormalisation:
    def test_klines_become_dataframe_with_expected_shape(self) -> None:
        # [open_time_ms, open, high, low, close, volume, ...]
        klines = [
            [1_700_000_000_000, "100.0", "102.5", "99.5", "101.0", "12.5"],
            [1_700_000_060_000, "101.0", "101.7", "100.5", "100.9", "13.1"],
        ]
        df = _binance_klines_to_df(klines)
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert len(df) == 2
        assert df.index.tz is not None and str(df.index.tz) == "UTC"
        assert df["close"].iloc[-1] == 100.9

    def test_empty_klines_raise(self) -> None:
        with pytest.raises(ValueError):
            _binance_klines_to_df([])

    def test_native_timeframe_passthrough(self) -> None:
        assert _plan_for("5m") == ("5m", None)
        assert _plan_for("1h") == ("1h", None)

    def test_non_native_timeframe_resamples(self) -> None:
        assert _plan_for("10m") == ("5m", "10min")
        assert _plan_for("45m") == ("15m", "45min")

    def test_unsupported_timeframe_rejected(self) -> None:
        with pytest.raises(ValueError):
            _plan_for("7m")


# ─────────────────────────────────────────────────────────────────────────────
# Binance symbol best-effort fallback
# ─────────────────────────────────────────────────────────────────────────────

class TestBinanceSymbolFallback:
    @pytest.mark.parametrize(
        "input_symbol,expected",
        [
            ("BTC_USDT",                          "BTCUSDT"),
            ("crypto_spot:BTC_USDT@BINANCE_SPOT", "BTCUSDT"),
            ("ETH/USDC",                          "ETHUSDC"),
            ("BTCUSDT",                           "BTCUSDT"),
            ("btc_usdt",                          "BTCUSDT"),
        ],
    )
    def test_strip_returns_binance_native(self, input_symbol: str, expected: str) -> None:
        assert _strip_canonical_crypto_suffix(input_symbol) == expected


# ─────────────────────────────────────────────────────────────────────────────
# BinanceClient end-to-end (kline fetch with monkeypatched httpx)
# ─────────────────────────────────────────────────────────────────────────────

class _StubResponse:
    def __init__(self, status_code: int, payload):
        self.status_code = status_code
        self._payload = payload
        self.text = "stub"

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"stub {self.status_code}")


class _StubAsyncClient:
    """Monkeypatched httpx.AsyncClient that returns a single _StubResponse."""

    def __init__(self, response: _StubResponse):
        self._response = response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def get(self, url, params=None, headers=None):
        return self._response


class TestBinanceClientNetwork:
    @pytest.mark.asyncio
    async def test_fetch_candles_uses_adapter_symbol(self, monkeypatch) -> None:
        payload = [
            [1_700_000_000_000, "100", "101", "99", "100.5", "10"],
            [1_700_000_060_000, "100.5", "101.5", "100", "101", "11"],
        ]
        captured = {}

        def _async_client_ctor(*args, **kwargs):
            return _StubAsyncClient(_StubResponse(200, payload))

        # Capture the URL request the client builds.
        import app.services.execution.market_data.binance_client as bc

        original_get = _StubAsyncClient.get

        async def _capturing_get(self, url, params=None, headers=None):  # noqa: ANN001
            captured["url"] = url
            captured["params"] = params
            return _StubResponse(200, payload)

        monkeypatch.setattr(_StubAsyncClient, "get", _capturing_get)
        monkeypatch.setattr(bc.httpx, "AsyncClient", _async_client_ctor)

        client = BinanceClient()
        df = await client.fetch_candles(
            symbol="BTC_USDT", timeframe="5m", lookback=2,
            adapter_symbol="BTCUSDT",
        )

        assert captured["params"]["symbol"] == "BTCUSDT"
        assert captured["params"]["interval"] == "5m"
        assert len(df) == 2
        assert df["close"].iloc[-1] == 101.0

        # Restore for any later tests.
        monkeypatch.setattr(_StubAsyncClient, "get", original_get)
