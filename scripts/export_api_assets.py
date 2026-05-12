from __future__ import annotations

import copy
import json
import re
import sys
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
QUANT_ENGINE_ROOT = ROOT / "quant_engine"
if str(QUANT_ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(QUANT_ENGINE_ROOT))

from app.main import app as trading_app


OUTPUT_DIR = ROOT / "postman"
POSTMAN_SCHEMA_URL = (
    "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
)


@dataclass(frozen=True)
class ServiceConfig:
    name: str
    slug: str
    app: Any
    base_url_variable: str
    default_base_url: str
    description: str


class QuantRunRequest(BaseModel):
    backtest_id: str
    strategy_id: str
    yaml_path: str


class QuantRunResponse(BaseModel):
    backtest_id: str
    status: str
    message: str


def build_quant_engine_openapi_app() -> FastAPI:
    app = FastAPI(
        title="Stretus Quant Engine",
        description=(
            "Real backtest calculator - downloads market data, simulates "
            "trades, calculates metrics."
        ),
        version="1.0.0",
    )

    @app.post("/run", response_model=QuantRunResponse)
    async def start_backtest(body: QuantRunRequest, background_tasks: BackgroundTasks):
        """
        Triggered by FastAPI when user starts a backtest.
        Runs the calculation in a background task and posts results back.
        """
        return QuantRunResponse(
            backtest_id=body.backtest_id,
            status="running",
            message="Backtest started. Results will be posted back to FastAPI.",
        )

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "service": "stretus-quant-engine",
            "version": "1.0.0",
            "fastapi_url": "http://localhost:8000",
        }

    @app.get("/status/{backtest_id}")
    async def get_status(backtest_id: str):
        return {"backtest_id": backtest_id, "status": "unknown"}

    return app


quant_app = build_quant_engine_openapi_app()


SERVICES = [
    ServiceConfig(
        name="Stretus Trading API",
        slug="stretus-trading-api",
        app=trading_app,
        base_url_variable="api_base_url",
        default_base_url="http://localhost:8000",
        description=(
            "Primary strategy-building API. Covers health checks, async chat, "
            "strategy confirmation, and backtesting."
        ),
    ),
    ServiceConfig(
        name="Stretus Quant Engine",
        slug="stretus-quant-engine",
        app=quant_app,
        base_url_variable="quant_base_url",
        default_base_url="http://localhost:8001",
        description=(
            "Async backtest worker service used by the main API."
        ),
    ),
]


MANUAL_REQUEST_EXAMPLES: dict[tuple[str, str, str], Any] = {
    (
        "stretus-trading-api",
        "post",
        "/api/v1/strategy/chats",
    ): {
        "title": "NIFTY RSI Strategy",
    },
    (
        "stretus-trading-api",
        "post",
        "/api/v1/strategy/chats/{session_id}/messages",
    ): {
        "content": (
            "Build a bullish NIFTY intraday strategy on 15m timeframe using "
            "RSI oversold entries and SMA-based confirmation."
        ),
    },
    (
        "stretus-trading-api",
        "post",
        "/api/v1/strategy/strategies",
    ): {
        "session_id": "{{session_id}}",
    },
    (
        "stretus-trading-api",
        "post",
        "/api/v1/strategy/backtest",
    ): {
        "strategy_id": "{{strategy_id}}",
    },
    (
        "stretus-trading-api",
        "put",
        "/api/v1/strategy/backtest/{backtest_id}/result",
    ): {
        "overview": {
            "net_return_pct": 14.2,
            "sharpe_ratio": 1.31,
            "max_drawdown_pct": -4.8,
            "win_rate_pct": 58.0,
            "total_trades": 42,
            "profit_factor": 1.67,
        },
        "performance": {
            "monthly_returns": [
                {"month": "2025-01", "return_pct": 3.1},
                {"month": "2025-02", "return_pct": 1.9},
            ],
            "equity_curve": [
                {"date": "2025-01-01", "value": 100000},
                {"date": "2025-02-01", "value": 103100},
            ],
            "stability_score": 0.74,
        },
        "trades": {
            "trade_list": [
                {
                    "entry_date": "2025-01-03",
                    "exit_date": "2025-01-06",
                    "symbol": "NIFTY",
                    "side": "long",
                    "pnl_pct": 1.8,
                }
            ],
            "avg_holding_period": "2 days",
            "trade_frequency": "weekly",
        },
        "risk": {
            "var_95": -1.9,
            "expected_shortfall": -2.7,
            "max_drawdown_days": 11,
            "drawdown_periods": [
                {"start": "2025-02-10", "end": "2025-02-21", "depth_pct": -4.8}
            ],
        },
        "markets": {
            "bull": {"return_pct": 8.4, "win_rate": 63.0, "trades": 19},
            "correction": {"return_pct": 2.1, "win_rate": 50.0, "trades": 10},
            "sideways": {"return_pct": 3.7, "win_rate": 55.0, "trades": 13},
        },
    },
    (
        "stretus-quant-engine",
        "post",
        "/run",
    ): {
        "backtest_id": "{{backtest_id}}",
        "strategy_id": "{{strategy_id}}",
        "yaml_path": "{{yaml_path}}",
    },
}


VARIABLE_DEFAULTS = OrderedDict(
    [
        ("api_base_url", "http://localhost:8000"),
        ("quant_base_url", "http://localhost:8001"),
        ("session_id", "replace-with-session-id"),
        ("strategy_id", "replace-with-strategy-id"),
        ("backtest_id", "replace-with-backtest-id"),
        ("signal_name", "rsi_oversold"),
        ("yaml_path", "strategies/sample_strategy.yaml"),
        ("query", "nifty breakout"),
        ("limit", "10"),
        ("category", "momentum"),
    ]
)

HTTP_METHODS = ("get", "post", "put", "patch", "delete")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def clean_tag(tag: str) -> str:
    cleaned = re.sub(r"^[^A-Za-z0-9]+", "", tag or "").strip()
    return cleaned or "General"


def collection_variable_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_") or "value"


def resolve_ref(spec: dict[str, Any], ref: str) -> dict[str, Any]:
    if not ref.startswith("#/"):
        raise ValueError(f"Unsupported ref: {ref}")

    current: Any = spec
    for part in ref[2:].split("/"):
        current = current[part]
    if not isinstance(current, dict):
        raise ValueError(f"Ref does not resolve to an object schema: {ref}")
    return current


def merge_ref_schema(spec: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    if "$ref" not in schema:
        return schema

    merged = copy.deepcopy(resolve_ref(spec, schema["$ref"]))
    for key, value in schema.items():
        if key == "$ref":
            continue
        merged[key] = value
    return merged


def scalar_example(schema: dict[str, Any]) -> Any:
    if "default" in schema:
        return copy.deepcopy(schema["default"])
    if "enum" in schema and schema["enum"]:
        return copy.deepcopy(schema["enum"][0])
    if "const" in schema:
        return copy.deepcopy(schema["const"])
    if "example" in schema:
        return copy.deepcopy(schema["example"])

    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        schema_type = next((item for item in schema_type if item != "null"), None)

    if schema_type == "string":
        fmt = schema.get("format")
        if fmt == "uuid":
            return "11111111-1111-1111-1111-111111111111"
        if fmt == "date-time":
            return "2026-01-01T00:00:00Z"
        if fmt == "date":
            return "2026-01-01"
        if fmt == "email":
            return "user@example.com"
        if fmt == "uri":
            return "https://example.com/resource"
        title = (schema.get("title") or "").lower()
        if "path" in title:
            return "/tmp/example.txt"
        return "string"
    if schema_type == "integer":
        minimum = schema.get("minimum")
        if minimum is not None:
            return minimum
        return 0
    if schema_type == "number":
        minimum = schema.get("minimum")
        if minimum is not None:
            return minimum
        return 0.0
    if schema_type == "boolean":
        return False
    return None


def example_from_schema(
    spec: dict[str, Any],
    schema: dict[str, Any] | None,
    seen_refs: set[str] | None = None,
) -> Any:
    if not schema:
        return None

    seen_refs = seen_refs or set()
    if "$ref" in schema:
        ref = schema["$ref"]
        if ref in seen_refs:
            return None
        seen_refs = set(seen_refs)
        seen_refs.add(ref)
        merged = merge_ref_schema(spec, schema)
        return example_from_schema(spec, merged, seen_refs)

    if "example" in schema:
        return copy.deepcopy(schema["example"])

    examples = schema.get("examples")
    if isinstance(examples, dict) and examples:
        first_example = next(iter(examples.values()))
        if isinstance(first_example, dict) and "value" in first_example:
            return copy.deepcopy(first_example["value"])
        return copy.deepcopy(first_example)

    if "allOf" in schema:
        merged_object: dict[str, Any] = {}
        collected = []
        for part in schema["allOf"]:
            example = example_from_schema(spec, part, seen_refs)
            if isinstance(example, dict):
                merged_object.update(example)
            elif example is not None:
                collected.append(example)
        if merged_object:
            return merged_object
        return collected[0] if collected else None

    variants = schema.get("anyOf") or schema.get("oneOf")
    if variants:
        non_null_variants = [
            variant
            for variant in variants
            if variant.get("type") != "null"
        ]
        chosen = non_null_variants[0] if non_null_variants else variants[0]
        return example_from_schema(spec, chosen, seen_refs)

    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        schema_type = next((item for item in schema_type if item != "null"), None)

    if not schema_type and "properties" in schema:
        schema_type = "object"

    if schema_type == "object":
        properties = schema.get("properties", {})
        result: dict[str, Any] = {}
        for property_name, property_schema in properties.items():
            property_example = example_from_schema(spec, property_schema, seen_refs)
            if property_example is not None:
                result[property_name] = property_example

        if result:
            return result

        additional_properties = schema.get("additionalProperties")
        if isinstance(additional_properties, dict):
            extra_example = example_from_schema(
                spec,
                additional_properties,
                seen_refs,
            )
            return {"key": extra_example if extra_example is not None else "value"}
        return {}

    if schema_type == "array":
        item_schema = schema.get("items")
        if not item_schema:
            return []
        item_example = example_from_schema(spec, item_schema, seen_refs)
        return [item_example] if item_example is not None else []

    return scalar_example(schema)


def build_request_example(
    service: ServiceConfig,
    spec: dict[str, Any],
    method: str,
    path: str,
    operation: dict[str, Any],
) -> Any:
    override = MANUAL_REQUEST_EXAMPLES.get((service.slug, method, path))
    if override is not None:
        return override

    request_body = operation.get("requestBody") or {}
    request_content = request_body.get("content") or {}

    if "application/json" not in request_content:
        return None

    media_type = request_content["application/json"]
    if "example" in media_type:
        return copy.deepcopy(media_type["example"])
    if "examples" in media_type and media_type["examples"]:
        first_example = next(iter(media_type["examples"].values()))
        if isinstance(first_example, dict) and "value" in first_example:
            return copy.deepcopy(first_example["value"])
        return copy.deepcopy(first_example)

    return example_from_schema(spec, media_type.get("schema"))


def merge_parameters(
    path_item: dict[str, Any],
    operation: dict[str, Any],
) -> list[dict[str, Any]]:
    merged: OrderedDict[tuple[str, str], dict[str, Any]] = OrderedDict()
    for parameter in path_item.get("parameters", []) + operation.get("parameters", []):
        key = (parameter.get("in", ""), parameter.get("name", ""))
        merged[key] = parameter
    return list(merged.values())


def path_to_postman(path: str) -> str:
    return re.sub(r"{([^}]+)}", r"{{\1}}", path)


def query_value_for_param(parameter: dict[str, Any]) -> str:
    name = parameter["name"]
    schema = parameter.get("schema", {})
    if "default" in schema:
        return str(schema["default"])

    variable_name = {
        "q": "query",
        "limit": "limit",
        "category": "category",
        "name": "signal_name",
    }.get(name)
    if variable_name:
        return "{{" + variable_name + "}}"

    normalized = collection_variable_name(name)
    return "{{" + normalized + "}}"


def build_raw_url(
    service: ServiceConfig,
    path: str,
    parameters: list[dict[str, Any]],
) -> str:
    raw_url = "{{" + service.base_url_variable + "}}" + path_to_postman(path)
    query_pairs = []
    for parameter in parameters:
        if parameter.get("in") != "query":
            continue
        query_pairs.append(f"{parameter['name']}={query_value_for_param(parameter)}")

    if query_pairs:
        raw_url += "?" + "&".join(query_pairs)
    return raw_url


def build_request_description(operation: dict[str, Any]) -> str:
    parts = []
    summary = operation.get("summary")
    description = (operation.get("description") or "").strip()
    if summary:
        parts.append(summary)
    if description:
        parts.append(description)
    responses = operation.get("responses") or {}
    if responses:
        parts.append(
            "Documented responses: " + ", ".join(sorted(responses.keys()))
        )
    return "\n\n".join(parts)


def build_request_item(
    service: ServiceConfig,
    spec: dict[str, Any],
    path: str,
    method: str,
    operation: dict[str, Any],
    path_item: dict[str, Any],
) -> dict[str, Any]:
    parameters = merge_parameters(path_item, operation)
    raw_url = build_raw_url(service, path, parameters)
    request_example = build_request_example(service, spec, method, path, operation)

    headers = []
    body = None
    if request_example is not None:
        headers.append({"key": "Content-Type", "value": "application/json"})
        body = {
            "mode": "raw",
            "raw": json.dumps(request_example, indent=2, ensure_ascii=True),
            "options": {"raw": {"language": "json"}},
        }

    request = {
        "method": method.upper(),
        "header": headers,
        "url": raw_url,
        "description": build_request_description(operation),
    }
    if body is not None:
        request["body"] = body

    return {
        "name": operation.get("summary") or f"{method.upper()} {path}",
        "request": request,
        "response": [],
    }


def build_service_folder(
    service: ServiceConfig,
    spec: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    tags: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    endpoints: list[dict[str, Any]] = []

    for path, path_item in spec.get("paths", {}).items():
        for method, operation in path_item.items():
            if method not in HTTP_METHODS:
                continue

            tag_name = clean_tag((operation.get("tags") or ["General"])[0])
            tags.setdefault(tag_name, [])
            tags[tag_name].append(
                build_request_item(service, spec, path, method, operation, path_item)
            )

            endpoints.append(
                {
                    "method": method.upper(),
                    "path": path,
                    "summary": operation.get("summary") or f"{method.upper()} {path}",
                    "tags": operation.get("tags") or [],
                    "operation_id": operation.get("operationId"),
                }
            )

    service_items: list[dict[str, Any]] = []
    for tag_name, requests in tags.items():
        service_items.append(
            {
                "name": tag_name,
                "item": requests,
            }
        )

    folder = {
        "name": service.name,
        "description": (
            f"{service.description}\n\n"
            f"Base URL variable: {{{{{service.base_url_variable}}}}}"
        ),
        "item": service_items,
    }
    return folder, endpoints


def build_collection(services_with_specs: list[tuple[ServiceConfig, dict[str, Any]]]) -> dict[str, Any]:
    collection_items = []
    manifest_services = []
    discovered_variables = OrderedDict((name, value) for name, value in VARIABLE_DEFAULTS.items())

    for service, spec in services_with_specs:
        folder, endpoints = build_service_folder(service, spec)
        collection_items.append(folder)

        for endpoint in endpoints:
            for match in re.findall(r"{([^}]+)}", endpoint["path"]):
                discovered_variables.setdefault(match, f"replace-with-{match}")

        manifest_services.append(
            {
                "name": service.name,
                "slug": service.slug,
                "base_url_variable": service.base_url_variable,
                "default_base_url": service.default_base_url,
                "openapi_file": f"{service.slug}.openapi.json",
                "endpoint_count": len(endpoints),
                "endpoints": endpoints,
            }
        )

    variables = [
        {"key": key, "value": value}
        for key, value in discovered_variables.items()
    ]

    collection = {
        "info": {
            "name": "Stretus Services",
            "description": (
                "Combined Postman collection for the Stretus Trading API "
                "and Stretus Quant Engine.\n\n"
                "Default local ports:\n"
                "- Main API: {{api_base_url}}\n"
                "- Quant Engine: {{quant_base_url}}\n\n"
                "Suggested flow:\n"
                "1. Create a chat session.\n"
                "2. Send messages and poll chat history until an assistant "
                "message is completed.\n"
                "3. Confirm the strategy and capture strategy_id.\n"
                "4. Trigger a backtest and poll for results."
            ),
            "schema": POSTMAN_SCHEMA_URL,
        },
        "item": collection_items,
        "variable": variables,
    }

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "output_directory": str(OUTPUT_DIR.relative_to(ROOT)),
        "postman_collection_file": "stretus-services.postman_collection.json",
        "services": manifest_services,
    }

    return {
        "collection": collection,
        "manifest": manifest,
    }


def export_service_openapi(service: ServiceConfig) -> dict[str, Any]:
    spec = copy.deepcopy(service.app.openapi())
    spec["servers"] = [
        {
            "url": service.default_base_url,
            "description": "Local default",
        }
    ]
    return spec


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    services_with_specs: list[tuple[ServiceConfig, dict[str, Any]]] = []
    for service in SERVICES:
        spec = export_service_openapi(service)
        services_with_specs.append((service, spec))
        write_json(OUTPUT_DIR / f"{service.slug}.openapi.json", spec)

    generated = build_collection(services_with_specs)
    write_json(
        OUTPUT_DIR / "stretus-services.postman_collection.json",
        generated["collection"],
    )
    write_json(
        OUTPUT_DIR / "stretus-api-manifest.json",
        generated["manifest"],
    )

    for service, spec in services_with_specs:
        endpoint_count = sum(
            1
            for path_item in spec.get("paths", {}).values()
            for method in path_item
            if method in HTTP_METHODS
        )
        print(
            f"Generated {service.slug}.openapi.json with {endpoint_count} endpoints"
        )

    print("Generated stretus-services.postman_collection.json")
    print("Generated stretus-api-manifest.json")


if __name__ == "__main__":
    main()
