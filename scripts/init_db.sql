-- ============================================================
--  Stretus – Complete Database Initialisation  (v2 — session_id)
--  Run once on a fresh PostgreSQL 16 instance.
--  Docker mounts this at /docker-entrypoint-initdb.d/01_init.sql
--  For manual setup: psql -U stretus -d stretus -f scripts/init_db.sql
-- ============================================================

CREATE SCHEMA IF NOT EXISTS strategy AUTHORIZATION stretus;

GRANT USAGE  ON SCHEMA ai_strategy TO stretus;
GRANT CREATE ON SCHEMA ai_strategy TO stretus;

ALTER DEFAULT PRIVILEGES IN SCHEMA ai_strategy
    GRANT ALL ON TABLES    TO stretus;
ALTER DEFAULT PRIVILEGES IN SCHEMA ai_strategy
    GRANT ALL ON SEQUENCES TO stretus;

-- ── ENUM types ────────────────────────────────────────────────
DO $$ BEGIN
    CREATE TYPE ai_strategy.chat_role AS ENUM ('user', 'assistant', 'system');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TYPE ai_strategy.strategy_status AS ENUM (
        'draft', 'confirmed', 'backtesting',
        'backtest_complete', 'live', 'paused', 'archived'
    );
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TYPE ai_strategy.backtest_status AS ENUM (
        'pending', 'running', 'completed', 'failed'
    );
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TYPE ai_strategy.message_status AS ENUM (
        'pending', 'processing', 'completed', 'failed'
    );
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- ── Trigger function ──────────────────────────────────────────
CREATE OR REPLACE FUNCTION ai_strategy.set_updated_at()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$;

-- ═══════════════════════════════════════════════════
--  TABLE: ai_strategy.chats
-- ═══════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS ai_strategy.chats (
    id         uuid        NOT NULL DEFAULT gen_random_uuid(),
    user_id    uuid,
    title      text,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT chats_pkey PRIMARY KEY (id)
) TABLESPACE pg_default;

ALTER TABLE IF EXISTS ai_strategy.chats OWNER TO stretus;

CREATE INDEX IF NOT EXISTS idx_chats_user_id
    ON ai_strategy.chats (user_id)
    WHERE user_id IS NOT NULL;

DROP TRIGGER IF EXISTS trg_chats_updated_at ON ai_strategy.chats;
CREATE TRIGGER trg_chats_updated_at
    BEFORE UPDATE ON ai_strategy.chats
    FOR EACH ROW EXECUTE FUNCTION ai_strategy.set_updated_at();

-- ═══════════════════════════════════════════════════
--  TABLE: ai_strategy.strategies   (session_id — not chat_id)
-- ═══════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS ai_strategy.strategies (
    id              uuid                     NOT NULL DEFAULT gen_random_uuid(),
    session_id      uuid                     NOT NULL,
    name            text                     NOT NULL,
    symbol          text                     NOT NULL,
    market          text                     NOT NULL,
    timeframe       text                     NOT NULL DEFAULT '1d',
    status          ai_strategy.strategy_status NOT NULL DEFAULT 'draft',
    yaml_path       text,
    strategy_config jsonb,
    created_at      timestamptz              NOT NULL DEFAULT now(),
    updated_at      timestamptz              NOT NULL DEFAULT now(),
    CONSTRAINT strategies_pkey PRIMARY KEY (id),
    CONSTRAINT strategies_session_id_fkey FOREIGN KEY (session_id)
        REFERENCES ai_strategy.chats (id)
        ON UPDATE NO ACTION ON DELETE CASCADE
) TABLESPACE pg_default;

ALTER TABLE IF EXISTS ai_strategy.strategies OWNER TO stretus;

CREATE INDEX IF NOT EXISTS idx_strategies_session_id
    ON ai_strategy.strategies (session_id);
CREATE INDEX IF NOT EXISTS idx_strategies_status
    ON ai_strategy.strategies (status);
CREATE INDEX IF NOT EXISTS idx_strategies_symbol
    ON ai_strategy.strategies (symbol);

DROP TRIGGER IF EXISTS trg_strategies_updated_at ON ai_strategy.strategies;
CREATE TRIGGER trg_strategies_updated_at
    BEFORE UPDATE ON ai_strategy.strategies
    FOR EACH ROW EXECUTE FUNCTION ai_strategy.set_updated_at();

-- ═══════════════════════════════════════════════════
--  TABLE: ai_strategy.backtest
-- ═══════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS ai_strategy.backtest (
    id            uuid                     NOT NULL DEFAULT gen_random_uuid(),
    strategy_id   uuid                     NOT NULL,
    status        ai_strategy.backtest_status NOT NULL DEFAULT 'pending',
    result_json   jsonb,
    error_message text,
    started_at    timestamptz,
    completed_at  timestamptz,
    created_at    timestamptz              NOT NULL DEFAULT now(),
    CONSTRAINT backtest_pkey PRIMARY KEY (id),
    CONSTRAINT backtest_strategy_id_fkey FOREIGN KEY (strategy_id)
        REFERENCES ai_strategy.strategies (id)
        ON UPDATE NO ACTION ON DELETE CASCADE
) TABLESPACE pg_default;

ALTER TABLE IF EXISTS ai_strategy.backtest OWNER TO stretus;

CREATE INDEX IF NOT EXISTS idx_backtest_strategy_id
    ON ai_strategy.backtest (strategy_id);
CREATE INDEX IF NOT EXISTS idx_backtest_status
    ON ai_strategy.backtest (status);

-- ═══════════════════════════════════════════════════
--  TABLE: ai_strategy.chat_messages  (session_id + new async columns)
-- ═══════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS ai_strategy.chat_messages (
    id                uuid                     NOT NULL DEFAULT gen_random_uuid(),
    session_id        uuid                     NOT NULL,
    role              ai_strategy.chat_role       NOT NULL,
    content           text                     NOT NULL,
    status            ai_strategy.message_status  NOT NULL DEFAULT 'completed',
    updated_at        timestamptz              NOT NULL DEFAULT now(),
    error_message     text,
    parent_message_id uuid,
    is_final          boolean                  NOT NULL DEFAULT true,
    token_count       integer,
    model             text,
    strategy_draft    jsonb,
    backtest_id       uuid,
    strategy_id       uuid,
    ref_backtest_id   text,
    strategy_json     jsonb,
    created_at        timestamptz              NOT NULL DEFAULT now(),
    CONSTRAINT chat_messages_pkey PRIMARY KEY (id),
    CONSTRAINT chat_messages_session_id_fkey FOREIGN KEY (session_id)
        REFERENCES ai_strategy.chats (id)
        ON UPDATE NO ACTION ON DELETE CASCADE,
    CONSTRAINT chat_messages_backtest_id_fkey FOREIGN KEY (backtest_id)
        REFERENCES ai_strategy.backtest (id)
        ON UPDATE NO ACTION ON DELETE SET NULL,
    CONSTRAINT chat_messages_strategy_id_fkey FOREIGN KEY (strategy_id)
        REFERENCES ai_strategy.strategies (id)
        ON UPDATE NO ACTION ON DELETE SET NULL,
    CONSTRAINT chat_messages_parent_message_id_fkey FOREIGN KEY (parent_message_id)
        REFERENCES ai_strategy.chat_messages (id)
        ON UPDATE NO ACTION ON DELETE SET NULL
) TABLESPACE pg_default;

ALTER TABLE IF EXISTS ai_strategy.chat_messages OWNER TO stretus;

COMMENT ON COLUMN ai_strategy.chat_messages.strategy_json
    IS 'Parsed strategy_json from assemble_strategy AI response (entry/exit signals, asset, parameters)';

CREATE INDEX IF NOT EXISTS idx_chat_messages_chat
    ON ai_strategy.chat_messages (session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_chat_messages_role
    ON ai_strategy.chat_messages (role);
CREATE INDEX IF NOT EXISTS idx_chat_messages_status
    ON ai_strategy.chat_messages (status);
CREATE INDEX IF NOT EXISTS idx_chat_messages_backtest_id
    ON ai_strategy.chat_messages (backtest_id)
    WHERE backtest_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_chat_messages_strategy_id
    ON ai_strategy.chat_messages (strategy_id)
    WHERE strategy_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_chat_messages_ref_backtest_id
    ON ai_strategy.chat_messages (ref_backtest_id)
    WHERE ref_backtest_id IS NOT NULL;

DROP TRIGGER IF EXISTS trg_chat_messages_updated_at ON ai_strategy.chat_messages;
CREATE TRIGGER trg_chat_messages_updated_at
    BEFORE UPDATE ON ai_strategy.chat_messages
    FOR EACH ROW EXECUTE FUNCTION ai_strategy.set_updated_at();

-- ── Done ──────────────────────────────────────────────────────
DO $$ BEGIN
    RAISE NOTICE '✅ Stretus schema v2 (session_id) initialised successfully.';
END $$;