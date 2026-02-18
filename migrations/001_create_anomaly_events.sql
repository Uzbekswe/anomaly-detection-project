-- Migration 001: Create anomaly_events table
-- Database: anomaly_db (PostgreSQL 15+)
-- Run: python migrations/run.py
-- Transaction control is handled by the migration runner (run.py)

-- Table: anomaly_events
CREATE TABLE IF NOT EXISTS anomaly_events (
    id              SERIAL PRIMARY KEY,
    sensor_id       VARCHAR(64) NOT NULL,
    detected_at     TIMESTAMPTZ NOT NULL,
    anomaly_score   FLOAT NOT NULL,
    is_anomaly      BOOLEAN NOT NULL,
    confidence      FLOAT NOT NULL,
    model_version   VARCHAR(128) NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_anomaly_events_sensor_id
    ON anomaly_events(sensor_id);

CREATE INDEX IF NOT EXISTS idx_anomaly_events_detected_at
    ON anomaly_events(detected_at);

-- Migration tracking
CREATE TABLE IF NOT EXISTS schema_migrations (
    version     INTEGER PRIMARY KEY,
    applied_at  TIMESTAMPTZ DEFAULT NOW(),
    description TEXT NOT NULL
);

INSERT INTO schema_migrations (version, description)
VALUES (1, 'Create anomaly_events table')
ON CONFLICT (version) DO NOTHING;
