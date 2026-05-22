"""ModelStore — SQLite persistence for Provider, Endpoint, and Model configurations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from cliver.db import SQLiteStore
from cliver.model.models import Provider, Endpoint, Model, _now

_SCHEMA = """
CREATE TABLE IF NOT EXISTS providers (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    type TEXT NOT NULL DEFAULT 'openai',
    api_key TEXT,
    rate_limit TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS endpoints (
    id TEXT PRIMARY KEY,
    provider_id TEXT NOT NULL REFERENCES providers(id) ON DELETE CASCADE,
    base_url TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(provider_id, base_url)
);

CREATE TABLE IF NOT EXISTS models (
    id TEXT PRIMARY KEY,
    provider_id TEXT NOT NULL REFERENCES providers(id) ON DELETE CASCADE,
    endpoint_id TEXT NOT NULL REFERENCES endpoints(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    capabilities TEXT,
    options TEXT,
    think_mode INTEGER,
    context_window INTEGER,
    pricing TEXT,
    is_default INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(provider_id, name)
);
"""


def _json(obj: Any) -> str:
    """Serialize a Python object to a JSON string."""
    if obj is None:
        return "null"
    return json.dumps(obj, ensure_ascii=False)


def _parse_json(value: Any) -> Any:
    """Parse a JSON string back to a Python object."""
    if value is None:
        return None
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value
    return value


class ModelStore:
    """CRUD store for providers, endpoints, and models.

    Instantiate with a db_path, or use ``ModelStore.from_config_dir(config_dir)``
    to resolve the default ``cliver.db`` path automatically.
    """

    def __init__(self, db_path: Path):
        self._db_path = Path(db_path)
        self._store: Optional[SQLiteStore] = None

    @classmethod
    def from_config_dir(cls, config_dir: Path) -> "ModelStore":
        """Create a store using the default database path for a config directory."""
        return cls(config_dir / "cliver.db")

    def _get_store(self) -> SQLiteStore:
        if self._store is None:
            self._store = SQLiteStore(self._db_path)
            self._store.execute_schema(_SCHEMA)
        return self._store

    # -- Provider CRUD ---------------------------------------------------------

    def create_provider(
        self,
        name: str,
        type: str = "openai",
        api_key: Optional[str] = None,
        rate_limit: Optional[Dict[str, Any]] = None,
    ) -> Provider:
        """Create a new provider."""
        provider = Provider(
            name=name,
            type=type,
            api_key=api_key,
            rate_limit=rate_limit,
        )
        with self._get_store().write() as db:
            db.execute(
                "INSERT INTO providers (id, name, type, api_key, rate_limit, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    provider.id,
                    provider.name,
                    provider.type,
                    provider.api_key,
                    _json(provider.rate_limit),
                    provider.created_at,
                    provider.updated_at,
                ),
            )
        return provider

    def list_providers(self) -> List[Provider]:
        """List all providers ordered by most recently updated."""
        with self._get_store().read() as db:
            rows = db.execute(
                "SELECT id, name, type, api_key, rate_limit, created_at, updated_at "
                "FROM providers ORDER BY updated_at DESC"
            ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["rate_limit"] = _parse_json(d.get("rate_limit"))
            result.append(Provider(**d))
        return result

    def get_provider(self, provider_id: str) -> Optional[Provider]:
        """Get a provider by id, or None if not found."""
        with self._get_store().read() as db:
            row = db.execute(
                "SELECT id, name, type, api_key, rate_limit, created_at, updated_at "
                "FROM providers WHERE id = ?",
                (provider_id,),
            ).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["rate_limit"] = _parse_json(d.get("rate_limit"))
        return Provider(**d)

    def update_provider(
        self,
        provider_id: str,
        name: Optional[str] = None,
        type: Optional[str] = None,
        api_key: Optional[str] = None,
        rate_limit: Any = None,
    ) -> Optional[Provider]:
        """Update a provider's fields. ``None`` fields are left unchanged.

        Args:
            provider_id: The id of the provider to update.
            name: New name, or None to leave unchanged.
            type: New type, or None to leave unchanged.
            api_key: New api_key, or None to leave unchanged.
            rate_limit: Pass ``_json.dumps(...)`` to set, or None to leave unchanged.
        """
        existing = self.get_provider(provider_id)
        if existing is None:
            return None
        if name is not None:
            existing.name = name
        if type is not None:
            existing.type = type
        if api_key is not None:
            existing.api_key = api_key
        if rate_limit is not None:
            existing.rate_limit = rate_limit

        existing.updated_at = _now()
        with self._get_store().write() as db:
            db.execute(
                "UPDATE providers SET name=?, type=?, api_key=?, rate_limit=?, updated_at=? "
                "WHERE id=?",
                (
                    existing.name,
                    existing.type,
                    existing.api_key,
                    _json(existing.rate_limit),
                    existing.updated_at,
                    provider_id,
                ),
            )
        return existing

    def delete_provider(self, provider_id: str) -> bool:
        """Delete a provider and all its endpoints and models (CASCADE)."""
        with self._get_store().write() as db:
            cursor = db.execute("DELETE FROM providers WHERE id = ?", (provider_id,))
        return cursor.rowcount > 0

    # -- Endpoint CRUD ---------------------------------------------------------

    def create_endpoint(
        self,
        provider_id: str,
        base_url: str,
    ) -> Endpoint:
        """Create a new endpoint for a provider."""
        endpoint = Endpoint(
            provider_id=provider_id,
            base_url=base_url,
        )
        with self._get_store().write() as db:
            db.execute(
                "INSERT INTO endpoints (id, provider_id, base_url, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    endpoint.id,
                    endpoint.provider_id,
                    endpoint.base_url,
                    endpoint.created_at,
                    endpoint.updated_at,
                ),
            )
        return endpoint

    def list_endpoints(self, provider_id: Optional[str] = None) -> List[Endpoint]:
        """List endpoints, optionally filtered by provider_id."""
        with self._get_store().read() as db:
            if provider_id:
                rows = db.execute(
                    "SELECT id, provider_id, base_url, created_at, updated_at "
                    "FROM endpoints WHERE provider_id = ? ORDER BY updated_at DESC",
                    (provider_id,),
                ).fetchall()
            else:
                rows = db.execute(
                    "SELECT id, provider_id, base_url, created_at, updated_at "
                    "FROM endpoints ORDER BY updated_at DESC"
                ).fetchall()
        return [Endpoint(**dict(r)) for r in rows]

    def get_endpoint(self, endpoint_id: str) -> Optional[Endpoint]:
        """Get an endpoint by id, or None if not found."""
        with self._get_store().read() as db:
            row = db.execute(
                "SELECT id, provider_id, base_url, created_at, updated_at "
                "FROM endpoints WHERE id = ?",
                (endpoint_id,),
            ).fetchone()
        if row is None:
            return None
        return Endpoint(**dict(row))

    def update_endpoint(
        self,
        endpoint_id: str,
        base_url: Optional[str] = None,
    ) -> Optional[Endpoint]:
        """Update an endpoint's fields."""
        existing = self.get_endpoint(endpoint_id)
        if existing is None:
            return None
        if base_url is not None:
            existing.base_url = base_url
        existing.updated_at = _now()
        with self._get_store().write() as db:
            db.execute(
                "UPDATE endpoints SET base_url=?, updated_at=? WHERE id=?",
                (
                    existing.base_url,
                    existing.updated_at,
                    endpoint_id,
                ),
            )
        return existing

    def delete_endpoint(self, endpoint_id: str) -> bool:
        """Delete an endpoint and its models (CASCADE)."""
        with self._get_store().write() as db:
            cursor = db.execute("DELETE FROM endpoints WHERE id = ?", (endpoint_id,))
        return cursor.rowcount > 0

    # -- Model CRUD ------------------------------------------------------------

    @staticmethod
    def _parse_model_row(row) -> Model:
        """Convert a sqlite3.Row to a Model instance, deserializing JSON fields."""
        d = dict(row)
        d["capabilities"] = _parse_json(d.get("capabilities")) or []
        d["options"] = _parse_json(d.get("options")) or {}
        d["pricing"] = _parse_json(d.get("pricing"))
        return Model(**d)

    def create_model(
        self,
        provider_id: str,
        endpoint_id: str,
        name: str,
        capabilities: Optional[List[str]] = None,
        options: Optional[Dict[str, Any]] = None,
        think_mode: Optional[int] = None,
        context_window: Optional[int] = None,
        pricing: Optional[Dict[str, Any]] = None,
        is_default: int = 0,
    ) -> Model:
        """Create a new model. If ``is_default=1``, clears other defaults first."""
        if is_default == 1:
            self._clear_default_model()
        model = Model(
            provider_id=provider_id,
            endpoint_id=endpoint_id,
            name=name,
            capabilities=capabilities or [],
            options=options or {},
            think_mode=think_mode,
            context_window=context_window,
            pricing=pricing,
            is_default=is_default,
        )
        with self._get_store().write() as db:
            db.execute(
                "INSERT INTO models (id, provider_id, endpoint_id, name, capabilities, options, "
                "think_mode, context_window, pricing, is_default, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    model.id,
                    model.provider_id,
                    model.endpoint_id,
                    model.name,
                    _json(model.capabilities),
                    _json(model.options),
                    model.think_mode,
                    model.context_window,
                    _json(model.pricing),
                    model.is_default,
                    model.created_at,
                    model.updated_at,
                ),
            )
        return model

    def list_models(self, capability: Optional[str] = None) -> List[Model]:
        """List all models, optionally filtered by a capability name."""
        with self._get_store().read() as db:
            if capability:
                rows = db.execute(
                    "SELECT id, provider_id, endpoint_id, name, capabilities, options, "
                    "think_mode, context_window, pricing, is_default, created_at, updated_at "
                    "FROM models WHERE capabilities LIKE ? ORDER BY updated_at DESC",
                    (f"%{capability}%",),
                ).fetchall()
            else:
                rows = db.execute(
                    "SELECT id, provider_id, endpoint_id, name, capabilities, options, "
                    "think_mode, context_window, pricing, is_default, created_at, updated_at "
                    "FROM models ORDER BY updated_at DESC"
                ).fetchall()
        return [self._parse_model_row(r) for r in rows]

    def get_model(self, model_id: str) -> Optional[Model]:
        """Get a model by id, or None if not found."""
        with self._get_store().read() as db:
            row = db.execute(
                "SELECT id, provider_id, endpoint_id, name, capabilities, options, "
                "think_mode, context_window, pricing, is_default, created_at, updated_at "
                "FROM models WHERE id = ?",
                (model_id,),
            ).fetchone()
        if row is None:
            return None
        return self._parse_model_row(row)

    def update_model(
        self,
        model_id: str,
        name: Optional[str] = None,
        capabilities: Optional[List[str]] = None,
        options: Optional[Dict[str, Any]] = None,
        think_mode: Optional[int] = None,
        context_window: Optional[int] = None,
        pricing: Optional[Dict[str, Any]] = None,
        is_default: Optional[int] = None,
    ) -> Optional[Model]:
        """Update a model's fields. ``None`` fields are left unchanged."""
        existing = self.get_model(model_id)
        if existing is None:
            return None
        if name is not None:
            existing.name = name
        if capabilities is not None:
            existing.capabilities = capabilities
        if options is not None:
            existing.options = options
        if think_mode is not None:
            existing.think_mode = think_mode
        if context_window is not None:
            existing.context_window = context_window
        if pricing is not None:
            existing.pricing = pricing
        if is_default is not None:
            if is_default == 1:
                self._clear_default_model()
            existing.is_default = is_default

        existing.updated_at = _now()
        with self._get_store().write() as db:
            db.execute(
                "UPDATE models SET name=?, capabilities=?, options=?, think_mode=?, "
                "context_window=?, pricing=?, is_default=?, updated_at=? WHERE id=?",
                (
                    existing.name,
                    _json(existing.capabilities),
                    _json(existing.options),
                    existing.think_mode,
                    existing.context_window,
                    _json(existing.pricing),
                    existing.is_default,
                    existing.updated_at,
                    model_id,
                ),
            )
        return existing

    def delete_model(self, model_id: str) -> bool:
        """Delete a model by id."""
        with self._get_store().write() as db:
            cursor = db.execute("DELETE FROM models WHERE id = ?", (model_id,))
        return cursor.rowcount > 0

    def set_default_model(self, model_id: str) -> Optional[Model]:
        """Set a model as the default, clearing any other defaults."""
        model = self.get_model(model_id)
        if model is None:
            return None
        self._clear_default_model()
        model.is_default = 1
        model.updated_at = _now()
        with self._get_store().write() as db:
            db.execute(
                "UPDATE models SET is_default=?, updated_at=? WHERE id=?",
                (model.is_default, model.updated_at, model_id),
            )
        return model

    def _clear_default_model(self) -> None:
        """Clear the default flag on all models."""
        with self._get_store().write() as db:
            db.execute("UPDATE models SET is_default=0 WHERE is_default=1")

    def get_default_model(self) -> Optional[Model]:
        """Get the default model (is_default=1), or None if none is set."""
        with self._get_store().read() as db:
            row = db.execute(
                "SELECT id, provider_id, endpoint_id, name, capabilities, options, "
                "think_mode, context_window, pricing, is_default, created_at, updated_at "
                "FROM models WHERE is_default = 1 LIMIT 1",
            ).fetchone()
        if row is None:
            return None
        return self._parse_model_row(row)

    def close(self) -> None:
        """Close the database connection."""
        if self._store:
            self._store.close()
            self._store = None
