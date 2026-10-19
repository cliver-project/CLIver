"""In-process local models for embeddings and reranking.

Importing this package is cheap: torch, sentence-transformers, and
FlagEmbedding are imported only when a model is actually loaded.  Install a
local extra to get a built-in backend:

* ``pip install "cliver-core[local-st]"`` — sentence-transformers backend;
* ``pip install "cliver-core[local-flag]"`` — FlagEmbedding backend;
* ``pip install "cliver-core[local]"`` — both.

Quick start::

    from cliver_core.capabilities import embed
    from cliver_core.local import LocalEmbeddingProvider, local_embed

    provider = LocalEmbeddingProvider(
        model="BAAI/bge-base-zh-v1.5",       # HF id or local directory
        backend="sentence-transformers",     # explicit, never inferred
    )
    binding = local_embed(provider=provider, normalize=True)

    result = await embed(texts=["hello", "world"], binding=binding)
    # ...
    await provider.close()

The provider is long-lived and owns the loaded model.  There is no
process-global model cache; share the provider object instead.  The
application is responsible for ``await provider.close()``.
"""

from cliver_core.local.backends import (
    EmbeddingBackend,
    LoadedEmbedding,
    LoadedRerank,
    LoadSpec,
    RerankBackend,
    register_backend,
    resolve_backend,
)
from cliver_core.local.bindings import local_embed, local_rerank
from cliver_core.local.errors import (
    LocalBackendNotSupportedError,
    LocalConfigurationError,
    LocalDependencyError,
    LocalDeviceError,
    LocalDimensionError,
    LocalInferenceError,
    LocalModelError,
    LocalModelLoadError,
    LocalProviderClosedError,
)
from cliver_core.local.info import LocalModelInfo
from cliver_core.local.providers import (
    LocalEmbeddingProvider,
    LocalProvider,
    LocalRerankProvider,
)

__all__ = [
    "EmbeddingBackend",
    "LoadSpec",
    "LoadedEmbedding",
    "LoadedRerank",
    "LocalBackendNotSupportedError",
    "LocalConfigurationError",
    "LocalDependencyError",
    "LocalDeviceError",
    "LocalDimensionError",
    "LocalEmbeddingProvider",
    "LocalInferenceError",
    "LocalModelError",
    "LocalModelInfo",
    "LocalModelLoadError",
    "LocalProvider",
    "LocalProviderClosedError",
    "LocalRerankProvider",
    "RerankBackend",
    "local_embed",
    "local_rerank",
    "register_backend",
    "resolve_backend",
]
