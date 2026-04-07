from telescope.backends.base import ReadBackend

# Concrete backends are NOT eagerly imported here — that would force loading
# the neo4j driver even when STORAGE_BACKEND=postgres is set, defeating the
# factory's deferred-import design. Import them directly from their modules.

__all__ = ["ReadBackend"]
