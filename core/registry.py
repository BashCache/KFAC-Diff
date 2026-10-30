from typing import Dict, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from core.base_pruner import BasePruner

class PrunerRegistry:
    _registry: Dict[str, Type["BasePruner"]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a pruning method by name."""
        def decorator(pruner_cls: Type["BasePruner"]):
            if name in cls._registry:
                raise ValueError(f"Pruner '{name}' is already registered.")
            cls._registry[name] = pruner_cls
            return pruner_cls
        return decorator

    @classmethod
    def get(cls, name: str) -> Type["BasePruner"]:
        if name not in cls._registry:
            raise ValueError(
                f"Unknown pruning method: '{name}'. "
                f"Available methods: {list(cls._registry.keys())}"
            )
        return cls._registry[name]

    @classmethod
    def available(cls) -> list:
        return list(cls._registry.keys())

# Module-level singleton
registry = PrunerRegistry()