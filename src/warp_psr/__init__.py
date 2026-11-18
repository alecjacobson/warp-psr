"""GPU Poisson and screened Poisson surface reconstruction in NVIDIA Warp."""
from .reconstruction import reconstruct, Reconstruction
from .field import LevelField


def __getattr__(name):
    # Keep optional device-handle definitions out of ordinary reconstruction imports.
    if name in ('Field', 'Field64', 'evaluate_field'):
        from .device_field import Field, Field64, evaluate_field
        exports = dict(Field=Field, Field64=Field64, evaluate_field=evaluate_field)
        globals().update(exports)
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def reconstruct_differentiable(*args, **kwargs):
    """Opt-in finite reconstruction VJP; call outside the supplied tape context."""
    from .reconstruction_adjoint import reconstruct_differentiable as implementation
    return implementation(*args, **kwargs)


__all__ = ['reconstruct','reconstruct_differentiable','Reconstruction','LevelField',
           'Field','Field64','evaluate_field']
