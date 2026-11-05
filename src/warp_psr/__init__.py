"""GPU Poisson and screened Poisson surface reconstruction in NVIDIA Warp."""
from .reconstruction import reconstruct, Reconstruction
from .field import LevelField

def reconstruct_differentiable(*args, **kwargs):
    """Opt-in finite reconstruction VJP; call outside the supplied tape context."""
    from .reconstruction_adjoint import reconstruct_differentiable as implementation
    return implementation(*args, **kwargs)


__all__ = ['reconstruct','reconstruct_differentiable','Reconstruction','LevelField']
