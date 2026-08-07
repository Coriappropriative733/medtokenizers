"""Utility functions for medtokenizers modules.

This module provides foundational building blocks used across the tokenization
pipeline, including differentiable rounding, numerical utilities, and
normalization layers.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import TYPE_CHECKING, TypeVar

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from jaxtyping import Float

F_co = TypeVar("F_co", bound=Callable)


def _is_compiling() -> bool:
    """Check if we're inside torch.compile/dynamo tracing."""
    try:
        return torch.compiler.is_compiling()
    except AttributeError:
        # Fallback for older PyTorch versions
        try:
            return torch._dynamo.is_compiling()  # type: ignore[attr-defined]
        except AttributeError:
            return False


def jaxtyped_compile_safe(typechecker: Callable) -> Callable[[F_co], F_co]:
    """Jaxtyping decorator that skips type checking during torch.compile.

    When torch.dynamo traces a function for compilation, beartype's runtime
    type checking can fail because dynamo uses proxy objects that don't
    satisfy the type annotations (even though the actual tensors do).

    This wrapper detects compilation context and bypasses type checking,
    while still applying it during normal eager execution.

    Usage:
        from medtokenizers.modules.utils import jaxtyped_compile_safe
        from beartype import beartype

        @jaxtyped_compile_safe(beartype)
        def forward(self, x: Float[torch.Tensor, "batch channels *spatial"]) -> ...:
            ...
    """
    from jaxtyping import jaxtyped

    def decorator(fn: F_co) -> F_co:
        # Apply jaxtyped with beartype for eager mode
        typed_fn = jaxtyped(typechecker=typechecker)(fn)

        @wraps(fn)
        def wrapper(*args, **kwargs):
            if _is_compiling():
                # Skip type checking during compilation
                return fn(*args, **kwargs)
            return typed_fn(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


class _RoundSTE(torch.autograd.Function):
    """Straight-Through Estimator for rounding with AMP-safe gradients.

    The straight-through estimator (STE) is a cornerstone technique for training
    discrete representations. It solves a fundamental problem: rounding is
    non-differentiable (gradient is zero almost everywhere), yet we need
    gradients to flow through quantization during backpropagation.

    The STE Solution
    ----------------
    Forward pass: y = round(x)  [discrete output]
    Backward pass: dy/dx = 1    [pretend rounding was identity]

    This "lie" works remarkably well in practice because:
    1. The gradient direction is preserved (sign of error is correct)
    2. Small input changes still produce proportional gradient signals
    3. The network learns to produce inputs that round to good codes

    AMP Stability
    -------------
    Under automatic mixed precision (float16), gradients can explode due to
    the limited dynamic range. This implementation clips gradients to prevent
    NaN/Inf propagation during training.

    References:
        Bengio et al. "Estimating or Propagating Gradients Through Stochastic
        Neurons for Conditional Computation" (2013)
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        """Round input to nearest integer."""
        return x.round()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """Pass gradients through with clipping for AMP stability.

        Clipping bounds are set to ±1e4 which is well within float16 range
        (max ~65504) while being large enough to not impede normal training.
        """
        return grad_output.clamp(-1e4, 1e4)


def round_ste(z: Float[torch.Tensor, ...]) -> Float[torch.Tensor, ...]:
    """Round tensor with straight-through gradient estimator.

    This is the primary quantization primitive used by FSQ and related methods.
    During forward pass, values are rounded to nearest integers. During backward
    pass, gradients flow through unchanged (identity gradient).

    Args:
        z: Input tensor of any shape. Values will be rounded to nearest integer.

    Returns:
        Rounded tensor with same shape and dtype as input. Gradients will
        flow through as if this operation were the identity function.

    Example:
        >>> x = torch.tensor([1.2, 2.7, -0.3], requires_grad=True)
        >>> y = round_ste(x)  # tensor([1., 3., -0.])
        >>> y.sum().backward()
        >>> x.grad  # tensor([1., 1., 1.]) -- gradient is 1 everywhere (STE)

    Note:
        For numerical stability under AMP, gradients are clipped to ±1e4.
        This prevents overflow in float16 while preserving gradient direction.
    """
    return _RoundSTE.apply(z)


def log(t: Float[torch.Tensor, ...], eps: float = 1e-5) -> Float[torch.Tensor, ...]:
    """Numerically stable logarithm with floor clamping.

    Prevents log(0) = -inf by clamping input to minimum epsilon value.
    Essential for entropy computations where probabilities may be zero.

    Args:
        t: Input tensor (typically probabilities in [0, 1])
        eps: Minimum value to clamp to before log. Default 1e-5.

    Returns:
        log(max(t, eps)) - safe logarithm of input
    """
    return t.clamp(min=eps).log()


def entropy(prob: Float[torch.Tensor, "... K"]) -> Float[torch.Tensor, ...]:
    """Compute Shannon entropy of probability distribution.

    H(p) = -Σ p_i * log(p_i)

    Used in codebook utilization metrics and entropy-regularized quantization.
    Higher entropy indicates more uniform codebook usage.

    Args:
        prob: Probability distribution over last dimension. Should sum to 1.

    Returns:
        Entropy value(s) with last dimension reduced.

    Example:
        >>> p_uniform = torch.tensor([0.25, 0.25, 0.25, 0.25])
        >>> entropy(p_uniform)  # Maximum entropy
        tensor(1.3863)
        >>> p_peaked = torch.tensor([0.97, 0.01, 0.01, 0.01])
        >>> entropy(p_peaked)  # Low entropy
        tensor(0.1545)
    """
    return (-prob * log(prob)).sum(dim=-1)


def nonlinearity(x: Float[torch.Tensor, ...]) -> Float[torch.Tensor, ...]:
    """Swish/SiLU activation function: x * sigmoid(x).

    The Swish activation (also called SiLU) provides smooth, non-monotonic
    activation that often outperforms ReLU in deep networks. It's self-gated,
    meaning the input modulates its own activation.

    Properties:
    - Smooth and differentiable everywhere
    - Non-monotonic (has a global minimum at x ≈ -1.28)
    - Unbounded above, bounded below (min ≈ -0.28)
    - Approximately linear for large positive x

    Args:
        x: Input tensor of any shape

    Returns:
        x * sigmoid(x), same shape as input

    Note:
        Uses F.silu() which has optimized CUDA kernels and enables
        better fusion with torch.compile().
    """
    return F.silu(x)


def Normalize(in_channels: int, num_groups: int = 32) -> nn.GroupNorm:
    """Create GroupNorm layer with automatic group count adjustment.

    GroupNorm normalizes over groups of channels, providing batch-size
    independent normalization. This is crucial for medical imaging where
    batch sizes are often 1-2 due to large 3D volumes.

    The function automatically adjusts num_groups to be a divisor of
    in_channels, ensuring valid GroupNorm configuration.

    Args:
        in_channels: Number of input channels
        num_groups: Target number of groups (will be reduced if needed)

    Returns:
        GroupNorm layer with adjusted num_groups

    Example:
        >>> norm = Normalize(64)   # 32 groups of 2 channels each
        >>> norm = Normalize(48)   # Adjusted to 16 groups of 3 channels
        >>> norm = Normalize(17)   # Adjusted to 1 group (17 is prime)
    """
    num_groups = min(num_groups, in_channels)
    while in_channels % num_groups != 0:
        num_groups -= 1
    return nn.GroupNorm(
        num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True
    )


def validate_tensor_input(
    x: torch.Tensor,
    expected_dim: int,
    expected_channels: int,
    name: str = "encode",
) -> None:
    """Validate input tensor shape, dtype, and properties at API boundaries.

    Provides clear, actionable error messages for common mistakes. Use this
    at encode/decode entry points to fail fast with helpful diagnostics.

    Args:
        x: Input tensor to validate
        expected_dim: Expected spatial dimensionality (2 or 3)
        expected_channels: Expected number of input channels
        name: Name of the calling method (for error messages)

    Raises:
        TypeError: If x is not a tensor or has wrong dtype
        ValueError: If x has wrong shape, dimensions, or contains NaN/Inf

    Example:
        >>> validate_tensor_input(x, expected_dim=3, expected_channels=1, name="encode")
        >>> # Raises ValueError if x is not (B, 1, H, W, D) floating point tensor
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError(
            f"{name}() expected torch.Tensor, got {type(x).__name__}. "
            "Convert your input to a tensor first."
        )

    # Check dtype is floating point
    if not x.is_floating_point():
        raise TypeError(
            f"{name}() expected floating point tensor, got {x.dtype}. "
            "Convert with x.float() or x.to(torch.float32)."
        )

    # Check number of dimensions
    expected_ndim = expected_dim + 2  # batch + channels + spatial
    if x.ndim != expected_ndim:
        raise ValueError(
            f"{name}() expected {expected_ndim}D tensor (batch, channels, *spatial), "
            f"got {x.ndim}D tensor with shape {tuple(x.shape)}. "
            f"For {expected_dim}D spatial, input should be "
            f"{'(B, C, H, W)' if expected_dim == 2 else '(B, C, H, W, D)'}."
        )

    # Check channel count
    actual_channels = x.shape[1]
    if actual_channels != expected_channels:
        raise ValueError(
            f"{name}() expected {expected_channels} input channels, "
            f"got {actual_channels}. Check your model's in_channels parameter."
        )

    # Check for NaN/Inf
    if torch.isnan(x).any():
        raise ValueError(
            f"{name}() received input containing NaN values. "
            "Check your data preprocessing pipeline."
        )
    if torch.isinf(x).any():
        raise ValueError(
            f"{name}() received input containing Inf values. "
            "Check your data preprocessing pipeline."
        )
