from __future__ import annotations

import sys
from pathlib import Path

try:  # pragma: no cover - imported only with the train extra.
    import torch
    from torch import Tensor
except ImportError:  # pragma: no cover
    torch = None
    Tensor = object  # type: ignore[assignment,misc]


def enable_hf_mamba_kernels() -> bool:
    """Expose HuggingFace kernel-hub Mamba kernels under their expected names."""

    try:
        from transformers.integrations import lazy_load_kernel
    except ImportError:
        return False

    causal_conv1d_kernel = lazy_load_kernel("causal-conv1d")
    if causal_conv1d_kernel is None or getattr(causal_conv1d_kernel, "__file__", None) is None:
        return False

    kernel_root = str(Path(causal_conv1d_kernel.__file__).parent)
    if kernel_root not in sys.path:
        sys.path.insert(0, kernel_root)

    try:
        from causal_conv1d import causal_conv1d_fn
        from causal_conv1d import causal_conv1d_update
        import causal_conv1d.causal_conv1d_interface as causal_interface
    except ImportError:
        return False

    if not hasattr(causal_interface, "causal_conv1d_cuda"):
        causal_interface.causal_conv1d_cuda = _CausalConv1dCudaAdapter

    mamba_kernel = lazy_load_kernel("mamba-ssm")
    if mamba_kernel is None:
        return False

    try:
        import importlib
        import transformers.models.mamba.modeling_mamba as hf_mamba

        scan_interface = importlib.import_module(
            f"{mamba_kernel.__name__}.ops.selective_scan_interface"
        )
    except (ImportError, AttributeError):
        return False

    if getattr(scan_interface, "causal_conv1d_fn", None) is None:
        scan_interface.causal_conv1d_fn = causal_conv1d_fn
    if getattr(scan_interface, "causal_conv1d_cuda", None) is None:
        scan_interface.causal_conv1d_cuda = causal_interface.causal_conv1d_cuda

    hf_mamba.causal_conv1d_fn = causal_conv1d_fn
    hf_mamba.causal_conv1d_update = causal_conv1d_update
    mamba_inner_fn = getattr(mamba_kernel, "mamba_inner_fn", None)
    if mamba_inner_fn is not None:
        hf_mamba.mamba_inner_fn = mamba_inner_fn
    selective_scan_fn = getattr(mamba_kernel, "selective_scan_fn", None)
    if selective_scan_fn is not None:
        hf_mamba.selective_scan_fn = selective_scan_fn
    return True


class _CausalConv1dCudaAdapter:
    @staticmethod
    def causal_conv1d_fwd(
        x: Tensor,
        weight: Tensor,
        bias: Tensor | None,
        seq_idx: Tensor | None,
        initial_states: Tensor | None,
        final_states_out: Tensor | None,
        silu_activation: bool,
    ) -> Tensor:
        from causal_conv1d.cpp_functions import causal_conv1d_fwd_function

        return causal_conv1d_fwd_function(
            x,
            weight,
            bias,
            seq_idx,
            initial_states,
            final_states_out,
            silu_activation,
        )

    @staticmethod
    def causal_conv1d_bwd(
        x: Tensor,
        weight: Tensor,
        bias: Tensor | None,
        dout: Tensor,
        seq_idx: Tensor | None,
        initial_states: Tensor | None,
        dfinal_states: Tensor | None,
        dx: Tensor | None,
        return_dinitial_states: bool,
        silu_activation: bool,
    ) -> tuple[Tensor | None, ...]:
        from causal_conv1d.cpp_functions import causal_conv1d_bwd_function

        return causal_conv1d_bwd_function(
            x,
            weight,
            bias,
            dout,
            seq_idx,
            initial_states,
            dfinal_states,
            dx,
            return_dinitial_states,
            silu_activation,
        )

    @staticmethod
    def causal_conv1d_update(
        x: Tensor,
        conv_state: Tensor,
        weight: Tensor,
        bias: Tensor | None,
        silu_activation: bool,
        cache_seqlens: Tensor | None,
        conv_state_indices: Tensor | None,
    ) -> Tensor:
        from causal_conv1d.cpp_functions import causal_conv1d_update_function

        return causal_conv1d_update_function(
            x,
            conv_state,
            weight,
            bias,
            silu_activation,
            cache_seqlens,
            conv_state_indices,
        )
