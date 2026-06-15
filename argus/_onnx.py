"""Shared ONNX Runtime provider selection for the face/embed backends.

Picks the execution providers from those *actually available* in the installed
onnxruntime build, so requesting ``device="cuda"`` uses the GPU when the CUDA EP is
present (``onnxruntime-gpu``) and otherwise falls back to CPU with a clear warning —
instead of silently degrading or erroring on a missing provider.
"""

from __future__ import annotations

import warnings

_preloaded = False


def _preload_cuda(ort) -> None:
    """Load CUDA/cuDNN from the nvidia ``*-cu12`` pip packages (once per process).

    onnxruntime-gpu ships no CUDA libraries of its own; ``preload_dlls`` dlopens them
    from the installed nvidia wheels so the CUDA EP can initialize without the libs being
    on the system loader path. No-op on builds/versions without it.
    """
    global _preloaded
    if _preloaded or not hasattr(ort, "preload_dlls"):
        return
    try:
        ort.preload_dlls()
    except Exception:  # best-effort; session creation still falls back to CPU on failure
        pass
    _preloaded = True


def onnx_providers(device: str | None) -> tuple[list[str], int]:
    """Return ``(providers, ctx_id)`` for the requested ``device``.

    ``device`` follows the SDK convention (``"cuda"``/``"cuda:0"``/``"cpu"``/``None``).
    CUDA is used only if requested *and* the CUDA execution provider is available; the
    returned ``providers`` always ends with ``CPUExecutionProvider`` so onnxruntime can
    fall back per-op. ``ctx_id`` is ``0`` for GPU, ``-1`` for CPU (insightface convention).
    """
    import onnxruntime as ort

    want_cuda = bool(device) and str(device).lower().startswith("cuda")
    if want_cuda:
        _preload_cuda(ort)
    has_cuda = "CUDAExecutionProvider" in ort.get_available_providers()

    if want_cuda and not has_cuda:
        warnings.warn(
            "CUDA requested but onnxruntime's CUDAExecutionProvider is unavailable "
            "(install the 'onnxruntime-gpu' package); falling back to CPU.",
            stacklevel=2,
        )

    use_cuda = want_cuda and has_cuda
    providers = (["CUDAExecutionProvider"] if use_cuda else []) + ["CPUExecutionProvider"]
    return providers, (0 if use_cuda else -1)
