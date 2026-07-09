# ABOUTME: Memory + device + seed utilities shared by the GPU steps (masks, vae) and the CPU step.
# ABOUTME: torch is imported lazily so the CPU spectrogram step never pays for it at import time.
import os
import gc
import logging
from typing import Optional

# Single, GPU-aware implementations consolidated from the old utils.py (CPU-only)
# and sam_utils.py (GPU) variants so there is exactly one of each name.


def cleanup_memory(logger: Optional[logging.Logger] = None,
                   force_gpu_cleanup: bool = False,
                   log_memory: bool = False) -> None:
    """Garbage-collect and (if a CUDA device is present) clear the GPU cache.

    Safe to call from the CPU-only step: torch is imported lazily and the GPU
    branch is skipped when CUDA is unavailable.

    Args:
        logger: Logger for output (default: module logger).
        force_gpu_cleanup: If True, synchronize + empty cache + log GPU usage.
        log_memory: If True, log CPU RSS via psutil (best-effort).
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    gc.collect()

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if force_gpu_cleanup:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                allocated = torch.cuda.memory_allocated() / 1024 ** 3
                reserved = torch.cuda.memory_reserved() / 1024 ** 3
                logger.info(f"GPU memory after cleanup - Allocated: {allocated:.2f}GB, "
                            f"Reserved: {reserved:.2f}GB")
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass

    if log_memory:
        try:
            import psutil
            memory_mb = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
            logger.debug(f"Memory usage: {memory_mb:.1f} MB RSS")
        except ImportError:
            pass


def log_memory_usage(logger: logging.Logger, stage: str = "") -> None:
    """Log current CPU (and GPU, if present) memory usage for monitoring."""
    try:
        import psutil
        cpu_gb = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3
        msg = f"Memory usage {stage}: CPU {cpu_gb:.2f}GB"
    except ImportError:
        msg = f"Memory usage {stage}: (psutil unavailable)"

    try:
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024 ** 3
            reserved = torch.cuda.memory_reserved() / 1024 ** 3
            peak = torch.cuda.max_memory_allocated() / 1024 ** 3
            msg += f", GPU allocated: {allocated:.2f}GB, reserved: {reserved:.2f}GB, peak: {peak:.2f}GB"
    except ImportError:
        pass

    logger.info(msg)


def setup_device(deterministic: bool = False, logger: Optional[logging.Logger] = None):
    """Set up the compute device (cuda/mps/cpu) with sensible perf flags.

    Enables bf16 autocast + TF32 on Ampere+ GPUs (unless deterministic). Returns a
    ``torch.device``.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    import torch
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
        if not deterministic:
            torch.backends.cudnn.benchmark = True
            logger.info("CuDNN benchmark enabled for performance")
        else:
            logger.info("Deterministic mode enabled - CuDNN benchmark disabled")
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("TF32 acceleration enabled for Ampere GPU")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple MPS device")
    else:
        device = torch.device("cpu")
        logger.warning("Using CPU - this will be much slower")

    logger.info(f"Compute device: {device}")
    return device
