from dataclasses import dataclass
import torch
import time

@dataclass
class BenchmarkResult:
    tokens_generated: int
    wall_seconds: float
    tokens_per_second: float
    peak_vram_gb: float

def reset_peak_vram() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

def measure(num_tokens: int, start_time: float) -> BenchmarkResult:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time
    peak = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
    tps = num_tokens / elapsed if elapsed > 0 else 0.0
    return BenchmarkResult(
        tokens_generated=num_tokens,
        wall_seconds=elapsed,
        tokens_per_second=tps,
        peak_vram_gb=peak,
    )