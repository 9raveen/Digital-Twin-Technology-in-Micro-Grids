"""
profiling.py
============
Performance profiling and optimization analysis.
"""

import time
import tracemalloc
import pandas as pd
from functools import wraps
from typing import Callable
import json
from config import OUTPUT_DIR


class Profiler:
    """Context manager and decorator for profiling code execution."""

    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.start_memory = None
        self.results = {'name': name, 'duration': None, 'memory_delta': None}

    def __enter__(self):
        self.start_time = time.time()
        tracemalloc.start()
        self.start_memory = tracemalloc.get_traced_memory()[0]
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        current_memory = tracemalloc.get_traced_memory()[0]
        self.results['duration'] = time.time() - self.start_time
        self.results['memory_delta'] = (current_memory - self.start_memory) / (1024**2)  # MB
        tracemalloc.stop()

        print(f"\n{'='*60}")
        print(f"PROFILING: {self.name}")
        print(f"{'='*60}")
        print(f"  Duration     : {self.results['duration']:.3f} seconds")
        print(f"  Memory delta : {self.results['memory_delta']:.2f} MB")
        print(f"{'='*60}\n")

    def __call__(self, func: Callable):
        """Decorator mode."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            self.start_time = time.time()
            tracemalloc.start()
            self.start_memory = tracemalloc.get_traced_memory()[0]

            result = func(*args, **kwargs)

            current_memory = tracemalloc.get_traced_memory()[0]
            self.results['duration'] = time.time() - self.start_time
            self.results['memory_delta'] = (current_memory - self.start_memory) / (1024**2)
            tracemalloc.stop()

            print(f"\n{'='*60}")
            print(f"PROFILING: {func.__name__}")
            print(f"{'='*60}")
            print(f"  Duration     : {self.results['duration']:.3f} seconds")
            print(f"  Memory delta : {self.results['memory_delta']:.2f} MB")
            print(f"{'='*60}\n")

            return result

        return wrapper


class PerformanceAnalyzer:
    """Analyze and compare performance across different configurations."""

    def __init__(self):
        self.benchmarks = []

    def add_benchmark(self, name: str, duration: float, memory: float,
                      metrics: dict = None):
        """Add a benchmark result."""
        benchmark = {
            'name': name,
            'duration': duration,
            'memory_mb': memory,
            **(metrics or {})
        }
        self.benchmarks.append(benchmark)

    def compare(self) -> pd.DataFrame:
        """Return comparison table of all benchmarks."""
        df = pd.DataFrame(self.benchmarks)
        return df.sort_values('duration')

    def report(self, filename: str = 'performance_report.json'):
        """Save performance report to JSON."""
        df = self.compare()
        report = {
            'benchmarks': df.to_dict('records'),
            'summary': {
                'fastest': df.iloc[0]['name'],
                'slowest': df.iloc[-1]['name'],
                'avg_duration': df['duration'].mean(),
                'total_memory': df['memory_mb'].sum()
            }
        }

        output_path = OUTPUT_DIR / filename
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n✓ Performance report saved to {output_path}")
        return report


def profile_simulation(func: Callable) -> Callable:
    """Decorator to profile simulation performance."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        with Profiler(name=f"Simulation: {func.__name__}") as prof:
            result = func(*args, **kwargs)

        return result

    return wrapper


def profile_ml_training(func: Callable) -> Callable:
    """Decorator to profile ML training performance."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        with Profiler(name=f"ML Training: {func.__name__}") as prof:
            result = func(*args, **kwargs)

        return result

    return wrapper


if __name__ == '__main__':
    print("Performance profiling module ready.")
    print("\nUsage:")
    print("  with Profiler('Operation name'):")
    print("      # your code here")
    print("\nOr as decorator:")
    print("  @Profiler('My function')")
    print("  def my_func():")
    print("      pass")
