#!/usr/bin/env python
"""Quick runner for digital twin simulation and ML model."""
import subprocess
import sys

print("\n" + "="*70)
print(" REGENERATING SIMULATION DATA WITH NOISE-ENABLED LABELS")
print("="*70)

try:
    result = subprocess.run([sys.executable, "digital_twin.py"], cwd=".", capture_output=False)
    if result.returncode != 0:
        print(f"\n[ERROR] digital_twin.py failed with code {result.returncode}")
        sys.exit(1)
except Exception as e:
    print(f"\n[ERROR] Failed to run digital_twin.py: {e}")
    sys.exit(1)

print("\n" + "="*70)
print(" RETRAINING ML MODELS WITH NEW DATA")
print("="*70)

try:
    result = subprocess.run([sys.executable, "ml_model.py"], cwd=".", capture_output=False)
    if result.returncode != 0:
        print(f"\n[ERROR] ml_model.py failed with code {result.returncode}")
        sys.exit(1)
except Exception as e:
    print(f"\n[ERROR] Failed to run ml_model.py: {e}")
    sys.exit(1)

print("\n✓ Simulation and retraining complete!")
