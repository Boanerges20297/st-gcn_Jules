import os
import subprocess
import sys

SCRIPT = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'scripts', 'eval_cvli.py')

def test_run_eval_cvli():
    """Run the CVLI evaluation script (smoke test)."""
    if not os.path.exists(SCRIPT):
        pytest.skip('eval script not found')
    # If model is missing, skip to avoid heavy failures
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'stgcn_cvli.pth')
    if not os.path.exists(model_path):
        import pytest
        pytest.skip('model checkpoint not present; skipping CVLI evaluation')

    proc = subprocess.run([sys.executable, SCRIPT], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=300)
    print(proc.stdout)
    if proc.returncode != 0:
        print('STDERR:', proc.stderr)
    assert proc.returncode == 0
