import os
import sys
from pathlib import Path

# Disable TF32 for every CUDA library in the process (incl. the sweep script's
# default-options ORT session) so GPU results match fp32 torch tightly. Must be
# set before CUDA initializes, hence here at conftest import time.
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"

# Project modules (config, construct, ...) live at the repo root, not in a package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
