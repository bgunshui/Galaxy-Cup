#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
if [[ "$(uname -s)" != Linux || "$(uname -m)" != aarch64 ]]; then
  echo 'This offline environment requires Linux aarch64 (Jetson Thor).' >&2
  exit 1
fi
if [[ ! -x "$HERE/runtime/bin/python3" ]]; then
  mkdir -p "$HERE/runtime"
  tar -xzf "$HERE/python310-aarch64.tar.gz" --strip-components=1 -C "$HERE/runtime"
fi
PYTHON="$HERE/runtime/bin/python3"
export PYTHONNOUSERSITE=1 PIP_DISABLE_PIP_VERSION_CHECK=1
# The official Thor image supplies the NVIDIA driver and CUDA 13.0 runtime.
# Install the framework/application wheels without trying to vendor or fetch
# CUDA Toolkit components; the wheelhouse intentionally contains none of them.
"$PYTHON" -m pip install --no-index --no-deps --find-links "$HERE/wheels" -r "$ROOT/requirements.txt"
"$PYTHON" - <<'PY'
import platform
import torch
import torchvision
import transformers
import peft
import tifffile
import imagecodecs
assert platform.machine() == 'aarch64'
assert torch.cuda.is_available(), 'NVIDIA driver/CUDA device is not available'
assert torch.version.cuda == '13.0', torch.version.cuda
capability = torch.cuda.get_device_capability()
assert capability == (11, 0), ('Expected Thor sm_110', capability)
torch.ones(1, device='cuda').add_(1)
torch.cuda.synchronize()
print('OFFLINE_ENVIRONMENT_READY', torch.__version__, torch.cuda.get_device_name())
PY
