#!/bin/bash
# Docker build helper: Patch torch.nn.Buffer usage in TDMPC2
# This script is called during Docker builds to ensure PyTorch 2.0+ compatibility

set -e

echo "Applying torch.nn.Buffer patches for PyTorch 2.0+ compatibility..."

# Patch scale.py
if [ -f "third_party/tdmpc2/tdmpc2/common/scale.py" ]; then
    sed -i 's/from torch.nn import Buffer/# Buffer removed in PyTorch 2.0+/' third_party/tdmpc2/tdmpc2/common/scale.py
    sed -i 's/self.value = Buffer(/self.register_buffer("value", /' third_party/tdmpc2/tdmpc2/common/scale.py
    sed -i 's/self._percentiles = Buffer(/self.register_buffer("_percentiles", /' third_party/tdmpc2/tdmpc2/common/scale.py
    echo "  ✓ Patched scale.py"
fi

# Patch tdmpc2.py
if [ -f "third_party/tdmpc2/tdmpc2/tdmpc2.py" ]; then
    sed -i 's/self._prev_mean = torch.nn.Buffer(/self.register_buffer("_prev_mean", /' third_party/tdmpc2/tdmpc2/tdmpc2.py
    echo "  ✓ Patched tdmpc2.py"
fi

echo "Patching complete!"
