#!/bin/bash
# Manual patch script for torch.nn.Buffer deprecation
# This patches TDMPC2 code to work with PyTorch 2.0+
# Note: Docker builds apply these patches automatically

set -e

echo "Patching TDMPC2 for PyTorch 2.0+ compatibility..."

# Patch scale.py
if [ -f "third_party/tdmpc2/tdmpc2/common/scale.py" ]; then
    echo "Patching scale.py..."
    sed -i.bak 's/from torch.nn import Buffer/# Buffer removed in PyTorch 2.0+/' third_party/tdmpc2/tdmpc2/common/scale.py
    sed -i 's/self.value = Buffer(/self.register_buffer("value", /' third_party/tdmpc2/tdmpc2/common/scale.py
    sed -i 's/self._percentiles = Buffer(/self.register_buffer("_percentiles", /' third_party/tdmpc2/tdmpc2/common/scale.py
    echo "  ✓ scale.py patched"
fi

# Patch tdmpc2.py
if [ -f "third_party/tdmpc2/tdmpc2/tdmpc2.py" ]; then
    echo "Patching tdmpc2.py..."
    sed -i.bak 's/self._prev_mean = torch.nn.Buffer(/self.register_buffer("_prev_mean", /' third_party/tdmpc2/tdmpc2/tdmpc2.py
    echo "  ✓ tdmpc2.py patched"
fi

echo ""
echo "Patching complete! Backup files created with .bak extension"
echo "To verify patches, run: python docker/smoke_test.py"
