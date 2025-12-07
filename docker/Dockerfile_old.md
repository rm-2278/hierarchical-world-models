# Use the highest, stable, and available CUDA 13.0 image
FROM pytorch/pytorch:2.9.1-cuda13.0-cudnn9-runtime

WORKDIR /workspace

# 1. Copy Dreamer requirements first
COPY third_party/dreamerv3/env/requirements.txt /tmp/dreamer_reqs.txt

# 2. Install Dreamer requirements BUT EXCLUDE JAX
# We use 'sed' to remove lines containing 'jax' or 'tensorflow' if it conflicts, 
# preventing the downgrade before it happens.
RUN sed -i '/jax/d' /tmp/dreamer_reqs.txt && \
    pip install --root-user-action=ignore -r /tmp/dreamer_reqs.txt

# 3. NOW install the correct JAX for CUDA 13 (The "King" stays on top)
RUN pip install --root-user-action=ignore \
  --upgrade \
  "jax[cuda13]" \
  -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# 4. Clean up plugins (Crucial: remove any cuda12 plugins that might have snuck in)
RUN pip uninstall -y jax-cuda12-plugin || true

# 5. Rest of the setup
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --root-user-action=ignore -r requirements.txt

COPY . .

CMD ["/bin/bash"]