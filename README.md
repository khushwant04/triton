# Triton: A Detailed Exploration

Learning notes and example kernels for GPU programming with CUDA concepts and
[OpenAI Triton](https://github.com/triton-lang/triton). The repository currently contains written
notes on GPU fundamentals and CUDA setup, a device-check script, and a first Triton kernel
(vector addition).

## Contents

- **Introduction & Overview** - basic GPU computing concepts and an overview of CUDA and Triton:
  [01. Introduction/README.md](01.%20Introduction/README.md).
- **GPU Fundamentals** - CPU vs GPU architecture, threads, blocks, and grids:
  [02. Fundamentals/README.md](02.%20Fundamentals/README.md).
- **Setup & CUDA Installation** - installing the CUDA toolkit, setting environment variables, and
  verifying with `nvcc`: [03. Setup/README.md](03.%20Setup/README.md).
- **CUDA check** - [`cuda_check.py`](cuda_check.py) prints the active PyTorch device (CUDA GPU or CPU).
- **Kernels** - [`kernels/vector_addition.py`](kernels/vector_addition.py): `vector_add_kernel` is a
  1-D Triton kernel where each program instance adds a `BLOCK_SIZE=1024` chunk of two vectors, with a
  bounds mask for the tail. `vector_add()` launches it over a `triton.cdiv(n, BLOCK_SIZE)` grid, and the
  `__main__` block compares the result with PyTorch's `x + y`.

## Project structure

```
.
├── 01. Introduction/README.md
├── 02. Fundamentals/README.md
├── 03. Setup/README.md
├── images/                   # figures used by the notes
├── kernels/vector_addition.py
├── cuda_check.py
└── requirements.txt
```

## Getting started

Triton kernels need an NVIDIA GPU (Linux) with a working driver; see
[03. Setup](03.%20Setup/README.md) for CUDA installation.

```sh
git clone https://github.com/khushwant04/triton.git
cd triton
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # Linux PyTorch wheels include Triton

python cuda_check.py               # e.g. "DEVICE: 0 - <GPU name> (cuda)"
python kernels/vector_addition.py  # prints both results and the max absolute difference
```

## References

- Tillet, Kung, Cox, *Triton: An Intermediate Language and Compiler for Tiled Neural Network
  Computations*, MAPL 2019. [doi:10.1145/3315508.3329973](https://doi.org/10.1145/3315508.3329973)
- [Triton documentation and tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html)
- [NVIDIA: CUDA Refresher - The CUDA Programming Model](https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/)
