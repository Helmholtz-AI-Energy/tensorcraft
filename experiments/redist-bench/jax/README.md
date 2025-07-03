# Jax Redistribution Benchmarks

## Installation

### Setup 1 (FTP-x86:amd-milan-210)

**Modules**
 - compiler/intel/2025.1_llvm (Maybe)
 - numlib/mkl/2025.1 (Maybe)
 - dot (Maybe)
 - toolkit/rocm/6.4.1
 - mpi/openmpi/5.0
 - devel/cuda/12.4 (Maybe)

**Jax Installation (CPU)**

1. Simply
```console
uv pip install jax
```


**Jax Installation (With rocm, did not work yet)**
Instructions: https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/jax-install.html#build-rocm-jax-from-source

1. Create env (instructions with uv)

```console
uv venv --python 3.12 venv/jax_rocm
source venv/jax_rocm/bin/activate
```

2. Create custom wheels

```console
git clone https://github.com/ROCm/jax jax_src
cd jax_src
python3 ./build/build.py build --wheels=jaxlib,jax-rocm-plugin,jax-rocm-pjrt --rocm_version=60 --rocm_path=/opt/rocm-6.4.1
uv pip install dist/*.whl
```

**Other dependencies**

- mpi4py
