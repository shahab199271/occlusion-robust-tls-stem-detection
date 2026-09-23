# LUMI software environment

The reference HPC platform is LUMI with Python 3.12 and AMD ROCm.

## Current LUMI reference module

The current LUMI software documentation lists the following PyTorch module as
the newest documented Python 3.12 ROCm environment:

```text
PyTorch/2.7.1-rocm-6.2.4-python-3.12-singularity-20250827
```

Core versions provided by that container:

| Component | Version |
| --- | --- |
| Python | 3.12 |
| ROCm | 6.2.4 |
| PyTorch | 2.7.1 |
| torchvision | 0.22.1 |
| torchaudio | 2.7.1 |
| torchdata | 0.10.0 |
| torchtext | 0.18.0+cpu |
| DeepSpeed | 0.17.4 |
| flash-attention | 2.7.3 |
| transformers | 4.55.3 |
| xformers | 0.0.32+09d42ac5.d20250822 |
| vLLM | 0.10.1+rocm624 |

For LUMI, use the container-provided PyTorch build. Install only the project
add-on packages with:

```bash
pip install -r requirements-lumi.txt
```

## Project Python packages

The pinned project package set is:

| Package | Version |
| --- | --- |
| NumPy | 2.5.3 |
| SciPy | 1.18.1 |
| pandas | 3.0.5 |
| Matplotlib | 3.11.1 |
| scikit-learn | 1.9.0 |
| tqdm | 4.70.0 |
| PyTorch Geometric | 2.7.0 |

PyTorch Geometric 2.7.0 is used here because it supports PyTorch 2.7. PyTorch
Geometric 2.8 targets PyTorch 2.9 and newer and is therefore not an appropriate
pin for the LUMI PyTorch 2.7.1 reference module.

The generic `requirements.txt` pins `torch==2.7.1` and
`torch-geometric==2.7.0` so a non-LUMI installation uses the same major
framework combination.

## Earlier project container

Earlier project runs used the LUMI application container:

```text
lumi-pytorch-rocm-6.2.1-python-3.12-pytorch-20240918-vllm-4075b35.sif
```

LUMI documentation identifies that image family as a Python 3.12 environment
with PyTorch `2.6.0.dev20240918+rocm6.2`, torchvision
`0.20.0.dev20240918+rocm6.2`, flash-attention `2.6.3`, transformers `4.44.2`,
and the corresponding ROCm/vLLM stack. The project virtual environment also
used PyTorch Geometric `2.5.3`.

The exact historical versions of every separately installed scientific Python
package were not preserved in the available logs. Historical environment
information is therefore kept separate from the pinned current reference
environment above.
