# Sparsely Connected Neural Network Representation of Lagrange Finite Element Functions

This repository contains the Python implementation associated with the preprint:

**Jiaxiong Hao, Yunqing Huang, and Nianyu Yi**  
**[Sparsely connected neural network representation of Lagrange finite element function](https://arxiv.org/abs/2609.23299)**  
*arXiv:2609.23299* [math.NA], submitted September 20, 2026.  
DOI: [10.48550/arXiv.2609.23299](https://doi.org/10.48550/arXiv.2609.23299)

## Overview

This project implements mesh-induced, sparsely connected neural network representations of Lagrange finite element functions. Unlike data-driven neural surrogates, these networks are constructed directly from finite element meshes and degrees of freedom, **without a training process**.

The repository provides one-, two-, and three-dimensional numerical examples, including linear (**P1**) and quadratic (**P2**) Lagrange elements. The examples examine function evaluation and, where implemented, spatial gradients through PyTorch automatic differentiation, with comparisons against conventional interpolation methods.

The same representation is also used to transfer finite element solutions between non-matching meshes in adaptive finite element computations for parabolic equations. The associated paper develops the framework for arbitrary-order Lagrange elements; the numerical implementations in this repository focus on the element types listed below.

## Requirements

Recommended environment (based on the related Python implementation; compatibility with every script in this repository has not been independently verified):

- Python 3.8.20
- FEALPy 1.1.20
- PyTorch 2.4.1

Additional dependencies:

- NumPy
- SciPy
- Matplotlib
- Gmsh and meshio (for mesh regeneration)

The adaptive finite element examples use legacy FEALPy interfaces, so FEALPy 1.1.20 is recommended over newer versions.

## Code Structure

### Finite Element Representation Examples

| File | Description |
| --- | --- |
| [`NNFEM1D.py`](NNFEM1D.py) | One-dimensional linear (P1) finite element representation. |
| [`NNFEM1D2P.py`](NNFEM1D2P.py) | One-dimensional quadratic (P2) finite element representation. |
| [`NNFEM2D.py`](NNFEM2D.py) | Two-dimensional linear (P1) finite element representation, including function and gradient comparisons. |
| [`NNFEM2D2P.py`](NNFEM2D2P.py) | Two-dimensional quadratic (P2) finite element representation. |
| [`NNFEM3D.py`](NNFEM3D.py) | Three-dimensional linear (P1) finite element representation on tetrahedral meshes. |

The examples construct finite element function representations and compare their evaluations with conventional piecewise interpolation. Files containing **`2P`** denote **quadratic (P2)** elements.

### Core and Auxiliary Modules

**[`NN4FEM.py`](NN4FEM.py)**  
Provides reusable neural network representations for P1 and P2 Lagrange finite element functions on two-dimensional triangular meshes. The module exposes `build_p1fem_network` and `build_p2fem_network`, which construct callable evaluators from an FEALPy finite element space and its coefficients.

**[`mesh_regenerator.py`](mesh_regenerator.py)**  
Provides error-indicator-based mesh regeneration routines using Gmsh and meshio, including routines for two-dimensional and three-dimensional computational domains.

### Adaptive Finite Element Examples

| File | Description |
| --- | --- |
| [`hr_AFEM_HEAT(NNFEM).py`](hr_AFEM_HEAT%28NNFEM%29.py) | Neural network-enhanced adaptive finite element example for a two-dimensional heat equation using P1 elements. |
| [`hr_AFEM_HEAT(NNFEM2P).py`](hr_AFEM_HEAT%28NNFEM2P%29.py) | Corresponding adaptive heat-equation example using P2 elements. |

These examples use the network representations to evaluate existing finite element solutions on newly generated meshes during the adaptive solution process.


## Usage

This repository provides two types of numerical examples: standalone construction and verification of neural representations of Lagrange finite element functions, and their use in adaptive finite element computations. **The networks are constructed directly from the mesh and finite element coefficients; no network training or pretrained weights are required.**

### 1. Finite Element Representation Examples

Each standalone script generates a test mesh and nodal data, constructs the corresponding sparse neural network, and compares its evaluation with conventional finite element interpolation.

| Script | Dimension | Element |
| --- | --- | --- |
| `NNFEM1D.py` | 1D | Linear (P1) |
| `NNFEM1D2P.py` | 1D | Quadratic (P2) |
| `NNFEM2D.py` | 2D | Linear (P1) |
| `NNFEM2D2P.py` | 2D | Quadratic (P2) |
| `NNFEM3D.py` | 3D | Linear (P1) |

For example, run:

```bash
python NNFEM1D.py
python NNFEM2D.py
python NNFEM3D.py
```

The examples report numerical comparisons of function values and, where implemented, gradients. Mesh resolution, test functions, and evaluation points can be changed in the corresponding scripts to investigate other configurations.

### 2. Using the Representation in Finite Element Computations

`NN4FEM.py` provides the functions `build_p1fem_network` and `build_p2fem_network` for constructing callable representations from an existing two-dimensional FEALPy finite element space and its solution coefficients.

```python
from NN4FEM import build_p1fem_network

# space: an existing P1 FEALPy finite element space
# uh: its finite element solution coefficients
predict = build_p1fem_network(space, uh)
values = predict(target_points)
```

Here, `target_points` contains the coordinates at which the finite element function is evaluated, including nodes of a newly generated mesh. For quadratic (P2) elements, use `build_p2fem_network` instead.

### 3. Adaptive Heat Equation Examples

The following scripts demonstrate how the constructed representations can be used in adaptive finite element computations for a two-dimensional heat equation:

```bash
python 'hr_AFEM_HEAT(NNFEM).py' --ns 11 --nt 100 --tol 0.05
python 'hr_AFEM_HEAT(NNFEM2P).py' --ns 11 --nt 100 --tol 0.05
```

The first example uses P1 elements, and the second uses P2 elements. Both construct a representation of the previous finite element solution to evaluate it on a regenerated mesh. Parameters `--ns`, `--nt`, `--tol`, and `--output` can be adjusted; the default output directory is `./AFEM-hr-heat`, and the scripts save numerical results as `results.mat`.

These adaptive examples depend on compatible legacy FEALPy interfaces as well as Gmsh and meshio for mesh regeneration. The commands describe the script entry points and have not been independently tested in a complete runtime environment.

## Citation

If you use this code in academic work, please cite:

```bibtex
@misc{Hao2026SparseNN,
  title         = {Sparsely connected neural network representation of Lagrange finite element function},
  author        = {Hao, Jiaxiong and Huang, Yunqing and Yi, Nianyu},
  year          = {2026},
  eprint        = {2609.23299},
  archivePrefix = {arXiv},
  primaryClass  = {math.NA},
  doi           = {10.48550/arXiv.2609.23299},
  url           = {https://arxiv.org/abs/2609.23299}
}
```
