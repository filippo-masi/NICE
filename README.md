## Neural integration for constitutive equations

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<center><img src="./_images/NICE.png"  alt="centered image" width="100%" height="51.15%"></center>

## Overview

This library provides the implementation in PyTorch of the **Neural integration for constitutive equations** (NICE) method [1]. The algorithms in this repository are implemented using `torch` [2] and `torchdiffeq` [3] libraries  Backpropagation through ODE solutions is supported using the adjoint method for constant memory cost. For usage of ODE solvers in deep learning applications, see reference [1].

As the solvers are implemented in PyTorch, algorithms in this repository are fully supported to run on the GPU.The **NICE** (Neural Integration for Constitutive Equations) method is a novel deep learning tool for the automatic discovery of constitutive equations from small data - partial and incomplete material state observations. 
The approach builds upon the solution of the initial value problem describing the time evolution of the material state and leverages the framework provided by neural differentials equations (Chen et al., 2018).
NICE can learn accurate, consistent, and robust constitutive models from incomplete, sparse, and noisy data collecting simple conventional experimental protocols. 

## Features

- **Neural Network Architecture**: Utilizes feed-forward artificial neural networks to model the evolution of constitutive equations.
- **Normalization**: Implements robust data normalization techniques for enhanced training performance.
- **ODE Integration**: Solves ordinary differential equations (ODEs) efficiently using the `torchdiffeq` library.
- **Early Stopping**: Includes early stopping functionality to prevent overfitting during training.

## Prerequisites

- Python 3.6+
- PyTorch


## References

If you use this code, please cite the related paper and repository:

[1] F Masi, I Einav (2023). "[Neural integration for constitutive equations using small data](https://doi.org/10.48550/arXiv.2311.07849)". arXiv preprint: 2311.07849.

    @article{masieinav2023,
    title={Neural integration for constitutive equations using small data},
    author={Masi, Filippo and Einav, Itai},
    journal={arXiv preprint 2311.07849},
    year={2023},
    doi={10.48550/arXiv.2311.07849}

    @article{masieinav2023repo,
    title={{\texttt{NICE: Neural integration for constitutive equations}}},
    author={Masi, Filippo and Einav, Itai},
    year={2023},
    url={https://github.com/filippo-masi/NICE}
    
    
[3] A Paszke, S Gross, S Chintala, G Chanan, E Yang, Z DeVito, Z Lin, A Desmaison, L Antiga, and A Lerer. Automatic differentiation in PyTorch. 2017.

[4] R TQ Chen. `torchdiffeq`, 2018. url: [https://github.com/rtqichen/torchdiffeq](https://github.com/rtqichen/torchdiffeq).


