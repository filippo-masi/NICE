## Neural integration for constitutive equations

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This library provides the implementation in PyTorch of the **Neural integration for constitutive equations** (NICE) method [1]. The algorithms in this repository are implemented using `torch` [2] and `torchdiffeq` [3] libraries, thus are fully supported to run on GPU.

<center><img src="./_images/NICE.png"  alt="centered image" width="100%" height="51.15%"></center>

## Features

The **NICE** (Neural Integration for Constitutive Equations) method is a novel deep learning tool for the automatic discovery of constitutive equations from small data - partial and incomplete material state observations. 
The approach builds upon the solution of the initial value problem describing the time evolution of the material state and leverages the framework provided by neural differentials equations [4].
NICE can learn accurate, consistent, and robust constitutive models from incomplete, sparse, and noisy data collecting simple conventional experimental protocols. 

## Basic usage

This library provides one main interface `NICE` which contains general-material algorithms for solving the initial value problems associated with the time evolution of the material state. 

To call the method:

```
import numpy as np
import torch
from torchdiffeq import odeint

from nice_module import NICE
```

odeint(func, y0, t)
where func is any callable implementing the ordinary differential equation f(t, x), y0 is an any-D Tensor representing the initial values, and t is a 1-D Tensor containing the evaluation points. The initial time is taken to be t[0].

Backpropagation through odeint goes through the internals of the solver. Note that this is not numerically stable for all solvers (but should probably be fine with the default dopri5 method). Instead, we encourage the use of the adjoint method explained in [1], which will allow solving with as many steps as necessary due to O(1) memory usage.

To use the adjoint method:

from torchdiffeq import odeint_adjoint as odeint

odeint(func, y0, t)
odeint_adjoint simply wraps around odeint, but will use only O(1) memory in exchange for solving an adjoint ODE in the backward call.

The biggest gotcha is that func must be a nn.Module when using the adjoint method. This is used to collect parameters of the differential equation.

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
    title={`NICE: Neural integration for constitutive equations`},
    author={Masi, Filippo and Einav, Itai},
    year={2023},
    url={https://github.com/filippo-masi/NICE}
    
[2] A Paszke, S Gross, S Chintala, G Chanan, E Yang, Z DeVito, Z Lin, A Desmaison, L Antiga, and A Lerer. Automatic differentiation in PyTorch. 2017.
[3] R TQ Chen. `torchdiffeq`, 2018. url: [https://github.com/rtqichen/torchdiffeq](https://github.com/rtqichen/torchdiffeq).
[4] R TQ Chen, Y Rubanova, J Bettencourt, and D K Duvenaud. Neural ordinary differential equations. Advances in neural information processing systems, 31, 2018. doi: 10.48550/arXiv.1806.07366.

