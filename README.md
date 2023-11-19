## Neural integration for constitutive equations

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


<center><img src="./_images/NICE.png"  alt="centered image" width="100%" height="51.15%"></center>

## Description

Welcome to the Novel Method repository! This repository houses an innovative approach to <briefly describe what your method does>. Our method leverages <key features or technologies> to achieve <unique benefits or results>. Whether you're a researcher, developer, or enthusiast, feel free to explore, experiment, and contribute to our novel solution.  ([Masi, Stefanou, 2022](https://doi.org/10.1016/j.cma.2022.115190)) to generate data for training Thermdoynamics-based Artificial Neural Networks and their validation.

## Features

- **Innovative Approach:** Utilizes a novel method to address <specific problem or challenge>.
- **Performance:** Achieves superior results compared to existing methods in terms of <performance metric>.
- **Flexibility:** Easily adaptable to different scenarios and applications.

## Getting Started

### Prerequisites

- Python 3.6+
- PyTorch



##### Usage

- The file ``` lattice_material.py ``` contains the classes for the constructor, assembly of lattice structures, and FE solver (Newton's method).
- ``` lattice_prescribed_path.py ``` contains the script for running the FE analysis of a lattice material unit cell, with periodic boundary conditions, given a prescribed strain increment path.
  - Constructor parameters: ```xmax, ymax, zmax``` are the total dimensions of the unit cell; ```nx, ny, nz``` are the number of nodes along each direction, and ```s``` is the magnitude of the perturbation (uniform spatial distribution) of the nodal coordinates
  - Boundary conditions: Dirichlet, Neumann, and periodic boundary conditions are implemented. The call is
    ```sh
    BC = [nodal_degree,value,"type"]
    ```
    with ```nodal_degree``` being the degree of freedom of a particular node (i.e., node's index in ```node_coordinates``` times 3 plus 3), ```value``` the prescribed value, and ```type``` the type of boundary condition ```"DC"``` for Dirichlet, ```"NM"``` for Neumann, ```"PR"``` for periodic.
- ``` lattice_data_gen.py ``` contains the script for running the data generation, with periodic boundary conditions, given a prescribed strain increment path.
- ``` lattice_torsional.py ``` contains the script for running the FE analysis of a lattice structure with fixed bottom end and imposed torsional displacement (see [Masi, Stefanou, 2022](https://doi.org/10.1016/j.cma.2022.115190)).

### 2. Multiscale simulation with TANN - ``` TANN - Numerical Geolab ```

Hands-on: employ TANN as a user-material to perform Finite Element analyses [using Numerical Geolab, 2].
The application consists of a 3D model subjected to torsional deformations. The material used represents the volume average behavior of a lattice microstructure with bars displaying elasto-plastic rate-independent behavior, with von Mises yield criterion, and kinematic hardening. For more, we refer to [1,2].


<img src="./TANN - Numerical Geolab/_images/displacement_vertical_AI.png"  width="25%" height="20%">

         Torsional warping: vertical displacement field due to a torsional deformation. The displacement fields were exported with the help of the third party software Paraview.


IMPORTANT: For running part of the script for the multiscale simulations, Numerical Geolab [2] software is needed. The software is currently under review and will be uploaded online soon.
For more information, [contact me](mailto:filippo.masi@sydney.edu.au)



### References

If you use this code, please cite the related paper and repository:

[1] F Masi, I Einav (2023). "[Neural integration for constitutive equations using small data](https://doi.org/10.48550/arXiv.2311.07849)". arXiv preprint: 2311.07849.

[2] F Masi, I Einav (2023), [```NICE: Neural integration for constitutive equations```](https://github.com/filippo-masi/NICE).


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
