# ArDCA
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://pagnani.github.io/ArDCA.jl/dev)
[![Build Status](https://github.com/pagnani/ArDCA.jl/workflows/CI/badge.svg)](https://github.com/pagnani/ArDCA.jl/actions)
[![Coverage](https://codecov.io/gh/pagnani/ArDCA.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/pagnani/ArDCA.jl)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


Autoregressive protein model learning through generalized logistic regression in Julia.
## Overview

The authors of this code are Jeanne Trinquier, Guido Uguzzoni, Andrea Pagnani, Francesco Zamponi, and Martin Weigt.

See also [this Wikipedia article](https://en.wikipedia.org/wiki/Direct_coupling_analysis) article for a general overview of the Direct Coupling Analysis technique. 

The code is written in [Julia](https://www.julialang.org/).

## Install

This is a registered package: to install enter `]` in the repl and

```
pkg> add ArDCA 
```
## Notebooks

There are two `jupyter` notebooks (Python, and Julia) to help using the Package.

The [tutorial.ipynb](julia-notebook/tutorial.ipynb) is for the julia version.
The [arDCA_sklearn.ipynb](python-notebook/arDCA_sklearn.ipynb) is for the python version.

## Data 

Data for five protein families (PF00014,PF00072, PF00076,PF00595,PF13354) are contained in the companion
[ArDCAData](https://github.com/pagnani/ArDCAData) package.

For didactic reasons we include locally in the `data` folder, the PF00014 dataset.

## Requirements

The minimal Julia version to run this code is 1.5. To run it in parallel 
using Julia multicore infrastructure, start julia with

```
$> julia -t numcores # ncores can be as large as your available number of threads
```

## Documentation

[Development version](https://pagnani.github.io/ArDCA.jl/dev)

## License

This project is covered under the MIT License.
