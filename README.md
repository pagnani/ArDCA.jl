# ArDCA

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://pagnani.github.io/ArDCA/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://pagnani.github.io/ArDCA/dev)
[![Build Status](https://github.com/pagnani/ArDCA/workflows/CI/badge.svg)](https://github.com/pagnani/ArDCA/actions)
[![Coverage](https://codecov.io/gh/pagnani/ArDCA/branch/master/graph/badge.svg)](https://codecov.io/gh/pagnani/ArDCA)
[[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


Autoregressive protein model learning through generalized logistic regression in Julia.
## Overview

The authors of this code are Jeanne Trinquier, Guido Uguzzoni, Andrea Pagnani, Francesco Zamponi, and Martin Weigt.

See also [this Wikipedia article](https://en.wikipedia.org/wiki/Direct_coupling_analysis) article for a general overview of the Direct Coupling Analysis technique. 

The code is written in [Julia](https://www.julialang.org/).

## Install

To install the package, enter in Pkg mode by pressing the <kbd>]</kbd> key,
then in the pkg prompt enter

```
julia> using Pkg; Pkg.add("https://github.com/pagnani/ArDCA"); 
```
## Notebooks

There are two `jupyter` notebooks (Python, and Julia) to help using the Package.

## Data 

Data for five protein families (PF00014,PF00072, PF00076,PF00595,PF13354) are contained in the companion
[ArDCAData](https://github.com/pagnani/ArDCAData) package.

## Requirements

The minimal Julia version to run this code is 1.5.

## License

This project is covered under the MIT License.
