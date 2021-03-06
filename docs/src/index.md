```@meta
CurrentModule = ArDCA
```

# ArDCA
* Autoreressive model learning for protein inference.

## Package Features

- Learn model from multiple sequence alignment
- Sample from the model 


See the [Index](@ref index) for the complete list of documented functions and types.

## Overview

Protein families are given in form of multiple sequence alignments (MSA) $D = (a^m_i |i = 1,\dots,L;\,m = 1,\dots,M)$ of $M$ proteins of aligned length $L$. The entries $a^m_i$ equal either one of the standard 20 amino acids, or the alignment gap $–$. In total, we have $q = 21$ possible different symbols in D. The aim of unsupervised generative modeling is to earn a statistical model $P(a_1,\dots,a_L)$ of (aligned) full-length sequences, which faithfully reflects the variability found in $D$: sequences belonging to the protein family of interest should have comparably high probabilities, unrelated sequences very small probabilities.
Here we propose a computationally efficient approach based on autoregressive models. 

We start from the exact decomposition:

$P(a_1,\dots,a_L) = P(a_1) \cdot P(a_2|a_1) \cdot \dots \cdot P(a_L|a_1,\dots,a_{L-1})$

Here, we use the following parametrization:

$P(a_i | a_1,\dots,a_{i-1}) = \frac{\exp \left\{ h_i(a_i) + \sum_{j=1}^{i-1} J_{i,j}(a_i,a_j)\right\} }{z_i(a_1,\dots,a_{i-1})}\,,$

where:

$z_i(a_1,\dots,a_{i-1})= \sum_{a=1}^{q} \exp \left\{ h_i(a) + \sum_{j=1}^{i-1} J_{i,j}(a,a_j)\right\} \,,$

is a the normalization factor. In machine learning, this
parametrization is known as soft-max regression, the generalization of logistic regression to multi-class labels.

# Usage

The typical pipeline to use the package is:

* Compute ArDCA parameters from a multiple sequence alignment:

``` 
julia> arnet,arvar=ardca(filefasta; kwds...)
```

* Generate `100` new sequences, and store it in an $L\times 100$ array of integers.

```
julia> Zgen =  sample(arnet,100);
```

## [Index](@id index)
```@index
```

```@autodocs
Modules = [ArDCA]
```
