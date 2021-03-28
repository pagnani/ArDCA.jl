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

# Multithreading

To fully exploit the the multicore parallel computation, julia should be invoked with

```
$ julia -t nthreads # put here nthreads equal to the number of cores you want to use
```

If you want to set permanently the number of threads to the desired value, you can either create a default environment variable `export JULIA_NUM_THREADS=24` in your `.bashrc`. More information [here](https://docs.julialang.org/en/v1.6/manual/multi-threading/)

# Tutorial

We will assume we have a Multiple Sequence Alignment (MSA)in FASTA format. We aim at

1. Given a MSA, generate a sample
2. Given a MSA, predict contacts
3. Given a MSA, predict the mutational effect in all (ungapped) position of a given target sequence 

## Load ArDCA package 

The following cell loads the package `ArDCA` (*Warning*: the first time it takes a while)

* The `mypkgdir` variable should be set to your `path/to/package` dir.

We will use the PF00014 protein family available in `data/PF14/` folder of the package/

```
mypkgdir = normpath(joinpath(pwd(),".."))
datadir=joinpath(mypkgdir,"data") # put here your path
using Pkg
Pkg.activate(mypkgdir)
using ArDCA
```
## Learn the autoregressive parameters

As a preliminary step, we learn the field and the coupling parameters $h,J$ from the MSA. To do so we use the `ardca` method that return the parameters (stored in `arnet` in the cell below), and the alignment in numerical format and other algorithms variables (stored in `arvar` in the cell below). The default autoregressive order is set to `:ENTROPIC`. We set the $L_2$ regularization to 0.02 for the $J$ and 0.001 for the $h$.

The keyword arguments for the `ardca` method are (with their default value):

* `epsconv::Real=1.0e-5` (convergenge parameter)

* `maxit::Int=1000` (maximum number of iteration - don't change)

* `verbose::Bool=true` (set to `false` to suppress printing on screen)

* `method::Symbol=:LD_LBFGS` (optimization method)

* `permorder::Union{Symbol,Vector{Ti}}=:ENTROPIC` (permutation order). Possible values are: `[:NATURAL, :ENTROPIC, :REV_ENTROPIC, :RANDOM]` or a custom permutation vector.


```
arnet,arvar=ardca("data/PF14/PF00014_mgap6.fasta.gz", verbose=false, lambdaJ=0.02,lambdaH=0.001);
```
## 1. Sequence Generation

We now generate `M` sequences using the `sample` method:

```
M = 1_000;
generated_alignment = sample(arnet,M);
```

The generated alignment has is a  𝐿×𝑀  matrix of integer where  𝐿  is the alignment's length, and  𝑀  the number of samples.

Interestingly, we for each sequence we can also compute the likelihood with the
sample_with_weights method.

```
loglikelihood,generated_alignment = sample_with_weights(arnet,M);
```
## 2. Contact Prediction

We can compute the epistatic score for residue-residue contact prediction. To do so, we can use the `epistatic_score` method. The epistatic score is computed on any target sequence of the MSA. Empirically, it turns out the the final score does not depend much on the choice of the target sequence. 

The autput is contained in a `Vector` of `Tuple` ranked in descendic score order. Each `Tuple` contains $i,j,s_{ij}$ where $s_{ij}$ is the epistatic score of the residue pair $i,j$. The residue numbering is that of the MSA, and not of the unaligned full sequences.

```
target_sequence = 1
score=epistatic_score(arnet,arvar,target_sequence)
```

## 3. Predicting mutational effects

For any reference sequence, we can easily predict the mutational effect for all single mutants. Of course we can extract this information only for the *non-gapped* residues of the target sequence. 

This is done with the `dms_single_site` method, which returns a `q×L` matrix `D` containing $\log(P(mut))/\log(P(wt))$ for all single
site mutants of the reference sequence `seqid` (i.e. the so-called wild type sequence), and `idxgap` a vector of indices of the residues of the reference sequence that contain gaps (i.e. the 21
amino-acid) for which the score has no sense and is set by convention to `+Inf`.

A negative value indicate a beneficial mutation, a value 0 indicate
the wild-type amino-acid.

```
target_sequence = 1
D,idxgap=dms_single_site(arnet,arvar,target_sequence)
```

## [Methods Reference](@id index)
```@index
```

```@autodocs
Modules = [ArDCA]
```
