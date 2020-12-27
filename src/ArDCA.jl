module ArDCA

using Random: randperm
using SharedArrays: SharedArray,sdata 
using Distributed: @distributed   
using Printf: @printf 
using LinearAlgebra: rmul!, norm
using ExtractMacro: @extract
using NLopt: Opt,ftol_abs!,xtol_rel!,xtol_abs!,ftol_rel!,maxeval!,min_objective!,optimize
using Distributions: wsample
using LoopVectorization: @avx 
using GaussDCA: read_fasta_alignment,remove_duplicate_seqs,compute_weights

export ardca,ArVar,ArAlg,ArNet,sample,sample_with_weights,epistatic_score

include("types.jl")
include("ar.jl")
include("utils.jl")
include("dca.jl")

end # end module
