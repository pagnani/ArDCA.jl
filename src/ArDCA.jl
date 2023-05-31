module ArDCA

using Random: randperm 
using Printf: @printf 
using LinearAlgebra: rmul!, norm
using ExtractMacro: @extract
using NLopt: Opt,ftol_abs!,xtol_rel!,xtol_abs!,ftol_rel!,maxeval!,min_objective!,optimize
using Distributions: wsample
using LoopVectorization: @turbo
using DCAUtils: read_fasta_alignment,remove_duplicate_sequences,compute_weights
using DCAUtils.ReadFastaAlignment: letter2num

export ardca,ArVar,ArAlg,ArNet,sample,sample_with_weights,epistatic_score,dms_single_site,loglikelihood,siteloglikelihood

include("types.jl")
include("ar.jl")
include("utils.jl")
include("dca.jl")
include("precompile.jl")
end # end module
