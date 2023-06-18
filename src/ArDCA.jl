module ArDCA

import Random: randperm 
import Printf: @printf 
import LinearAlgebra: rmul!, norm
import ExtractMacro: @extract
import NLopt: Opt,ftol_abs!,xtol_rel!,xtol_abs!,ftol_rel!,maxeval!,min_objective!,optimize
import Distributions: wsample
import LoopVectorization: @turbo
import DCAUtils: read_fasta_alignment,remove_duplicate_sequences,compute_weights
import DCAUtils.ReadFastaAlignment: letter2num

export ardca,ArVar,ArAlg,ArNet,sample,sample_with_weights,epistatic_score,dms_single_site,loglikelihood,siteloglikelihood

include("types.jl")
include("ar.jl")
include("utils.jl")
include("dca.jl")
include("precompile.jl")
end # end module
