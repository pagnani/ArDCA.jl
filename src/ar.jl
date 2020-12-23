const allpermorder = [:NATURAL, :ENTROPIC, :REV_ENTROPIC, :RANDOM]

function ardca(Z::Array{Ti,2},W::Vector{Float64};
                fracmax::Real=0.3,
                remove_dups::Bool=true,
                min_separation::Int=1,
                theta=:auto,
                lambdaJ::Real=0.01,
                lambdaH::Real=0.01,
                epsconv::Real=1.0e-5,
                maxit::Int=1000,
                verbose::Bool=true,
                method::Symbol=:LD_LBFGS,
                permorder::Union{Symbol,Vector{Ti}}=:ENTROPIC
                ) where Ti <: Integer

   
    checkpermorder(permorder)
    all(x -> x > 0, W) || throw(DomainError("vector W should normalized and with all positive elements"))
    isapprox(sum(W), 1) || throw(DomainError("sum(W) ≠ 1. Consider normalizing the vector W"))
    N, M = size(Z)
    M = length(W)
    q = Int(maximum(Z))
    aralg = ArAlg(method, verbose, epsconv, maxit)
    arvar = ArVar(N, M, q, lambdaJ, lambdaH, Z, W, permorder)
    θ,psval = minimize_arnet(aralg, arvar)
    #score, FN, Jtensor, htensor =  compute_score(Jmat, plmvar, min_separation)
    #return ArOut(sdata(pslike), Jtensor, htensor, score)
    ArNet(θ,arvar),arvar,psval
    #return θ,psval,arvar
    #return aralg,arvar
end

function checkpermorder(po::Symbol)
    po ∈ allpermorder || error("permorder :$po not iplemented: only $allpermorder are defined");
end

(checkpermorder(po::Vector{Ti}) where Ti <: Integer) = isperm(po) || error("permorder is not a permutation")

function ardca(filename::String;
                theta::Union{Symbol,Real}=:auto,
                max_gap_fraction::Real=0.9,
                remove_dups::Bool=true,
                kwds...)
    W, Z, N, M, q = read_fasta(filename, max_gap_fraction, theta, remove_dups)
    ardca(Z, W; kwds...)
end

function minimize_arnet(alg::ArAlg, var::ArVar{Ti}) where Ti
    @extract var : N q q2
    @extract alg : epsconv maxit method
    vecps = SharedArray{Float64}(N - 1)
    θ = @distributed vcat for site in 1:N-1
        x0 = zeros(Float64, site * q2 + q)
        opt = Opt(method, length(x0))
        ftol_abs!(opt, epsconv)
        xtol_rel!(opt, epsconv)
        xtol_abs!(opt, epsconv)
        ftol_rel!(opt, epsconv)
        maxeval!( opt, maxit)
        min_objective!(opt, (x, g) -> optimfunwrapper(x, g, site, var))
        elapstime = @elapsed  (minf, minx, ret) = optimize(opt, x0)
        l2n = l2norm_asym(minx,var)
        alg.verbose && @printf("site = %d\tpl = %.4f\ttime = %.4f\t", site, minf, elapstime)
        alg.verbose && println("status = $ret")
        vecps[site] = minf
        minx
    end
    return θ, vecps
end

function optimfunwrapper(x::Vector, g::Vector, site, var)
    g === nothing && (g = zeros(Float64, length(x)))
    return pslikeandgrad!(x, g, site,  var)
end

function pslikeandgrad!(x::Vector{Float64}, grad::Vector{Float64}, site::Int, arvar::ArVar)
    @extract arvar : N M q q2 lambdaJ lambdaH
    LL = length(x)
    Z = sdata(arvar.Z)
    W = sdata(arvar.W)
    IdxZ = sdata(arvar.IdxZ)
    for i = 1:LL - q
        grad[i] = 2.0 * lambdaJ  * x[i]
    end
    for i = (LL - q + 1):LL
        grad[i] = 2.0 * lambdaH * x[i]
    end
    pseudolike = 0.0
    vecene = zeros(Float64, q)
    expvecenesumnorm = zeros(Float64, q)
    @inbounds for m in 1:M
        izm = view(IdxZ, :, m)
        zsm = Z[site+1,m] # the i index of P(x_i|x_1,...,x_i-1) corresponds here to i+1
        fillvecene!(vecene, x, site, izm, q, N)
        lnorm = logsumexp(vecene)
        expvecenesumnorm .= @. exp(vecene - lnorm)
        pseudolike -= W[m] * (vecene[ zsm ] - lnorm)
        sq2 = site * q2 
        @avx for i in 1:site
            for s in 1:q
                grad[ izm[i] + s ] += W[m] * expvecenesumnorm[s]                
            end
            grad[ izm[i] + zsm ] -= W[m]
        end
        @avx for s = 1:q
            grad[ sq2 + s ] += W[m] * expvecenesumnorm[s]            
        end
        grad[ sq2 + zsm ] -= W[m]
    end
    pseudolike += l2norm_asym(x, arvar)
end

function fillvecene!(vecene::Vector{Float64}, x::Vector{Float64}, site::Int, IdxSeq::AbstractArray{Int,1}, q::Int, N::Int)
    q2 = q^2
    sq2 = site * q2 
    @inbounds for l in 1:q
        scra = 0.0
        @avx for i in 1:site
            scra += x[IdxSeq[i] + l]
        end
        scra += x[sq2 + l] # sum H
        vecene[l] = scra
    end
end

function logsumexp(X::Vector)
    u = maximum(X)
    isfinite(u) || return float(u)
    return u + log(sum(x -> exp(x - u), X))
end

function l2norm_asym(vec::Array{Float64,1}, arvar::ArVar)
    @extract arvar : q N lambdaJ lambdaH
    LL = length(vec)
    mysum1 = 0.0
    @inbounds @avx for i = 1:(LL - q)
        mysum1 += vec[i] * vec[i]
    end
    mysum1 *= lambdaJ
    mysum2 = 0.0
    @inbounds @avx for i = (LL - q + 1):LL
        mysum2 += vec[i] * vec[i]
    end
    mysum2 *= lambdaH
    return mysum1 + mysum2
end