function read_fasta(filename::AbstractString,max_gap_fraction::Real, theta::Any, remove_dups::Bool)
    Z = read_fasta_alignment(filename, max_gap_fraction)
    if remove_dups
        Z, _ = remove_duplicate_seqs(Z)
    end
    N, M = size(Z)
    q = round(Int,maximum(Z))
    q > 32 && error("parameter q=$q is too big (max 31 is allowed)")
    W , Meff = compute_weights(Z,q,theta)
    rmul!(W, 1.0/Meff)
    Zint=round.(Int,Z)
    return W, Zint,N,M,q
end

function computep0(var)
    @extract var : W Z q
    p0 = zeros(q)
    for i in 1:length(W)
        p0[Z[1,i]] += W[i]
    end
    p0
end

function compute_empirical_freqs(Z::AbstractArray{Ti,2},W::AbstractVector{Float64},q::Ti) where Ti<:Integer
    N,M = size(Z)
    f = zeros(q,N)
    @inbounds for i in 1:N
        for s in 1:M
            f[Z[i,s],i] += W[s]
        end
    end
    f
end

function entropy(Z::AbstractArray{Ti,2},W::AbstractVector{Float64}) where Ti<:Integer
    N,M = size(Z)
    q = maximum(Z)
    f = compute_empirical_freqs(Z,W,q)
    S = zeros(N)
    @inbounds for i in 1:N
        _s = 0.0
        for a in 1:q
            _s -= f[a,i] > 0 ? f[a,i]*log(f[a,i]) : 0.0
        end
        S[i] = _s
    end
    S
end

function unpack_params(θ,arvar::ArVar)
    @extract arvar : N q

    arrJ = Array{Float64,3}[]
    arrH = Vector{Float64}[]
    ctr = 0
    for site in 1:N-1
        _arrJ = zeros(q,q,length(1:site))
        for i in 1:site
            for a in 1:q 
                for b in 1:q
                    ctr += 1
                    _arrJ[b,a,i]=θ[ctr]
                end
            end
        end
        push!(arrJ,_arrJ)
        _arrH = zeros(q)
        for a in 1:q
            ctr += 1
            _arrH[a] = θ[ctr]
        end
        push!(arrH,_arrH)
    end
    @assert ctr == length(θ)
    computep0(arvar),arrJ,arrH
end

# almost literally taken from Flux.jl 
function softmax(xs::AbstractArray)
    max_ = maximum(xs)
    exp_ = exp.(xs .- max_)
    exp_ ./ sum(exp_)
end

function sample(arnet::ArNet,msamples)
    @extract arnet : H J p0 idxperm
    q = length(p0)
    N = length(H) # here N is N-1 !!
    backorder = sortperm(idxperm)
    totH = Vector{Float64}(undef,q)
    res = @inbounds @distributed hcat for i in 1:msamples
        sample_z = Vector{Int}(undef,N+1)
        sample_z[1] = wsample(1:q,p0)
        for site in 1:N
            Js = J[site]
            h = H[site]
            for a in 1:q
                _s=0.0
                for i in 1:site
                    _s += Js[a,sample_z[i],i]
                end
                _s += h[a]            
                totH[a] = _s
            end
            p = softmax(totH)
            sample_z[site+1] = wsample(1:q,p)
        end
        sample_z
    end
    permuterow!(res,backorder)
end

function sample_with_weights(arnet::ArNet,msamples)
    @extract arnet : H J p0 idxperm
    q = length(p0)
    N = length(H) # here N is N-1 !!
    backorder = sortperm(idxperm)
    totH = Vector{Float64}(undef,q)
    W = SharedArray{Float64}(msamples)
    res = @inbounds @distributed hcat for i in 1:msamples
        sample_z = Vector{Int}(undef,N+1)
        sample_z[1] = wsample(1:q,p0)
        logw = p0[sample_z[1]]
        for site in 1:N
            Js = J[site]
            h = H[site]
            for a in 1:q
                _s=0.0
                for i in 1:site
                    _s += Js[a,sample_z[i],i]
                end
                _s += h[a]            
                totH[a] = _s
            end
            p = softmax(totH)
            sample_z[site+1] = wsample(1:q,p)
            logw += p[sample_z[site+1]]
        end
        W[i] = exp(logw)
        sample_z
    end
    permuterow!(res,backorder)
    W/sum(W), res
end

function permuterow!(x::AbstractMatrix,p::Vector)
    isperm(p) || error("not a permutation")
    for j in 1:size(x,2)
        vx = @view x[:,j]
        Base.permute!(vx,p)
    end
    x
end

function permuterow!(x::AbstractVector,p::Vector)  
    isperm(p) || error("not a permutation")
    Base.permute!(x,p)
end

# warning the gauge of H[:,1] is to be determined !!
function tensorize(arnet::ArNet) 
    @extract arnet : J H idxperm p0
    N = length(idxperm)
    q = length(H[1])
    outJ = zeros(q,q,N,N)
    outH = zeros(q,N)
    shiftH0 = sum(log.(p0))/q
    outH[:,idxperm[1]] .= log.(p0) .- shiftH0
    for i in 1:N-1
        si = idxperm[i+1]
        Js = J[i]
        outH[:,si] .= H[i]
        for j in 1:i
            sj = idxperm[j]            
            outJ[:,:,si,sj] .= Js[:,:,j]
            outJ[:,:,sj,si] .= Js[:,:,j]'
        end
    end
    outJ,outH
end