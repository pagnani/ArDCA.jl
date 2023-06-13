function read_fasta(filename::AbstractString, max_gap_fraction::Real, theta::Any, remove_dups::Bool)
    Z = read_fasta_alignment(filename, max_gap_fraction)
    if remove_dups
        Z, _ = remove_duplicate_sequences(Z)
    end
    N, M = size(Z)
    q = round(Int, maximum(Z))
    W, Meff = compute_weights(Z,  theta)
    return W, Z, N, M, q
end

function computep0(var)
    @extract var:W Z q
    p0 = zeros(q)
    for i in 1:length(W)
        p0[Z[1,i]] += W[i]
    end
    return p0
end

function compute_empirical_freqs(Z::AbstractArray{Ti,2}, W::AbstractVector{Float64}, q::Ti) where Ti <: Integer
    N, M = size(Z)
    f = zeros(q, N)
    @inbounds for i in 1:N
        for s in 1:M
            f[Z[i,s],i] += W[s]
        end
    end
    return f
end

function entropy(Z::AbstractArray{Ti,2}, W::AbstractVector{Float64}) where Ti <: Integer
    N, M = size(Z)
    q = maximum(Z)
    f = compute_empirical_freqs(Z, W, q)
    S = zeros(N)
    @inbounds for i in 1:N
        _s = 0.0
        for a in 1:q
            _s -= f[a,i] > 0 ? f[a,i] * log(f[a,i]) : 0.0
        end
        S[i] = _s
    end
    return S
end

function unpack_params(θ, arvar::ArVar)
    @extract arvar:N q

    arrJ = Array{Float64,3}[]
    arrH = Vector{Float64}[]
    ctr = 0
    for site in 1:N - 1
        _arrJ = zeros(q, q, length(1:site))
        for i in 1:site
            for a in 1:q 
                for b in 1:q
                    ctr += 1
                    _arrJ[b,a,i] = θ[ctr]
                end
            end
        end
        push!(arrJ, _arrJ)
        _arrH = zeros(q)
        for a in 1:q
            ctr += 1
            _arrH[a] = θ[ctr]
        end
        push!(arrH, _arrH)
    end
    @assert ctr == length(θ)
    return computep0(arvar), arrJ, arrH
end

function softmax!(r::Vector{Float64}, x::Vector{Float64})
    n = length(x)
    u = maximum(x)
    s = 0.
    @inbounds @avx for i = 1:n
        r[i] = exp(x[i] - u)
        s += r[i]
    end
    invs =  inv(s)
    @inbounds @avx for i = 1:n
        r[i] *= invs
    end
    return r
end

"""
    softmax(x::AbstractArray{<:Real})
Return the [`softmax transformation`](https://en.wikipedia.org/wiki/Softmax_function) applied to `x`
"""
softmax!(x::Vector{Float64}) = softmax!(x, x)
softmax(x::Vector{Float64}) = softmax!(similar(x, Float64), x)

"""
    sample(arnet::ArNet, msamples::Int)
Return a generated alignment in the form of a `N × msamples`  matrix of type `::Matrix{Int}`  
# Examples
```
julia> arnet,arvar=ardca("file.fasta",verbose=true,permorder=:ENTROPIC, lambdaJ=0.001,lambdaH=0.001);
julia> Zgen=Zgen=sample(arnet,1000);
```
"""
function sample(arnet::ArNet, msamples::Int)
    @extract arnet:H J p0 idxperm
    q = length(p0)
    N = length(H) # here N is N-1 !!
    backorder = sortperm(idxperm)
    totH = Vector{Float64}(undef, q)
    res = @inbounds @distributed hcat for i in 1:msamples
        sample_z = Vector{Int}(undef, N + 1)
        sample_z[1] = wsample(1:q, p0)
        for site in 1:N
            Js = J[site]
            h = H[site]
            copy!(totH,h)
            @avx for i in 1:site
                for a in 1:q
                    totH[a] += Js[a,sample_z[i],i]
                end
            end
            p = softmax(totH)
            sample_z[site + 1] = wsample(1:q, p)
        end
        sample_z
    end
    permuterow!(res, backorder)
end

"""
    sample_with_weights(arnet::ArNet, msamples::Int)
Return a generated alignment in the form of a `N × msamples`  matrix of type `::Matrix{Int}` and the relative probabilities under the module

# Examples
```
julia> arnet,arvar=ardca("file.fasta",verbose=true,permorder=:ENTROPIC, lambdaJ=0.001,lambdaH=0.001);
julia> Wgen,Zgen=sample_with_weights(arnet,1000);
```
"""
function sample_with_weights(arnet::ArNet, msamples)
    @extract arnet:H J p0 idxperm
    q = length(p0)
    N = length(H) # here N is N-1 !!
    backorder = sortperm(idxperm)
    totH = Vector{Float64}(undef, q)
    W = SharedArray{Float64}(msamples)
    res = @inbounds @distributed hcat for i in 1:msamples
        sample_z = Vector{Int}(undef, N + 1)
        sample_z[1] = wsample(1:q, p0)
        logw = log(p0[sample_z[1]])
        for site in 1:N
            Js = J[site]
            h = H[site]
            copy!(totH,h)
            @avx for i in 1:site
                for a in 1:q
                    totH[a] += Js[a,sample_z[i],i]
                end
            end
            p = softmax(totH)
            sample_z[site + 1] = wsample(1:q, p)
            logw += log(p[sample_z[site + 1]])
        end
        W[i] = exp(logw)
        sample_z
    end
    permuterow!(res, backorder)
    return W , res
end

function permuterow!(x::AbstractMatrix, p::Vector)
    isperm(p) || error("not a permutation")
    for j in 1:axes(x, 2)
        vx = @view x[:,j]
        Base.permute!(vx, p)
    end
    return x
end

function permuterow!(x::AbstractVector, p::Vector)  
    isperm(p) || error("not a permutation")
    Base.permute!(x, p)
end

# warning the gauge of H[:,1] is to be determined !!
function tensorize(arnet::ArNet; tiny::Float64=1e-16) 
    @extract arnet:J H idxperm p0
    N = length(idxperm)
    q = length(H[1])
    p0pc = (1.0-tiny)*p0 .+ tiny/q
    outJ = zeros(q, q, N, N)
    outH = zeros(q, N)
    shiftH0 = sum(log.(p0pc)) / q
    outH[:,idxperm[1]] .= log.(p0pc) .- shiftH0
    for i in 1:N - 1
        si = idxperm[i + 1]
        Js = J[i]
        outH[:,si] .= H[i]
        for j in 1:i
            sj = idxperm[j]            
            outJ[:,:,si,sj] .= Js[:,:,j]
            outJ[:,:,sj,si] .= Js[:,:,j]'
        end
    end
    return outJ, outH
end

"""
    dms_single_site(arnet::ArNet, arvar::ArVar, seqid::Int; pc::Float64=0.1)
    
Return a `q×L` matrix of containing `-log(P(mut))/log(P(seq))` for all single
site mutants of the reference sequence `seqid`, and a vector of the indices of
the residues of the reference sequence that contain gaps (i.e. the 21
amino-acid) for which the score has no sense and is set by convention to `+Inf`.
A negative value indicate a beneficial mutation, a value 0 indicate
the wild-type amino-acid.
"""
function dms_single_site(arnet::ArNet, arvar::ArVar, seqid::Int; pc::Float64=0.1)
    @extract arnet : H J p0 idxperm
    @extract arvar : Z M N q 

    1 ≤ seqid ≤ M || error("seqid=$seqid should be in the interval [1,...,$M]")
    
    ppc = (1-pc) * p0 + pc * ones(q)/q
    Da = fill(Inf64,q,N)
    xori = Z[:,seqid]
    xmut = copy(xori)
    idxnogap=findall(x->x!=21,xori)
    arlike = zeros(N)
    arlike0 = zeros(N)
    _outputarnet!(arlike0,xori,J,H,ppc,N,q)
    ll0 = -sum(log.(arlike0)) 
    
    @inbounds for i in idxnogap
        if xori[i] == 21
            continue
        end
        for a in 1:q
            if a != xori[i]
                xmut[i] = a
                _outputarnet!(arlike,xmut, J, H, ppc, N, q)
                Da[a,i] = -sum(log.(arlike)) - ll0
            else
                Da[a,i] = 0.0
            end
        end
        xmut[i] = xori[i] #reset xmut to the original velue 
    end
    invperm = sortperm(idxperm)
    return Da[:,invperm],sort!(idxperm[setdiff(1:N,idxnogap)])
end