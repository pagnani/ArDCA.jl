struct ArVar{Ti <: Integer}
    N::Int
    M::Int
    q::Int
    q2::Int
    lambdaJ::Float64
    lambdaH::Float64
    Z::SharedArray{Ti,2}
    W::SharedArray{Float64,1}
    IdxZ::SharedArray{Ti,2} # partial index computation to speed up energy calculation
    idxperm::Array{Ti,1}
    
    function ArVar(N, M, q, lambdaJ, lambdaH, Z::Array{Ti,2}, W::Array{Float64,1}, permorder::Union{Symbol,Vector{Ti}}) where Ti <: Integer
        idxperm = if typeof(permorder) == Symbol
            S = entropy(Z,W)
            if permorder === :ENTROPIC
                sortperm(S)
            elseif permorder === :REV_ENTROPIC
                sortperm(S,rev=true)
            elseif permorder === :RANDOM
                randperm(N)
            elseif permorder === :NATURAL
                collect(1:N)
            else
                error("the end of the world has come")
            end
        elseif typeof(permorder) <: Vector{Ti}
            isperm(permorder) && (permorder)
        else
            error("permorder can only be a Symbol or a Vector")
        end
        permuterow!(Z,idxperm)
        
        sZ = SharedArray{Ti}(size(Z))
        sZ[:] = Z
        sW = SharedArray{Float64}(size(W))
        sW[:] = W
        

        IdxZ = Array{Ti,2}(undef, N, M)
        q2 = q * q
        for i in 1:M
            for j in 1:N
                IdxZ[j,i] = Ti(j - 1) * Ti(q2) + Ti(q) * (Z[j,i] - one(Ti))
            end
        end
        sIdxZ = SharedArray{Ti}(size(IdxZ))
        sIdxZ[:] = IdxZ
        new{Ti}(N, M, q, q^2, lambdaJ, lambdaH, sZ, sW, sIdxZ,idxperm)
    end
end

struct ArAlg
    method::Symbol
    verbose::Bool
    epsconv::Float64
    maxit::Int
end

struct ArNet
    idxperm::Array{Int,1}
    p0::Array{Float64,1}
    J::Array{Array{Float64,3},1}
    H::Array{Array{Float64,1},1}
end

ArNet(θ,var::ArVar) = ArNet(var.idxperm, unpack_params(θ, var)...)

function (A::ArNet)(x::AbstractVector{T}) where T <: Integer 
    @extract A:J H p0 idxperm
    backorder = sortperm(idxperm)
    N = length(x)
    length(H) == N - 1 || throw(DimensionMismatch("incompatible size between input and fields"))
    q = length(p0)
    permuterow!(x,idxperm)
    permuterow!(_outputarnet(x, J, H, p0, N, q),backorder)
end

function (A::ArNet)(x::AbstractMatrix{T}) where T <: Integer 
    @extract A : J H p0 idxperm
    backorder = sortperm(idxperm)
    N, M = size(x)
    length(H) == N - 1 || throw(DimensionMismatch("incompatible size between input and fields"))
    q = length(p0)
    permuterow!(x,idxperm)
    output = @distributed hcat for i in 1:M
        vx = @view x[:,i]        
        _outputarnet(vx, J, H, p0, N, q)
    end
    permuterow!(output,backorder)
end

function (A::ArNet)(arvar::ArVar)
    @extract A : J H p0 idxperm
    @extract arvar : Z q N M
    backorder=sortperm(idxperm)
    output = @distributed hcat for i in 1:M
        vx = @view Z[:,i]        
        _outputarnet(vx, J, H, p0, N, q)
    end
    permuterow!(output,backorder)
end

function _outputarnet(xs, J, H, p0, N, q)
    x = sdata(xs)
    dest = zeros(N)
    dest[1] = p0[x[1]]
    totH = zeros(q)
    @inbounds for site in 1:N-1
        Js = J[site]
        h = H[site]
        for a in 1:q
            totH[a] = 0.0
            @simd for i in 1:site
                totH[a] += Js[a,x[i],i]
            end
            totH[a] += h[a]
        end
        dest[site+1]=softmax(totH)[x[site+1]]
    end
    dest
end