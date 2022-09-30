struct ArVar{Ti <: Integer}
    N::Int
    M::Int
    q::Int
    q2::Int
    lambdaJ::Float64
    lambdaH::Float64
    Z::Array{Ti,2}
    W::Array{Float64,1}
    IdxZ::Array{Int,2} # partial index computation to speed up energy calculation
    idxperm::Array{Int,1}
    
    function ArVar(N, M, q, lambdaJ, lambdaH, Z::Array{Ti,2}, W::Array{Float64,1}, permorder::Union{Symbol,Vector{Int}}) where Ti <: Integer
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
        elseif typeof(permorder) <: Vector
            (length(permorder) != N) && error("length permorder ≠ $N")
            isperm(permorder) && (permorder)
        else
            error("permorder can only be a Symbol or a Vector")
        end
        permuterow!(Z,idxperm)
        IdxZ = Array{Int,2}(undef, N, M)
        q2 = q * q
        for i in 1:M
            for j in 1:N
                IdxZ[j,i] = (j - 1) * q2 + q * (Z[j,i] - 1)
            end
        end
        new{Ti}(N, M, q, q^2, lambdaJ, lambdaH, Z, W, IdxZ,idxperm)
    end
end

function Base.show(io::IO, arvar::ArVar)
    @extract arvar : N M q lambdaJ lambdaH Z
    print(io,"ArVar [N=$N M=$M q=$q λJ = $lambdaJ λH = $lambdaH Z::$(eltype(Z))]")
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

function Base.show(io::IO, arnet::ArNet)
    N = length(arnet.idxperm)
    q = length(arnet.H[1])
    print(io,"ArNet [N=$N q=$q]")
end

function (A::ArNet)(x::Vector{T}) where T <: Integer 
    @extract A:J H p0 idxperm
    backorder = sortperm(idxperm)
    N = length(x)
    length(H) == N - 1 || throw(DimensionMismatch("incompatible size between input and fields"))
    q = length(p0)
    permuterow!(x,idxperm)
    res = permuterow!(_outputarnet(x, J, H, p0, N, q),backorder)
    permuterow!(x,backorder)
    return res
end

function (A::ArNet)(x::Matrix{T}) where {T<:Integer}
    @extract A:J H p0 idxperm
    backorder = sortperm(idxperm)
    N, M = size(x)
    length(H) == N - 1 || throw(DimensionMismatch("incompatible size between input and fields"))
    q = length(p0)
    output = Array{eltype(p0),2}(undef, N, M)
    permuterow!(x, idxperm)
    #Threads.@threads
    for i in 1:M
        output[:, i] .= _outputarnet(view(x, :, i), J, H, p0, N, q)
    end
    res = permuterow!(output, backorder)
    permuterow!(x,backorder)
    return res
end

function (A::ArNet)(arvar::ArVar)
    @extract A:J H p0 idxperm
    @extract arvar:Z W
    backorder = sortperm(idxperm)
    N, M = size(Z)
    length(H) == N - 1 || throw(DimensionMismatch("incompatible size between input and fields"))
    q = length(p0)
    output = Array{eltype(p0),2}(undef, N, M)
    #permuterow!(x, idxperm)
    #Threads.@threads
    for i in 1:M
        output[:, i] .= _outputarnet(view(Z, :, i), J, H, p0, N, q) 
    end
    #res = permuterow!(output, backorder)
    return output
end

function _outputarnet( xs, J, H, p0, N, q)
    dest = Vector{Float64}(undef,N)
    _outputarnet!(dest, xs, J, H, p0, N, q)
end

let DtotH = Dict{Tuple{Int,Int},Vector{Float64}}()
    global _outputarnet!
    function _outputarnet!(dest, x, J, H, p0, N, q)
        dest[1] = p0[x[1]]
        totH = Base.get!(DtotH,(q,Threads.threadid()),Vector{Float64}(undef,q))
        #fill!(totH,0.0)
        @inbounds for site in 1:N-1
            Js = J[site]
            h = H[site]
            copy!(totH,h)
            @avx for i in 1:site
                for a in 1:q
                    totH[a] += Js[a,x[i],i]
                end
            end
            softmax!(totH)
            dest[site+1]=totH[x[site+1]]
        end
        dest
    end
end