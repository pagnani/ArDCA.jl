struct ArVar{Ti <: Integer}
    N::Int
    M::Int
    q::Int
    q2::Int
    lambdaJ::Float64
    lambdaH::Float64
    Z::SharedArray{Ti,2}
    W::SharedArray{Float64,1}
    IdxZ::SharedArray{Int,2} # partial index computation to speed up energy calculation
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
        
        IdxZ = Array{Int,2}(undef, N, M)
        q2 = q * q
        for i in 1:M
            for j in 1:N
                IdxZ[j,i] = (j - 1) * q2 + q * (Z[j,i] - 1)
            end
        end
        sIdxZ = SharedArray{Int}(size(IdxZ))
        sIdxZ[:] = IdxZ
        new{Ti}(N, M, q, q^2, lambdaJ, lambdaH, sZ, sW, sIdxZ,idxperm)
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
        _outputarnet(Z[:,i], J, H, p0, N, q)
    end
    permuterow!(output,backorder)
end

function _outputarnet( xs, J, H, p0, N, q)
    dest = Vector{Float64}(undef,N)
    _outputarnet!(dest, xs, J, H, p0, N, q)
end

let DtotH = Dict{Int,Vector{Float64}}()
    global _outputarnet!
    function _outputarnet!(dest, xs, J, H, p0, N, q)
        x = sdata(xs)
        dest[1] = p0[x[1]]
        totH = Base.get!(DtotH,q,Vector{Float64}(undef,q))
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




