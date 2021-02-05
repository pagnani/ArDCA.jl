module TestDCA
using Test, ArDCA
function energy(x, J, h)
    q, q, N, N = size(J)
    eJ = 0.0
    @inbounds @simd for i in 1:N - 1
        for j in i + 1:N
            eJ -= J[x[i],x[j],i,j]
        end
    end
    eh = 0.0
    @inbounds @simd for i in 1:N
        eh -= h[x[i],i]
    end
    return eJ + eh
end

function complete_dataset(N, q)
    Z = zeros(Int, N, q^N)
    bounds = ntuple(x -> q, N)
    ctr = 0
    @inbounds for i in CartesianIndices(bounds)
        ctr += 1
        for j in 1:N
            Z[j,ctr] = i[j]
        end
    end
    Z
end


function generateWZJh(N, q)
    Jasym = randn(q, q, N, N)
    h = randn(q, N)
    J = 0.5 * (permutedims(Jasym, [2,1,4,3]) + Jasym)
    for i in 1:N
        for a in 1:q
            for b in 1:q
                J[a,b,i,i] = 0.0
            end
        end
    end
    Z = complete_dataset(N, q)
    W = [exp(-energy(Z[:,i], J, h)) for i in 1:q^N]
    W .= W ./ sum(W)
    return W, Z, J, h
end

function testDCA(N,q;
                 verbose::Bool=false,
                 epsconv::Real=1e-50,
                 lambdaJ::Real=0.0,
                 lambdaH::Real=0.0,
                 maxit::Integer=10000,
                 method::Symbol=:LD_LBFGS,
                 permorder::Union{Symbol,Vector{Int}}=:NATURAL,
                 dBthreshold::Real=-40)

    W, Z, J, h = generateWZJh(N, q)
    arnet, arvar = ardca(Z,W,
                    lambdaJ=lambdaJ,
                    lambdaH=lambdaH,
                    epsconv=epsconv,
                    verbose=verbose,
                    maxit=maxit,
                    method=method,
                    permorder=permorder)

    
    dBval = 10 * log10(sum(abs2, (sum(log.(arnet(arvar)), dims=1) |> vec .|> exp) - W) / length(W))
    println("testing N=$N\tq=$q\tpermorder=$permorder\tΔ=$dBval [dB]")
    @test dBval < dBthreshold
end
for q in 2:4
    for N in 2:5
        for order in [:NATURAL,:RANDOM, :ENTROPIC, :REV_ENTROPIC]
            testDCA(N, q, permorder=order,epsconv=1e-20)
        end
        println()
    end
end
printstyled("All TestDCA passed!\n",color=:light_green,bold=true)
end # end