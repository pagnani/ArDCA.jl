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
                 epsconv::Real=1e20,
                 lambdaJ::Real=0.0,
                 lambdaH::Real=0.0,
                 asym::Bool=true,
                 maxit::Integer=1000,
                 method::Symbol=:LD_LBFGS,
                 epstest::Real=1e-5,
                 permorder::Union{Symbol,Vector{Int}}=:NATURAL)

    W, Z, J, h = generateWZJh(N, q)
    arnet, arvar, psval = ardca(Z,W,
                    lambdaJ=lambdaJ,
                    lambdaH=lambdaH,
                    epsconv=epsconv,
                    verbose=verbose,
                    maxit=maxit,
                    method=method,
                    permorder=permorder)
    
    #Jzsg, hzsg = PottsGauge.gauge(J, h, ZeroSumGauge())
    
    @test 10 * log10(sum(abs2, (sum(log.(arnet(arvar)), dims=1) |> vec .|> exp) - W) / length(W)) < 70
    arnet,arvar,psval,J,h,W,Z
end

for order in [:NATURAL, :ENTROPIC, :REV_ENTROPIC, :RANDOM]
    for q in 2:3:4
        for N in 2:5
            testDCA(N, q, permorder=order)
        end 
    end
end
printstyled("All TestDCA passed!\n",color=:light_green,bold=true)
end # end