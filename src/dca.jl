function epistatic_score(arnet::ArNet, arvar::ArVar, seqid::Int; pc::Float64=0.1,min_separation::Int=1)
    @extract arnet : H J p0 idxperm
    @extract arvar : Z M N q 

    1 ≤ seqid ≤ M || error("seqid=$seqid should be in the interval [1,...,$M]")

    Da = zeros(q,N)
    Dab = zeros(q,q,N,N)

    xori = Z[:,seqid]
    xmut = copy(xori)
    
    arlike = zeros(N)
    ppc = (1-pc) * p0 + pc * ones(q)/q
    #_outputarnet!(arlike,xmut, J, H, ppc, N, q)
    # E0 =  sum(log.(arlike))
    # E0=0.0
    @inbounds for i in 1:N
        for a in 1:q
            xmut[i] = a
            _outputarnet!(arlike,xmut, J, H, ppc, N, q)
            Da[a,i] = -sum(log.(arlike))
        end
        xmut[i] = xori[i] #reset xmut to the original velue 
    end  
    
    @inbounds for i in 1:N-1
        for a in 1:q
            xmut[i] = a
            for j in i+1:N
                for b in 1:q
                    xmut[j] = b
                    _outputarnet!(arlike,xmut,J,H,ppc,N,q)        
                    Dab[b,a,j,i] = -sum(log.(arlike))
                    #Dab[a,b,i,j] = Dab[b,a,j,i]
                end
                xmut[j] = xori[j]
            end
        end
        xmut[i] = xori[i]
    end
    
    Jret = zeros(q,q,N,N)
    @inbounds for i in 1:N-1 
        for j in i+1:N
            for a in 1:q
                for b in 1:q 
                    Jret[b,a,j,i] = Dab[b,a,j,i] - Da[b,j] - Da[a,i]
                    #Jret[a,b,i,j] = Jret[b,a,j,i]
                end
            end
        end
    end
    Jzsg = zsg(Jret)
    FN = compute_APC(Jzsg, N, q)
    score = compute_ranking(FN, min_separation)
    
    permtuple=Tuple{Int,Int,Float64}[]
    sizehint!(permtuple,length(permtuple))
    for s in score
        si,sj,val= idxperm[s[1]],idxperm[s[2]],s[3]
        if si > sj 
            si,sj = sj,si
        end
        push!(permtuple,(si,sj,val))
    end
    return permtuple
end

function zsg(J::Array{Float64,4})
    q,q,N,N = size(J)
    Jzsg = zeros(q,q,N,N)
    @inbounds for i in 1:N-1
        for j in i+1:N
            Jzsg[:,:,j,i] .= J[:,:,j,i] - repeat(sum(J[:,:,j,i], dims=1)/q, q, 1) - repeat(sum(J[:,:,j,i], dims=2)/q, 1, q) .+ sum(J[:,:,j,i])/q^2
            #Jzsg[:,:,i,j] .= Jzsg[:,:,j,i]'
        end 
    end
    Jzsg
end

function compute_APC(J::Array{Float64,4},N,q)
    FN = fill(0.0, N,N)
    @inbounds for i=1:N-1
        for j=i+1:N
            FN[j,i] = norm(J[1:q-1,1:q-1,j,i],2)
            FN[i,j] = FN[j,i]
        end
    end
    FN = correct_APC(FN)
    return FN
end

function correct_APC(S::Matrix)
    N = size(S, 1)
    Si = sum(S, dims=1)
    Sj = sum(S, dims=2)
    Sa = sum(S) * (1 - 1/N)

    S -= (Sj * Si) / Sa
    return S
end

function compute_ranking(S::Matrix{Float64}, min_separation::Int = 5)

    N = size(S, 1)
    R = Array{Tuple{Int,Int,Float64}}(undef, div((N-min_separation)*(N-min_separation+1), 2))
    counter = 0
    for i = 1:N-min_separation, j = i+min_separation:N
        counter += 1
        R[counter] = (i, j, S[j,i])
    end

    sort!(R, by=x->x[3], rev=true)
    return R

end
