using DelimitedFiles
mutable struct AnalOut
    arnet
    arvar
    resori::Union{PlmOut,Nothing}
    resgen::Union{PlmOut,Nothing}
    resdca::Union{Nothing,Array{Tuple{Int64,Int64,Float64}}}
    rocori::Union{Nothing,Array{Tuple{Int64,Int64,Float64,Float64}}}
    rocgen::Union{Nothing,Array{Tuple{Int64,Int64,Float64,Float64}}}
    rocdca::Union{Nothing,Array{Tuple{Int64,Int64,Float64,Float64}}}
    Zgen::Union{Nothing,Array{Int,2}}
end

function runardca!(out, familyid; kwds...)
    filefasta = joinpath(familyid,filter(x->endswith(x,".fasta.gz"), readdir(familyid))[1])
    arnet,arvar=ardca(filefasta; kwds...)
    out.arnet, out.arvar = arnet,arvar
end

function sample_and_analyze_results!(out::AnalOut,familyid::String; theta=:auto,Msample=nothing)
    #resnet,resori,resgen,rocori,rocgen = out.resnet, out.resori, out.resgen,out.rocori, out.rocgen
    filefasta = joinpath(familyid,filter(x->endswith(x,".fasta.gz"), readdir(familyid))[1])
    filescore = joinpath(familyid, "tab_cont")
    M = Msample===nothing ? out.arvar.M : Msample
    Wgen,Zgen= ArDCA.sample_with_weights(out.arnet,M)
    out.Zgen = Zgen
    out.resgen = plmdca_asym(Zgen,ones(size(Zgen,2))./size(Zgen,2))
    out.resdca = epistatic_score(out.arnet,out.arvar,1)
    out.rocgen = computescore(out.resgen.score,filescore)
    out.rocdca = computescore(out.resdca,filescore)
    if out.resori === nothing 
        out.resori = plmdca_asym(filefasta, theta=theta)
        out.rocori = computescore(out.resori.score,filescore)
    end
    nothing
end

function computescore(score,filedist::String; mindist::Int=4, cutoff::Float64=7.0)
    d = readdlm(filedist)
    dist = Dict((round(Int,d[i,1]),round(Int,d[i,2])) => d[i,4] for i in axes(d,1))
    nc2 = length(score)
    #nc2 == size(d,1) || throw(DimensionMismatch("incompatible length $nc2 $(size(d,1))"))
    out = Tuple{Int,Int,Float64,Float64}[]
    ctrtot = 0
    ctr = 0
    for i in 1:nc2
        sitei,sitej,plmscore = score[i][1],score[i][2], score[i][3]
        dij = if haskey(dist,(sitei,sitej)) 
            dist[(sitei,sitej)]
        else
           continue
        end
        if sitej - sitei > mindist
            ctrtot += 1
            if dij < cutoff
                ctr += 1
            end
            push!(out,(sitei,sitej, plmscore, ctr/ctrtot))
        end
    end 
    out
end

function plotres(out; xlim=(),scale=:log)
    lJ,lH = out.arvar.lambdaJ,out.arvar.lambdaH
    plotfun = (scale===:log) ? semilogx : plot
    close("all") 
    plotfun(map(x->x[4],out.rocori))
    plotfun(map(x->x[4],out.rocgen))
    plotfun(map(x->x[4],out.rocdca))
    plt.legend(["plm","generated  λJ = $lJ λH=$lH","dca score"])
    plt.xlim(xlim...)
end
#plmpfam = plmdca_asym(filefasta); #compute score from original data
#resnet=m.create_RPM_network(recorr,depth=1,idxperm=1:53); #crea AutoRegressinve network
#alphabet = [ 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21]
#          [ A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y,-]

let 
    num2let = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I','K', 'L','M', 'N','P','Q', 'R', 'S','T', 'V', 'W', 'Y','-']
    global write_fasta
    function write_fasta(filedest::String, out::AnalOut)
        N,M = size(out.Zgen)
        fp = open(filedest,"w")
        #fp=stdout
        for s in 1:M
            println(fp,"> Seq $s")
            for a in 1:N
                print(fp,num2let[Int(out.Zgen[a,s])])
            end
            println(fp)
        end
        close(fp)
    end
end

function write_score(filedest::String,out::AnalOut)

    fp = open(filedest,"w")
    for el in out.resgen.score
        println(fp, el[1]," ",el[2]," ",el[3])
    end
    close(fp)
end

function runall(PFfile::String)

    datadir=joinpath(ENV["HOME"],"Dropbox","five families autoregressive")

    familyid = joinpath(datadir,PFfile)
    fileal = joinpath(familyid,filter(x->endswith(x,".fasta"), readdir(familyid))[1])
    filedist = joinpath(familyid, "tab_cont")

    resplm = plmdca(fileal)
    rocplm = computescore(resplm.score,filedist)
    arnet,arvar=ardca(fileal,verbose=true,permorder=:ENTROPIC, lambdaJ=0.02,lambdaH=0.001) 
    arscore=ArDCA.epistatic_score(arnet,arvar,1, pc=0.1) 
    rocar = computescore(arscore,filedist);
    arnet1,arvar1=ardca(fileal,verbose=true,permorder=:ENTROPIC, lambdaJ=0.0001,lambdaH=0.0001);
    Zgen=sample(arnet1,arvar.M)
    resgen=plmdca(Zgen,ones(size(Zgen,2))./size(Zgen,2))
    rocgen=computescore(resgen.score, filedist)
    close("all"); 
    semilogx(map(x->x[4],rocplm))
    semilogx(map(x->x[4],rocgen))
    semilogx(map(x->x[4],rocar))
    plt.legend(["plmDCA on alignment","plmDCA on generated","ΔΔE score"])
    gcf()
    return AnalOut(arnet,arvar,
                    resplm,
                    resgen,
                    arscore, 
                    rocplm,
                    rocgen,
                    rocar,Zgen)
end