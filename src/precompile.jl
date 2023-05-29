using PrecompileTools

@setup_workload begin
    N = 10
    M = 10
    Z = rand(1:21, N, M)
    W = rand(M)
    W ./= sum(W)
    @compile_workload begin
        redirect_stdout(devnull) do
            arnet,arvar = ardca(Z, W)
            res2 = epistatic_score(arnet,arvar,1)
        end
    end
end