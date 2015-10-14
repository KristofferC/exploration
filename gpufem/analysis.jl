#Pkg.add("Plots")
#Pkg.clone("https://github.com/KristofferC/JuAFEM.jl")
#Pkg.add("https://github.com/KristofferC/CUSPARSE.jl")
#Pkg.checkout("CUSPARSE", "kc/csrsv!")
#Pkg.add("CUDArt")
#Pkg.add("CUBLAS")

using Plots
using JuAFEM

include("setupfem.jl")
include("gpusolve.jl")

function analyze(nx, ny)
    K, f = setup_femprob(nx, ny)

    tchol = @elapsed u = cholfact(Symmetric(K, :U)) \ f
    tol = norm(K * u - f)
    tgpu = @elapsed gpucg(K, f, rand(size(f)), tol, 1000)
    return tchol, tgpu
end


tgpus = Float64[]
tchols = Float64[]
ns = [10, 50, 100, 150, 200]
for n in ns
    println("$n")
    tchol, tgpu = analyze(n, n)
    push!(tchols, tchol)
    push!(tgpus, tgpu)
end

plot(ns.*ns, tchols; label="PC Direct Cholesky")
plot!(ns.*ns, tgpus; label="GPU CG Part. Chol. Precond.")
xlabel!("Elements")
ylabel!("Time")
