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
    tgpu_chol = @elapsed gpucg(K, f, zeros(size(f)), tol, 5000, :chol)
    tgpu_jac = @elapsed gpucg(K, f, zeros(size(f)), tol, 5000, :jac)
    tgpu = @elapsed gpucg(K, f, zeros(size(f)), tol, 5000, :none)

    K32 = convert(SparseMatrixCSC{Float32, Int}, K)
    f32 = convert(Vector{Float32}, f)
    tgpu_32 = 0.0 #@elapsed gpucg(K32, f32, zeros(Float32, size(f)), tol, 5000, false)
    return length(f), tchol, tgpu, tgpu_chol, tgpu_jac, tgpu_32
end


tgpus = Float64[]
tgpus32 = Float64[]
tgpus_pc = Float64[]
tchols = Float64[]
ns = [50, 100]
for n in ns
    dofs, tchol, tgpu, tgpu_chol, tgpu_32 = analyze(n, n)
    println("dofs: $dofs")
    push!(tchols, tchol)
    push!(tgpus_pc, tgpu_chol)
    push!(tgpus, tgpu)
    push!(tgpus32, tgpu_32)
end

plot(ns.*ns, tchols; label="PC Direct Cholesky")
plot!(ns.*ns, tgpus; label="GPU CG")
plot!(ns.*ns, tgpus_pc; label="GPU CG Part. Chol. Precond")
plot!(ns.*ns, tgpus32; label="GPU CG 32")
xlabel!("Elements")
ylabel!("Time")

