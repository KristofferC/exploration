#Pkg.add("Plots")
#Pkg.clone("https://github.com/KristofferC/JuAFEM.jl")
#Pkg.add("https://github.com/KristofferC/CUSPARSE.jl")
#Pkg.checkout("CUSPARSE", "kc/csrsv!")
#Pkg.clone("CUDArt")
#Pkg.checkout("CUSPARSE", "kc/csrsv!")
#Pkg.add("CUBLAS")

using Plots
using JuAFEM
pyplot()

include("setupfem.jl")
include("gpusolve.jl")
include("cpusolve.jl")

function analyze(ns, doplot=true)

    tchols = Float64[]
    tgpus = Float64[]
    tgpus_32 = Float64[]
    tgpus_chol = Float64[]
    tgpus_jac = Float64[]
    tcpus = Float64[]
    tcpus_32 = Float64[]

    kgpus = Int[]
    kgpus_32 = Int[]
    kgpus_chol = Int[]
    kgpus_jac = Int[]
    kcpus = Int[]
    kcpus_32 = Int[]

    dofs = Int[]
    for n in ns
        maxiters = 5000
        nx = ny = n
        K, f = setup_femprob(nx, ny)
        push!(dofs, length(f))

        tchol = @elapsed u = cholfact(Symmetric(K, :U)) \ f
        push!(tchols, tchol)
        tol = norm(K * u - f)

        #println(gpucg(K, f, zeros(size(f)), tol, maxiters, :chol))
        #tgpu_chol = @elapsed kgpu_chol = gpucg(K, f, zeros(size(f)), tol, maxiters, :chol)
        #println(kgpu_chol)
        #push!(tgpus_chol, tgpu_chol)
        #push!(kgpus_chol, kgpu_chol)
#
        #tgpu_jac = @elapsed kgpu_jac = gpucg(K, f, zeros(size(f)), tol, maxiters, :jac)
        #push!(tgpus_jac, tgpu_jac)
        #push!(kgpus_jac, kgpu_jac)

        tgpu = @elapsed kgpu = gpucg(K, f, zeros(size(f)), tol, maxiters, :none)
        push!(tgpus, tgpu)
        push!(kgpus, kgpu)

        tcpu = @elapsed kcpu = cpucg(K, f, zeros(size(f)), tol, maxiters)
        push!(tcpus, tcpu)
        push!(kcpus, kcpu)

        K32 = convert(SparseMatrixCSC{Float32, Int}, K)
        f32 = convert(Vector{Float32}, f)
        tgpu_32 = @elapsed kgpu_32 = gpucg(K32, f32, zeros(Float32, size(f)), tol, maxiters, :none)
        push!(tgpus_32, tgpu_32)
        push!(kgpus_32, kgpu_32)

        tcpu_32 = @elapsed kcpu_32 = cpucg(K32, f32, zeros(Float32, size(f)), tol, maxiters)
        push!(tcpus_32, tcpu_32)
        push!(kcpus_32, kcpu_32)

        println("------------------------")

    end


    if doplot
        p = plot()
        plot!(dofs, tchols; label="PC Direct Cholesky")
        #plot!(dofs, tgpus_chol; label="GPU CG Inc. Chol.")
       # plot!(dofs, tgpus_jac; label="GPU CG Jac")
        plot!(dofs, tgpus; label="GPU CG None")
        plot!(dofs, tgpus_32; label="GPU CG 32 None")
        plot!(dofs, tcpus; label="CPU CG None")
        plot!(dofs, tcpus_32; label="CPU CG 32 None")
        xlabel!("dofs")
        ylabel!("Time")
        display(p)

       # p = plot()
       # plot!(dofs, kgpus_chol; label="GPU CG Inc. Chol.")
       # plot!(dofs, kgpus_jac; label="GPU CG Jac")
       # plot!(dofs, kgpus; label="GPU CG None")
       # plot!(dofs, kgpus_32; label="GPU CG 32 None")
       # plot!(dofs, kcpus; label="CPU CG None")
       # xlabel!("dofs")
       # ylabel!("Iterations until convergence")
       # display(p)
    end
end

ns = [5, 50, 100, 150, 200, 250, 300]
analyze(ns)

