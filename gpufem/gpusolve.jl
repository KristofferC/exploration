# Ref http://docs.nvidia.com/cuda/incomplete-lu-cholesky/

using CUSPARSE
using CUDArt
using CUBLAS

import Base: dot, copy!, norm, scale!

function gpucg(h_A, h_f, h_x₀, tol, max_iters, prec=:none)
    if !in(prec, (:chol, :jac, :none))
        throw(ArgumentError("Invalid precond method: $prec, available: :chol, :jac, :none"))
    end
    T = eltype(h_A)

    A = CudaSparseMatrixCSR(h_A)
    f = CUSPARSE.CudaArray(h_f)

    zₖm = CUSPARSE.CudaArray(zeros(eltype(h_f), length(h_f),1))
    zₖ = vec(zₖm)

    rₖm = CUSPARSE.CudaArray(zeros(eltype(h_f), length(h_f),1))
    rₖ = vec(rₖm)

    pₖ = CUSPARSE.CudaArray(zeros(h_f))
    Apₖ = CUSPARSE.CudaArray(zeros(h_f))
    xₖ = CUSPARSE.CudaArray(h_x₀)

    if prec == :jac
        Minv = CUSPARSE.CudaArray(1./diag(h_A))
    end

    # Compute incomplete cholesky
    if prec == :chol
        tₖ = CUSPARSE.CudaArray(zeros(h_f))
        M = CudaSparseMatrixCSR(triu(h_A))
        info = CUSPARSE.csrsv_analysis('N', 'S', 'U', M, 'O')
        CUSPARSE.csric0!('N', 'S', M,info, 'O')
        infoRt = CUSPARSE.csrsv_analysis('T', 'T', 'U', M, 'O')
        infoR =  CUSPARSE.csrsv_analysis('N', 'T', 'U', M, 'O')
    end

    CUSPARSE.csrmv!('N',-one(T), A, xₖ, zero(T), rₖ, 'O') # r <- -Axₖ
    axpy!(rₖ, one(T), f) # r <- f - Axₖ
    nrmr0 = norm(rₖ)# ||rₖ|| = ||f - Axₖ||

    rₖᵀzₖ = zero(T)

    k = 0
    for k = 0:max_iters
        # zₖ = inv(M) * rₖ
        if prec == :chol
            CUSPARSE.csrsv_solve!('T', 'U', one(T), M, rₖ, tₖ, infoRt, 'O')
            CUSPARSE.csrsv_solve!('N', 'U', one(T), M, tₖ, zₖ, infoR, 'O')
        elseif prec == :jac
            CUBLAS.dgmm!('L', rₖm, Minv, zₖm)
        else prec == :none
            copy!(zₖ, rₖ)
        end

        if k == 0
            rₖᵀzₖ = rₖ ⋅ zₖ
            copy!(pₖ, zₖ)
        else
            rᵀₖ₋₁zₖ₋₁ = rₖᵀzₖ
            rₖᵀzₖ = rₖ ⋅ zₖ
            β = rₖᵀzₖ / rᵀₖ₋₁zₖ₋₁
            axpy!(zₖ, β, pₖ) # zₖ <- zₖ + β*pₖ
            copy!(pₖ, zₖ) # pₖ <- zₖ + β*pₖ
        end


        CUSPARSE.csrmv!('N', one(T), A, pₖ, zero(T), Apₖ,'O') # Apₖ <- A * pₖ
        pₖᵀApₖ = pₖ ⋅ Apₖ
        α = rₖᵀzₖ / pₖᵀApₖ

        axpy!(xₖ, α, pₖ) # xₖ <- xₖ + α * pₖ
        axpy!(rₖ, -α, Apₖ) # rₖ <- rₖ - α * Apₖ

        res = norm(rₖ)
        if (res < tol)
            println("CG GPU; $prec  $T Converged in $k iterations")
            break
        end
        if k == max_iters
            warn("Did not converge")
            break
        end
        if isnan(res)
            warn("Residual is NaN")
            break
        end
    end

    if prec == :chol
        CUSPARSE.cusparseDestroySolveAnalysisInfo(info)
        CUSPARSE.cusparseDestroySolveAnalysisInfo(infoRt)
        CUSPARSE.cusparseDestroySolveAnalysisInfo(infoR)
    end
    return  k
end


function dot{T}(v1::CUDArt.CudaArray{T,1}, v2::CUDArt.CudaArray{T,1})
    @assert length(v1) == length(v2)
    CUBLAS.dot(length(v1), v1, 1, v2, 1)
end

function norm{T}(v::CUDArt.CudaArray{T,1})
    CUBLAS.nrm2(length(v), v, 1)
end

function scale!{T}(v::CUDArt.CudaArray{T,1}, t::Number)
    CUBLAS.scal!(length(v), t, v, 1)
end

function axpy!{T}(y::CUDArt.CudaArray{T,1}, α::Number, x::CUDArt.CudaArray{T,1})
    @assert length(y) == length(x)
    CUBLAS.axpy!(length(y), α, x, 1, y, 1)
end



