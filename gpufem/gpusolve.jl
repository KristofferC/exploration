# Ref http://docs.nvidia.com/cuda/incomplete-lu-cholesky/

using CUSPARSE
using CUDArt
using CUBLAS

import Base: dot, copy!, norm, scale!

function gpucg(h_A, h_f, h_x₀, tol, max_iters, prec=:none)
    T = eltype(h_A)

    A = CudaSparseMatrixCSR(h_A)
    f = CUSPARSE.CudaArray(h_f)

    zₖm = CUSPARSE.CudaArray(zeros(eltype(h_f), length(h_f),1))
    zₖ = vec(zₖm)

    rₖm = CUSPARSE.CudaArray(zeros(eltype(h_f), length(h_f),1))
    rₖ = vec(rₖm)

    tₖ = CUSPARSE.CudaArray(zeros(h_f))
    pₖ = CUSPARSE.CudaArray(zeros(h_f))
    Apₖ = CUSPARSE.CudaArray(zeros(h_f))
    xₖ = CUSPARSE.CudaArray(h_x₀)

    # Compute inverse to diagonal of h_A
    if prec == :jac
        Minv = CUSPARSE.CudaArray(1./diag(h_A))
    end

    # Compute incomplete cholesky
    if prec == :chol
        X = CudaSparseMatrixCSR(triu(h_A))
        info = CUSPARSE.csrsv_analysis('N', 'S', 'U', X, 'O')
        CUSPARSE.csric0!('N', 'S', X,info, 'O')
        infoRt = CUSPARSE.csrsv_analysis('T', 'T', 'U', X, 'O')
        infoR =  CUSPARSE.csrsv_analysis('N', 'T', 'U', X, 'O')
    end

    rₖᵀzₖ = one(T)

    CUSPARSE.csrmv!('N',-one(T), A, xₖ, zero(T), rₖ, 'O') # r <- -Axₖ
    CUBLAS.axpy!(n, one(T), f, 1, rₖ, 1) # r <- f - Axₖ
    nrmr0 = norm(rₖ)# ||rₖ|| = ||f - Axₖ||


    for k = 0:max_iters
        # zₖ = inv(M) * rₖ
        if prec == :chol
            CUSPARSE.csrsv_solve!('T', 'U', one(T), X, rₖ, tₖ, infoRt, 'O')
            CUSPARSE.csrsv_solve!('N', 'U', one(T), X, tₖ, zₖ, infoR, 'O')
        elseif prec == :jac
            CUBLAS.dgmm!('L', rₖm, Minv, zₖm)
        elseif prec == :none
            copy!(zₖ, rₖ)
        else
            throw(ArgumentError("Invalid precond method, :chol, :jac, :none"))
        end
        rᵀₖ₋₁zₖ₋₁ = rₖᵀzₖ
        rₖᵀzₖ = rₖ ⋅ zₖ
        if k == 0
            copy!(pₖ, zₖ)
        else
            β = rₖᵀzₖ / rᵀₖ₋₁zₖ₋₁
            axpy!(zₖ, β, pₖ) # zₖ <- zₖ + β*pₖ
            copy!(pₖ, zₖ) # pₖ <- zₖ + β*pₖ
        end

        CUSPARSE.csrmv!('N', one(T), A, pₖ, zero(T), Apₖ,'O') # Apₖ <- A * pₖ
        pₖᵀApₖ = pₖ ⋅ Apₖ
        α = rₖᵀzₖ / pₖᵀApₖ

        axpy!(rₖ, -α, Apₖ) # rₖ <- rₖ - α * Apₖ
        axpy!(xₖ, α, pₖ) # xₖ <- xₖ + α * pₖ
        res = norm(rₖ)

        if (res < tol)
            println("CG; $prec  $T Converged in $k iterations")
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
    return to_host(xₖ)
    # TODO: Destroy infos
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



