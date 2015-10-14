# Ref http://docs.nvidia.com/cuda/incomplete-lu-cholesky/

using CUSPARSE
using CUDArt
using CUBLAS

import Base: dot, copy!, norm, scale!

function gpucg(h_A, h_f, h_x₀, tol, max_iters)

    Au = CudaSparseMatrixCSR(triu(h_A))
    A = CudaSparseMatrixCSR(h_A)
    f = CUSPARSE.CudaArray(h_f)
    zₖ = CUSPARSE.CudaArray(zeros(h_f))
    rₖ = CUSPARSE.CudaArray(zeros(h_f))
    tₖ = CUSPARSE.CudaArray(zeros(h_f))
    pₖ = CUSPARSE.CudaArray(zeros(h_f))
    Apₖ = CUSPARSE.CudaArray(zeros(h_f))
    xₖ = CUSPARSE.CudaArray(h_x₀)

    # Compute incomplete cholesky
    info = CUSPARSE.csrsv_analysis('N','S','U',Au,'O')
    X = CUSPARSE.csric0('N','S',Au,info,'O')

    # Compute infos
    infoRt = CUSPARSE.csrsv_analysis('T', 'T', 'U', X, 'O')
    infoR =  CUSPARSE.csrsv_analysis('N', 'T', 'U', X, 'O')

    n = length(rₖ)
    rₖᵀzₖ = 1.0

    CUSPARSE.csrmv!('N',-1.0, A, xₖ, 0.0, rₖ, 'O') # r <- -Axₖ
    CUBLAS.axpy!(n, 1.0, f, 1, rₖ, 1) # r <- f - Axₖ
    nrmr0 = norm(rₖ)# ||rₖ|| = ||f - Axₖ||


    for k = 0:max_iters
        # z = inv(M) * r
        CUSPARSE.csrsv_solve!('T', 'U', 1.0, X, rₖ, tₖ, infoRt, 'O')
        CUSPARSE.csrsv_solve!('N', 'U', 1.0, X, tₖ, zₖ, infoR, 'O')
        rᵀₖ₋₁zₖ₋₁ = rₖᵀzₖ
        rₖᵀzₖ = rₖ ⋅ zₖ
        if k == 1
            copy!(pₖ, zₖ)
        else
            β = rₖᵀzₖ / rᵀₖ₋₁zₖ₋₁
            axpy!(zₖ, β, pₖ) # zₖ <- zₖ + β*pₖ
            copy!(pₖ, zₖ) # pₖ <- zₖ + β*pₖ
        end

        CUSPARSE.csrmv!('N', 1.0, A, pₖ, 0.0, Apₖ,'O') # Apₖ <- A * pₖ
        pₖᵀApₖ = pₖ ⋅ Apₖ
        α = rₖᵀzₖ / pₖᵀApₖ

        axpy!(rₖ, -α, Apₖ) # rₖ <- rₖ - α * Apₖ
        axpy!(xₖ, α, pₖ) # xₖ <- xₖ + α * pₖ
        res = norm(rₖ)
        if (res < tol)
            break
        end
        if k == max_iters
            warn("Did not converge")
            break
        end
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

