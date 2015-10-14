# Ref http://docs.nvidia.com/cuda/incomplete-lu-cholesky/

using CUSPARSE
using CUDArt
using CUBLAS

function gpucg(K, f, x₀, tol)

    Ku = triu(K)
    d_Ku = CudaSparseMatrixCSR(Ku)
    d_K = CudaSparseMatrixCSR(K)
    d_f = CUSPARSE.CudaArray(f)
    d_z = CUSPARSE.CudaArray(similar(f))
    d_x = CUSPARSE.CudaArray(x₀)
    d_r = CUSPARSE.CudaArray(zeros(f))
    d_t = CUSPARSE.CudaArray(zeros(f))
    d_z = CUSPARSE.CudaArray(zeros(f))
    d_p = CUSPARSE.CudaArray(zeros(f))
    d_q = CUSPARSE.CudaArray(zeros(f))

    # Compute incomplete cholesky
    info = CUSPARSE.csrsv_analysis('N','S','U',d_Ku,'O')
    d_X = CUSPARSE.csric0('N','S',d_Ku,info,'O')

    # Compute infos
    infoRt = CUSPARSE.csrsv_analysis('T','T','U',d_X,'O')
    infoR = CUSPARSE.csrsv_analysis('N','T','U',d_X,'O')

    n = length(d_r)
    ρ = 1.0

    CUSPARSE.csrmv!('N',1.0,d_K,d_x,0.0,d_r,'O') # r <-  -K x₀
    CUBLAS.scal!(n, -1.0, d_r, 1) # f <- -f
    CUBLAS.axpy!(n, 1.0, d_f, 1, d_r, 1) # r <- f + r = f - K x₀
    nrmr0 = CUBLAS.nrm2(n, d_r, 1) # ||r||

    # init guess = x0
    # r <_ A x0
    # for i = ...
    #      Mz <- r, sparse lower and upper
    #      pi <_ rTz
    #      i == 1
    #         z <-
    #     β <- ρ * ρ
    #      p <- z + β
    #    q <- A p # sp mat vec
    #    α -> ρ p T
    #     x <- x + α
    #    r <- r - α
    #
    for i = 1:10^4
        CUSPARSE.csrsv_solve!('T', 'U', 1.0, d_X, d_r, d_t,infoRt,'O')
        CUSPARSE.csrsv_solve!('N', 'U', 1.0, d_X, d_t, d_z,infoR,'O')
        ρp = ρ
        ρ = CUBLAS.dot(n, d_r, 1, d_z, 1)
        if i == 1
            CUBLAS.blascopy!(n, d_z, 1, d_p, 1)
        else
            β = ρ / ρp
            CUBLAS.axpy!(n, β, d_p, 1, d_z, 1)
            CUBLAS.blascopy!(n, d_z, 1, d_p, 1)
        end

        CUSPARSE.csrmv!('N',1.0,d_K,d_p,0.0,d_q,'O') # q = K * p
        temp = CUBLAS.dot(n, d_p, 1, d_q, 1)
        α = ρ / temp
        CUBLAS.axpy!(n, α, d_p, 1, d_x, 1);
        CUBLAS.axpy!(n, -α, d_q, 1, d_r, 1);
        nrmr = CUBLAS.nrm2(n, d_r, 1)
        #println("Norm: $nrmr")
        if (nrmr < tol)
            break
        end
    end
    return to_host(d_x)
    # TODO: Destroy infos
end

