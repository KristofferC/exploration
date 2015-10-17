function cpucg(h_A, h_f, h_x₀, tol, max_iters)

    T = eltype(h_A)

    f = h_f
    A = h_A

    zₖ = zeros(h_f)
    rₖ = zeros(h_f)
    pₖ = zeros(h_f)
    Apₖ = zeros(h_f)
    xₖ = copy(h_x₀)

    n = length(f)


    scale!(rₖ, -one(T))  # r <- -Axₖ
    BLAS.axpy!(n, one(T), f, 1, rₖ, 1) # r <- f - Axₖ
    nrmr0 = norm(rₖ)# ||rₖ|| = ||f - Axₖ||
    rₖᵀzₖ = zero(T)

    k = 0
    for k = 0:max_iters
        copy!(zₖ, rₖ)
        if k == 0
            rₖᵀzₖ = rₖ ⋅ zₖ
            copy!(pₖ, zₖ)
        else
            rᵀₖ₋₁zₖ₋₁ = rₖᵀzₖ
            rₖᵀzₖ = rₖ ⋅ zₖ
            β = rₖᵀzₖ / rᵀₖ₋₁zₖ₋₁
            BLAS.axpy!(n, β, pₖ, 1, zₖ, 1) # zₖ <- zₖ + β*pₖ
            copy!(pₖ, zₖ) # pₖ <- zₖ + β*pₖ
        end

        A_mul_B!(Apₖ, A, pₖ)
        pₖᵀApₖ = pₖ ⋅ Apₖ
        α = rₖᵀzₖ / pₖᵀApₖ

        BLAS.axpy!(n, α, pₖ, 1, xₖ, 1) # xₖ <- xₖ + α * pₖ
        BLAS.axpy!(n, -α, Apₖ, 1, rₖ, 1) # xₖ <- xₖ + α * pₖ

        res = norm(rₖ)
        if (res < tol)
            println("CG CPU; $T Converged in $k iterations")
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
    return  k
end
