using Parameters
using ForwardDiff
using NLsolve
using Voigt
using Voigt.Unicode
using Devectorize

const VOIGT_SIZE = 6

const II = veye(VOIGT_SIZE) ⊗ veye(VOIGT_SIZE)
const Idev = veye(VOIGT_SIZE, VOIGT_SIZE) - 1/3 * II

function JUMAT(ε::Vector{Float64}, parameters_vec::Vector{Float64}, prev_matstat_vec::Vector{Float64},
matstat_vec::Vector{Float64}, t::Float64, ∆t::Float64)

    length(parameters_vec) == 10 || throw(ArgumentError("Invalid parameter length"))
    length(prev_matstat_vec) == 3*VOIGT_SIZE + 3 || throw(ArgumentError("Invalid prev_matstat length"))
    length(matstat_vec) == 3*VOIGT_SIZE + 3 || throw(ArgumentError("Invalid matstat length"))
    length(ε) == VOIGT_SIZE || throw(ArgumentError("Invalid ε length"))
    prev_matstat_vec[end] == 1 || prev_matstat_vec[end] == 0 || throw(ArgumentError("Invalid loading, should be 0 or 1"))

    # "Deserialize" the vectors
    ms = MisesMixedHardViscMS(prev_matstat_vec[1:VOIGT_SIZE],
                              prev_matstat_vec[VOIGT_SIZE+1:2*VOIGT_SIZE],
                              prev_matstat_vec[2*VOIGT_SIZE+1:3*VOIGT_SIZE],
                              prev_matstat_vec[3*VOIGT_SIZE+1],
                              prev_matstat_vec[3*VOIGT_SIZE+2],
                              prev_matstat_vec[3*VOIGT_SIZE+3])

    mp = MisesMixedHardViscMP(E = parameters_vec[1],
                              ν = parameters_vec[2],
                              σy = parameters_vec[3],
                              H = parameters_vec[4],
                              r = parameters_vec[5],
                              κ∞ = parameters_vec[6],
                              α∞ = parameters_vec[7],
                              tstar = parameters_vec[8],
                              σc = parameters_vec[9],
                              n = parameters_vec[10])

    # Actualliy solve
    new_matstat = stress(ε, ∆t, mp, ms)
    # Serialize the matstat
    matstat_vec[1:6] = new_matstat.ₙεₚ
    matstat_vec[7:12] = new_matstat.ₙσ
    matstat_vec[13:18] = new_matstat.ₙα
    matstat_vec[19] = new_matstat.ₙκ
    matstat_vec[20] = new_matstat.ₙμ
    matstat_vec[21] = new_matstat.loading
end


@with_kw immutable MisesMixedHardViscMS
    ₙεₚ::Vector{Float64} = zeros(VOIGT_SIZE)
    ₙσ::Vector{Float64} = zeros(VOIGT_SIZE)
    ₙα::Vector{Float64} = zeros(VOIGT_SIZE)
    ₙκ::Float64 = 0.0
    ₙμ::Float64 = 0.0
    loading::Bool = false
end

@with_kw immutable MisesMixedHardViscMP
    E::Float64
    ν::Float64
    Ee::Matrix{Float64} =   (G = E / 2(1 + ν);
                            K = E / 3(1 - 2ν);
                            Ee = 2 * G * Idev + K * II;
                            if VOIGT_SIZE == 6
                                Ee[4:6, 4:6] /= 2.0
                            end;
                            Ee)
    σy::Float64
    H::Float64
    r::Float64
    κ∞::Float64
    α∞::Float64
    tstar::Float64
    σc::Float64
    n::Float64
end

vm(x) = sqrt(3/2) * vnorm(dev(x))

function stress(ε, ∆t, matpar::MisesMixedHardViscMP, matstat::MisesMixedHardViscMS)
    @unpack matpar: E, Ee, ν, σy, H, r, κ∞, α∞
    @unpack matstat: ₙεₚ, ₙα, ₙκ, ₙμ
    if VOIGT_SIZE == 9
        ε = unsym(ε)
        ε[4:9] /= 2.0
    end

    σₜᵣ = Ee * (ε - ₙεₚ)
    Φ = vm(σₜᵣ - ₙα) - σy - ₙκ

    if Φ < 0
        return MisesMixedHardViscMS(ₙεₚ, σₜᵣ, ₙα, ₙκ, ₙμ, false)
    else
        # Takes the unknown state vector and σdev_para
        R(x) = compute_residual(x, ∆t, σₜᵣ, matpar, matstat)
        dRdx = jacobian(R)

        # Initial guess
        x0 = [σₜᵣ; ₙκ; ₙα; ₙμ]

        res = nlsolve(not_in_place(R, dRdx), x0; iterations = 50, ftol = 1e-3)
        if !NLsolve.converged(res)
            error("No convergence in material routine")
        end
        X = res.zero::Vector{Float64}

        σ = X[1:VOIGT_SIZE]
        κ = X[VOIGT_SIZE + 1]
        α = X[VOIGT_SIZE + 2:2*VOIGT_SIZE + 1]
        μ = X[2*VOIGT_SIZE + 1]

        σ_red_dev = dev(σ - α)
        σ_red_e = vm(σ - α)
        εₚ = ₙεₚ + 3/2 * μ / σ_red_e * σ_red_dev

        return MisesMixedHardViscMS(εₚ, σ, α, κ, μ, true)
    end
end

function compute_residual(x, ∆t, σₜᵣ, matpar, matstat)
    @unpack matpar: E, ν, σy, H, r, κ∞, α∞, σc, tstar, n
    @unpack matstat: ₙεₚ, ₙα, ₙκ, ₙμ

    σ = x[1:VOIGT_SIZE]
    κ = x[VOIGT_SIZE + 1]
    α = x[VOIGT_SIZE + 2:2*VOIGT_SIZE + 1]
    μ = x[2*VOIGT_SIZE + 1]

    G = E / 2(1 + ν)

    σ_red_dev = dev(σ - α)
    σ_red_e = vm(σ - α)
    σ_red_dev_hat = σ_red_dev / σ_red_e

    ν = 3/2 * σ_red_dev / σ_red_e

    @devec R_σ = σ - σₜᵣ + 2 .* G .* μ .* ν
    R_κ = κ - ₙκ - r * H * μ * (1 - κ / κ∞)
    @devec R_α = α - ₙα - (1-r) .* H .* μ .* (2./3 .* ν - α./α∞)
    Φ = max(zero(eltype(σ)), σ_red_e - σy - κ)
    R_Φ = ∆t * η(Φ, σc, n) - tstar * μ
   return [R_σ; R_κ; R_α; R_Φ]
end

η(Φ, σc, n) = (max(zero(Φ), Φ) / σc)^n


function ATS(ε, ∆t, matpar::MisesMixedHardViscMP, matstat::MisesMixedHardViscMS)
    @unpack matpar: E, Ee, ν, σy, H, r, κ∞, α∞
    @unpack matstat: ₙεₚ, ₙσ, ₙα, ₙκ, ₙμ, loading

    X = [ₙσ; ₙκ; ₙα ; ₙμ]

    if !loading
        if VOIGT_SIZE == 9
            Ee = Ee[1:6,1:6]
            Ee[:,4:6] /= 2.0
        end
        return Ee
    else

        R_x(q) = compute_residual(q, ∆t, ₙσ, matpar, matstat)

        dRdX = jacobian(R_x, X)
        dσdε = dRdX[1:VOIGT_SIZE, 1:VOIGT_SIZE] \ Ee
        return dσdε
    end
end
