println("LOADED")

function JLMAT(ε::Vector{Float64}, σ::Vector{Float64})
    E = 200e9
    ν = 0.3
    G = E / (2(1 + ν))
    f = E / ((1+ν) * (1-2ν))
    λ = f * ν
    M = f * (1 - ν)
    println(ε)
    D = Float64[ M      λ      λ    0.0    0.0   0.0;
                 λ      M      λ    0.0    0.0   0.0;
                 λ      λ      M    0.0    0.0   0.0;
                0.0    0.0    0.0    G     0.0   0.0;
                0.0    0.0    0.0   0.0     G    0.0;
                0.0    0.0    0.0   0.0    0.0    G]
    σ[:] = D * ε
end



