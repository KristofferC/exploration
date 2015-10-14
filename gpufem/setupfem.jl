function setup_femprob(nx, ny)
    Edof, Ex, Ey, B1,
        B2, B3, B4, coords, dofs = JuAFEM.gen_quad_mesh([0.0, 0.0], [1.0, 1.0], nx, ny, 2);

    cx = 0.1
    cy = 0.05
    bc_dofs = union(B1[:], B2[:], B3[:], B4[:])
    bc = zeros(length(bc_dofs), 2)
    for i in 1:size(bc, 1)
        dof = bc_dofs[i]
        node = div(dof+1, 2)
        bc[i, 1] = dof
        bc[i, 2] = cx * coords[1, node] + cy * coords[2, node,]
    end

    a = start_assemble()
    D = hooke(2, 250e9, 0.3)
    for e in 1:size(Edof, 2)
        ex = Ex[:, Edof[1,e]]
        ey = Ey[:, Edof[1,e]]
        Ke, _ = plani4e(ex, ey, [2, 1, 2], D)
        assemble(Edof[:, e], a, Ke)
    end
    K = end_assemble(a)

    d_pres = convert(Vector{Int}, bc[:,1])
    d_free = setdiff(1:length(coords), d_pres)
    a_pres = bc[:,2]

    Kr = K[d_free, d_free]
    fr = (- K[d_free, d_pres] * a_pres)

    return Kr, fr
end