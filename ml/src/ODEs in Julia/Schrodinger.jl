## Loading packages
using OrdinaryDiffEq, ModelingToolkit, MethodOfLines, DomainSets

# Parameters, variables, and derivatives
@parameters t x
@variables ur(..) ui(..)   # real and imaginary parts of u

# Derivative Operators
##Dt = d/dt  
Dt = Differential(t)
##Dxx - d^/dx^2
Dxx = Differential(x)^2

# Boundary Condition
bcs = [
    ur(0, x) ~ sin(2π * x),
    ui(0, x) ~ 0.0,
    ur(t, 0) ~ 0.0,
    ur(t, 1) ~ 0.0,
    ui(t, 0) ~ 0.0,
    ui(t, 1) ~ 0.0
]

#domains
domains = [t ∈ Interval(0.0, 1.0),
           x ∈ Interval(0.0, 1.0)] 


#equation
# # Split into:
# ∂ψ_re/∂t =  ∂²ψ_im/∂x²
# ∂ψ_im/∂t = -∂²ψ_re/∂x²
eqs = [
    Dt(ur(t, x)) ~  Dxx(ui(t, x)),
    Dt(ui(t, x)) ~ -Dxx(ur(t, x))
]

# PDE system
@named pdesys = PDESystem(eqs, bcs, domains, [t, x], [ur(t, x), ui(t, x)])

#Discretize using Method of Lines (MOL) 
dx = 0.01
discretization = MOLFiniteDifference([x => dx], t)

# Convert the PDE system into an ODE problem
prob = discretize(pdesys, discretization)


# Solve the ODE 
sol = solve(prob, Tsit5(), saveat=0.2)

# Visualization & Animation 
discrete_x = sol[x]
discrete_t = sol[t]

anim = @animate for (i, t_val) in enumerate(discrete_t)
    u_re = sol[ur(t, x)][i, :]
    u_im = sol[ui(t, x)][i, :]

    plot(discrete_x, u_re, 
         label="Real Part", 
         linecolor=:blue, 
         lw=2, 
         ylim=(-1.2, 1.2))
    
    plot!(discrete_x, u_im, 
          label="Imaginary Part", 
          linecolor=:red, 
          lw=2)
    title!("Time: $(round(t_val, digits=3))") # [cite: 31]
    xlabel!("x")
    ylabel!("ψ(t,x)")
end

# Save the resulting animation [cite: 30]
gif(anim, "schrodinger_evolution.gif", fps = 20)

