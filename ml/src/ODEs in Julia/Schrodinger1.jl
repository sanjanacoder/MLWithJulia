using ModelingToolkit, MethodOfLines, OrdinaryDiffEq, DomainSets, Plots

# 1. Define the Variables and Parameters
@parameters t x
@variables ψ(..)
Dt = Differential(t)
Dxx = Differential(x)^2

# 2. Define the PDE, Initial Conditions (IC), and Boundary Conditions (BC)
# Note: In Julia, 'im' represents the imaginary unit 'i'
eq  = im * Dt(ψ(t, x)) ~ Dxx(ψ(t, x)) # V(x) is 0 [cite: 12, 13]

bcs = [ψ(0, x) ~ sin(2π * x),       # Initial Condition [cite: 15]
       ψ(t, 0) ~ 0.0,               # Boundary Condition at x=0 [cite: 17]
       ψ(t, 1) ~ 0.0]               # Boundary Condition at x=1 [cite: 18]

# 3. Define the Domain
domains = [t ∈ Interval(0.0, 1.0),
           x ∈ Interval(0.0, 1.0)]  # Spatial domain [0, 1] [cite: 13]

# 4. Create the PDESystem
@named pdesys = PDESystem(eq, bcs, domains, [t, x], [ψ(t, x)])

# 5. Discretize using Method of Lines (MOL) 
dx = 0.01 # Spatial step size
order = 2 # Finitedifference order
discretization = MOLFiniteDifference([x => dx], t)

# Convert the PDE system into an ODE problem
prob = discretize(pdesys, discretization)

# 6. Solve the ODE 
sol = solve(prob, TRBDF2(), saveat=0.01)

# 7. Visualization & Animation 
x_vals = sol[x]
t_vals = sol[t]

anim = @animate for i in 1:length(t_vals)
    # Extract the complex wave function values at time index i
    curr_psi = sol[ψ(t, x)][i, :]
    
    plot(x_vals, real.(curr_psi), label="Real Part", lw=2, ylim=(-1.5, 1.5))
    plot!(x_vals, imag.(curr_psi), label="Imaginary Part", lw=2)
    title!("Wave Function Evolution at t = $(round(t_vals[i], digits=3))")
    xlabel!("Position (x)")
    ylabel!("ψ(t,x)")
end

# Save the resulting animation [cite: 30]
gif(anim, "schrodinger_evolution.gif", fps = 20)