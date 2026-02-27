#The ModelingToolkit PDE interface for this example looks like this:

##NeuralPDE.jl


using NeuralPDE, Lux, ModelingToolkit, Optimization, OptimizationOptimJL
import ModelingToolkit: Interval

@parameters x y
@variables u(..)
@derivatives Dxx'' ~ x
@derivatives Dyy'' ~ y

# 2D PDE
eq = Dxx(u(x, y)) + Dyy(u(x, y)) ~ -(π .* x) .* (π .* y)

# Boundary conditions
bcs = [u(0, y) ~ 0.0, u(1, y) ~ 0.0,
    u(x, 0) ~ 0.0, u(x, 1) ~ 0.0]
# Space and time domains
domains = [x ∈ Interval(0.0, 1.0),
    y ∈ Interval(0.0, 1.0)]

#Here, we define the neural network, where the input of NN equals the number of dimensions and output equals the number of equations in the system.

# Neural network
dim = 2 # number of dimensions
chain = Lux.Chain(Lux.Dense(dim, 16, Lux.σ), Lux.Dense(16, 16, Lux.σ), Lux.Dense(16, 1))

# Here, we build PhysicsInformedNN algorithm where dx is the step of discretization where strategy stores information for choosing a training strategy.

# Discretization
dx = 0.05
discretization = PhysicsInformedNN(chain, GridTraining(dx))

# As described in the API docs, we now need to define the PDESystem and create PINNs problem using the discretize method.

@named pde_system = PDESystem(eq, bcs, domains, [x, y], [u(x, y)])
prob = discretize(pde_system, discretization)

# Here, we define the callback function and the optimizer. And now we can solve the PDE using PINNs (with the number of epochs maxiters=1000).

#Optimizer
opt = OptimizationOptimJL.BFGS()

callback = function (p, l)
    println("Current loss is: $l")
    return false
end

res = Optimization.solve(prob, opt, callback = callback, maxiters = 1000)
phi = discretization.phi

# We can plot the predicted solution of the PDE and compare it with the analytical solution to plot the relative error.

xs, ys = [infimum(d.domain):(dx / 10):supremum(d.domain) for d in domains]
#analytic_sol_func(x, y) = (sin(pi * x) * sin(pi * y)) / (2pi^2)

u_predict = reshape([first(phi([x, y], res.u)) for x in xs for y in ys],
                    (length(xs), length(ys)))
#u_real = reshape([analytic_sol_func(x, y) for x in xs for y in ys],
                 #(length(xs), length(ys)))
#diff_u = abs.(u_predict .- u_real)

using Plots

#p1 = plot(xs, ys, u_real, linetype = :contourf, title = "analytic");
p2 = plot(xs, ys, u_predict, linetype = :contourf, title = "predict");
#p3 = plot(xs, ys, diff_u, linetype = :contourf, title = "error");
#plot(p1, p2, p3)
plot(p2)