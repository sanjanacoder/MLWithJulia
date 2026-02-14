using DifferentialEquations

# Defining the problem
f(u, p, t) = 1.01 * u
u0 = 1/2
tspan = (0.0, 1.0)

prob = ODEProblem(f, u0, tspan)

# Solving the problem
sol = solve(prob, Tsit5())

sol[5]

# Plot the solution
using Plots
plot(sol, linewidth = 5, title = "Solution to ODE", xaxis = "Time(t)", yaxis = "u(t)", label="My thick line")


plot!(sol.t, t->0.5*exp(1.01t), lw = 3, ls = :dash, label = "True Solution!")