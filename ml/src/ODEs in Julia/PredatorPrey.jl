using Lux, DiffEqFlux, DifferentialEquations, Optimization, OptimizationOptimisers, OptimizationOptimJL, Random, Plots
using ComponentArrays, Zygote, Statistics

# --- Generate Data ---
const α_val, β_val, δ_val, γ_val = 1.3, 0.9, 0.8, 0.8
u0 = [1.0, 1.0] 
tspan = (0.0, 10.0)
N_points = 50
t = range(tspan[1], tspan[2], length=N_points)

function lotka_volterra!(du, u, p, t)
    x, y = u
    du[1] = α_val*x - β_val*x*y
    du[2] = -δ_val*y + γ_val*x*y
end

prob = ODEProblem(lotka_volterra!, u0, tspan)
sol = solve(prob, Tsit5(), saveat=t)
sol_arr = Array(sol)

true_it1 = [β_val * sol[1, i] * sol[2, i] for i in 1:length(t)]

# --- Construct the UDE ---
rng = Random.default_rng()

NN1 = Lux.Chain(Lux.Dense(2, 32, tanh), Lux.Dense(32, 1))
NN2 = Lux.Chain(Lux.Dense(2, 32, tanh), Lux.Dense(32, 1))

p1, st1 = Lux.setup(rng, NN1)
p2, st2 = Lux.setup(rng, NN2)
p_init = ComponentArray(prey_layer = p1, pred_layer = p2)

function ude_dynamics!(du, u, p, t)
    x, y = u
    # Scale inputs to NN by 0.5 to keep them in the sensitive 'tanh' region
    it1 = abs(NN1([x, y] .* 0.5f0, p.prey_layer, st1)[1][1])
    it2 = abs(NN2([x, y] .* 0.5f0, p.pred_layer, st2)[1][1])
    du[1] = α_val*x - it1
    du[2] = -δ_val*y + it2
end

prob_ude = ODEProblem(ude_dynamics!, u0, tspan)

# --- Training ---
function predict_ude(θ)
    Array(solve(prob_ude, AutoTsit5(Rosenbrock23()), p=θ, saveat=t,
          sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true))))
end

function loss_ude(θ, p)
    x_hat = predict_ude(θ)
    if any(isnan, x_hat) return 1e6 end
    return sum(abs2, sol_arr .- x_hat)
end

adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction(loss_ude, adtype)
optprob = Optimization.OptimizationProblem(optf, p_init)

println("Stage 1: ADAM...")
res1 = Optimization.solve(optprob, OptimizationOptimisers.ADAM(0.005), maxiters = 1000)

println("Stage 2: BFGS...")
optprob2 = Optimization.OptimizationProblem(optf, res1.u)
res2 = Optimization.solve(optprob2, BFGS(), maxiters = 200)

# --- Evaluation ---
final_p = res2.u
final_pred = predict_ude(final_p)
learned_it1 = [abs(NN1(sol_arr[:, i] .* 0.5f0, final_p.prey_layer, st1)[1][1]) for i in 1:length(t)]

println("-"^30)
println("Final Loss Value: ", res2.objective)
println("Population MSE:   ", mean(abs2, sol_arr .- final_pred))
println("-"^30)

# Plots
# Calculate both interaction terms 
learned_it1 = [abs(NN1(sol_arr[:, i] .* 0.5f0, final_p.prey_layer, st1)[1][1]) for i in 1:length(t)]
learned_it2 = [abs(NN2(sol_arr[:, i] .* 0.5f0, final_p.pred_layer, st2)[1][1]) for i in 1:length(t)]
true_it2 = [phys_params.γ * sol[1, i] * sol[2, i] for i in 1:length(t)]

# Plot 1: Prey Population Fit
p1 = plot(t, sol_arr[1,:], label="Prey Data (True)", color=:blue, lw=2)
plot!(p1, t, final_pred[1,:], label="Prey UDE Pred", ls=:dash, color=:red, lw=2)
xaxis!(p1, "Time (Days)")
yaxis!(p1, "Population")
title!(p1, "Prey Convergence")

# Plot 2: Predator Population Fit
p2 = plot(t, sol_arr[2,:], label="Predator Data (True)", color=:purple, lw=2)
plot!(p2, t, final_pred[2,:], label="Predator UDE Pred", ls=:dash, color=:magenta, lw=2)
xaxis!(p2, "Time (Days)")
yaxis!(p2, "Population")
title!(p2, "Predator Convergence")

# Plot 3: Interaction (βxy)
p3 = plot(t, true_it1, label="True βxy", color=:green, lw=2)
plot!(p3, t, learned_it1, label="NN1 ", ls=:dash, color=:orange, lw=2)
xaxis!(p3, "Time (Days)")
yaxis!(p3, "Rate")
title!(p3, "βxy Interaction Term")

# Plot 4: Interaction (γxy)
p4 = plot(t, true_it2, label="True γxy", color=:darkgreen, lw=2)
plot!(p4, t, learned_it2, label="NN2 ", ls=:dash, color=:yellowgreen, lw=2)
xaxis!(p4, "Time (Days)")
yaxis!(p4, "Rate")
title!(p4, " γxy Interaction Term")

# Final combined plot 
plot(p1, p2, p3, p4, layout=(2,2), size=(1000, 800), margin=5Plots.mm)