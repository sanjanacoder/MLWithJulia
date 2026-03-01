using ComponentArrays, Lux, DiffEqFlux, OrdinaryDiffEq, Optimization, OptimizationOptimJL,
    OptimizationOptimisers, Random, Plots, Statistics

rng = Random.default_rng()

# --- Generate Synthetic Data (Traditional SIR) ---
N = 1000.0f0
u0 = Float32[999.0, 1.0, 0.0] 
tspan = (0.0f0, 160.0f0)
tsteps = range(tspan[1], tspan[2], length = 40)

function true_sir_ode(du, u, p, t)
    S, I, R = u
    β, γ, N_pop = 0.3f0, 0.1f0, 1000.0f0
    du[1] = -β * S * I / N_pop
    du[2] = β * S * I / N_pop - γ * I
    du[3] = γ * I
end

prob_true = ODEProblem(true_sir_ode, u0, tspan)
## Generating the ground truth data
ode_data = Array(solve(prob_true, Tsit5(), saveat = tsteps))

# --- Define the Neural ODE ---
dudt2 = Lux.Chain(Lux.Dense(3, 64, tanh), Lux.Dense(64, 3))
p, st = Lux.setup(rng, dudt2)
pinit = ComponentArray(p)

prob_neuralode = NeuralODE(dudt2, tspan, Tsit5(); saveat = tsteps)

function predict_neuralode(p)
    return Array(prob_neuralode(u0, p, st)[1])
end

function loss_neuralode(p, _)
    pred = predict_neuralode(p)
    return sum(abs2, ode_data .- pred)
end

# --- Optimization ---
adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction(loss_neuralode, adtype)
optprob = Optimization.OptimizationProblem(optf, pinit)

println("--- Starting Training ---")
res1 = Optimization.solve(optprob, OptimizationOptimisers.Adam(0.01); maxiters = 1200)
optprob2 = remake(optprob; u0 = res1.u)
res2 = Optimization.solve(optprob2, Optim.BFGS(; initial_stepnorm = 0.01); allow_f_increases = false)

# --- Accuracy Analysis ---
final_pred = predict_neuralode(res2.u)
final_loss = loss_neuralode(res2.u, nothing)

# Mean Absolute Error (MAE) calculation
mae_per_compartment = mean(abs.(ode_data .- final_pred), dims=2)
total_mae = mean(abs.(ode_data .- final_pred))
rel_error = (total_mae / N) * 100

# --- Print Accuracy Metrics with Units ---
println("\n" * "="^45)
println("ACCURACY REPORT")
println("="^45)
println("Final L2 Loss:    ", round(final_loss, digits=2), " individuals²")
println("Global MAE:       ", round(total_mae, digits=4), " individuals")
println("Relative Error:   ", round(rel_error, digits=4), " %")

println("\n--- MAE Per Compartment (individuals) ---")
println("S (Susceptible):  ", round(mae_per_compartment[1], digits=4))
println("I (Infected):     ", round(mae_per_compartment[2], digits=4))
println("R (Recovered):    ", round(mae_per_compartment[3], digits=4))
println("="^45)

# --- Final Comparison Plot ---
plt = plot(tsteps, ode_data', labels=["True S" "True I" "True R"], lw=2)
plot!(plt, tsteps, final_pred', labels=["Pred S" "Pred I" "Pred R"], ls=:dash, lw=2, 
      title="Neural ODE vs Traditional SIR", xlabel="Days", ylabel="Number of Individuals")
display(plt)