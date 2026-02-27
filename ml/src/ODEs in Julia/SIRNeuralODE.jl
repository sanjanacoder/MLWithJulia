using DifferentialEquations, Lux, DiffEqFlux, Optimization, OptimizationOptimJL, OptimizationOptimisers, Plots, Random, ComponentArrays


# --- Data Generation  ---
function sir_model!(du, u, p, t)
    S, I, R = u
    β, γ = 0.3f0, 0.1f0  # N is implicitly 1.0 now
    du[1] = -β * S * I
    du[2] = β * S * I - γ * I
    du[3] = γ * I
end

# Normalize population: S+I+R = 1.0
N = 1000.0 / 1000.0
I0 = 1.0 / 1000.0
R0 = 0.0 / 1000.0
u0 = Float32[N-I0, I0, R0] 
tspan = (0.0f0, 160.0f0)
tsteps = range(tspan[1], tspan[2], length=40)
prob_true = ODEProblem(sir_model!, u0, tspan)
true_data = Array(solve(prob_true, Tsit5(), saveat=tsteps))

# --- Neural ODE Setup ---
rng = Random.default_rng()
nn = Lux.Chain(
    Lux.Dense(3, 32, tanh), 
    Lux.Dense(32, 3)
)
p_nn, st_nn = Lux.setup(rng, nn)

node_layer = NeuralODE(nn, tspan, Tsit5(), saveat=tsteps)

# --- Loss Function ---
function loss_func(ps, _)
    pred = Array(node_layer(u0, ps, st_nn)[1])
    loss = sum(abs2, true_data .- pred)
    return loss
end

# --- Optimization ---
adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction(loss_func, adtype)
optprob = Optimization.OptimizationProblem(optf, ComponentArray(p_nn))

# Step 1: Adam 
println("Starting Adam...")
res1 = Optimization.solve(optprob, OptimizationOptimisers.Adam(0.005), maxiters=500)

# Step 2: BFGS 
println("Starting BFGS...")
optprob2 = Optimization.OptimizationProblem(optf, res1.u)
res2 = Optimization.solve(optprob2, OptimizationOptimJL.BFGS(), maxiters=200)

println("Final Retcode: ", res2.retcode)

# --- Visualization ---
# Un-normalize for the plot to match your original N=1000 scale
scale = 1000.0
final_pred = Array(node_layer(u0, res2.u, st_nn)[1]) * scale
plot_data = true_data * scale

plt = plot(tsteps, plot_data', lw=3, ls=:dash, 
           label=["S (True)" "I (True)" "R (True)"], 
           color=[:blue :red :green], title="Neural ODE vs Ground Truth")
plot!(plt, tsteps, final_pred', lw=2, alpha=0.7, 
      label=["S (NODE)" "I (NODE)" "R (NODE)"], 
      color=[:blue :red :green])
xlabel!("Days")
ylabel!("Population")
display(plt)
