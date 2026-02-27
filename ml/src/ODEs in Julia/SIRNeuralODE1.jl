#=
# --- 1. Generate Synthetic Ground Truth (Traditional SIR) ---
function sir_model!(du, u, p, t)
    S, I, R = u
    β, γ, N = 0.3, 0.1, 1000.0
    du[1] = -β * S * I / N
    du[2] = β * S * I / N - γ * I
    du[3] = γ * I
end

u0 = Float32[999.0, 1.0, 0.0]
tspan = (0.0f0, 40.0f0)
tsteps = range(tspan[1], tspan[2], length=40)

prob_true = ODEProblem(sir_model!, u0, tspan)
sol_true = solve(prob_true, Tsit5(), saveat=tsteps)
true_data = Array(sol_true)

# 5. Visualize the synthetic data
plot(sol_true, 
     xlabel="Time (days)", 
     ylabel="Population", 
     label=["Susceptible" "Infected" "Recovered"],
     title="SIR Model Synthetic Data",
     lw=2)

# --- 2. Define the Neural ODE ---
rng = Random.default_rng()
# Neural Network to approximate the derivative: input 3 (S,I,R) -> output 3 (dS,dI,dR)
nn = Lux.Chain(
    Lux.Dense(3, 16, tanh),
    Lux.Dense(16, 16, tanh),
    Lux.Dense(16, 3)
)
p_nn, st_nn = Lux.setup(rng, nn)

# Wrap NN in a NeuralODE layer
node = NeuralODE(nn, tspan, Tsit5(), saveat=tsteps)

# --- 3. Training / Optimization ---
function loss_func(ps, _)
    pred = Array(node(u0, ps, st_nn)[1])
    loss = sum(abs2, true_data .- pred)
    return loss
end

# --- 4. Optimization ---
adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction(loss_func, adtype)
optprob = Optimization.OptimizationProblem(optf, ComponentArray(p_nn))

# Step 1: Adam
res1 = Optimization.solve(optprob, OptimizationOptimisers.Adam(0.01), maxiters=200)

# Step 2: BFGS (Fine-tuning)
optprob2 = Optimization.OptimizationProblem(optf, res1.u)
res2 = Optimization.solve(optprob2, OptimizationOptimJL.BFGS(), maxiters=100)


# --- 4. Comparison and Visualization ---
final_pred = Array(node(u0, res2.u, st_nn)[1])

p1 = plot(tsteps, true_data', lw=3, label=["S (True)" "I (True)" "R (True)"], 
          ls=:dash, color=[:blue :red :green], title="Neural ODE vs. Traditional SIR")
plot!(p1, tsteps, final_pred', lw=2, label=["S (NODE)" "I (NODE)" "R (NODE)"], 
      color=[:blue :red :green], alpha=0.7)
xlabel!("Time (Days)")
ylabel!("Population")
=#