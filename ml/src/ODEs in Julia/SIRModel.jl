# System of ODES
# d(S)/dt = beta * S * I / N
# d(I)/dt = (beta * S * I)/N - (gamma * I)
# d(R)/dt = gamma * I
#
# Independent variable : t
# Functions:
#   S(t)
#   R(t)
#   I(t)
# Paramaters/Constant:
#   N = 1000.0
#   beta = 0.3
#   gamma = 0.1
# Initial Conditions:
#   I0 = 1
#   R0 = 0
#.  S0 = N - I0 - R0
#.Tspan : (0, 160.0)


using DifferentialEquations
using Plots

# 1. system of ODEs
function sir_model!(du, u, param, t)
    S, I, R = u
    beta, gamma, N = param
    
    du[1] = -beta * S * I / N           # dS/dt
    du[2] = (beta * S * I / N) - (gamma * I) # dI/dt
    du[3] = gamma * I                  # dR/dt
end

# 2. Parameters and Initial Conditions
N = 1000.0
beta = 0.3
gamma = 0.1

p = [beta, gamma, N]
u0 = [N - 1.0, 1.0, 0.0]  # [S0, I0, R0]
tspan = (0.0, 160.0)

# 3. Define and Solve the Problem
prob = ODEProblem(sir_model!, u0, tspan, p)
sol = solve(prob, Tsit5()) # Using Tsit5 algorithm 

# 4. Plot the results
plot(sol, 
     title="SIR Model Simulation", 
     xlabel="Time (days)", 
     ylabel="Population", 
     label=["Susceptible" "Infected" "Recovered"],
     lw=2)
