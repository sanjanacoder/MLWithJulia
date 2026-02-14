# System of ODES
# d(theta)/dt = omega(t)
# d(omega)/dt = (-3g/2l)sin(theta) + (3/ml^2)*M
#
# Independent variable : t
# Functions:
#   theta(t)
# Paramaters/Constant:
#   l = 1.0
#   m = 1.0
#   g = 9.81
#   M(t) = ?
# Initial Conditions:
#   theta0 = 0.01
#.  omega0 = 0.0
#.Tspan : (0, 10.0)


using DifferentialEquations
using Plots

# 1.system of ODEs
function pendulum_system!(du, u, p, t)
    θ, ω = u             # functions
    l, m, g, M = p       # parameters
    
    # d(theta)/dt = omega
    du[1] = ω
    
    # d(omega)/dt = (-3g/2l)sin(theta) + (3/ml^2)*M
    du[2] = (-3*g / (2*l)) * sin(θ) + (3 / (m * l^2)) * M
end

# 2. Constants and Parameters
l = 1.0
m = 1.0
g = 9.81
M = 0.0  # External torque (can be a function of t if needed)
params = (l, m, g, M)

# 3. Initial Conditions and Timespan
u0 = [0.01, 0.0]      # [theta0, omega0]
tspan = (0.0, 10.0)

# 4. Define and Solve the Problem
prob = ODEProblem(pendulum_system!, u0, tspan, params)
sol = solve(prob, Tsit5(), reltol=1e-8, abstol=1e-8)

# 5. Visualize the Results
plot(sol, 
     title="Pendulum Motion", 
     xaxis="Time (t)", 
     ylabel="Angular Values", 
     label=["Theta" "Omega"],
     lw=2)
