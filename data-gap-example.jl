using LinearAlgebra
using Plots

# Cardinality of the state, action, observation and parameter space
nS = 2;
nA = 2;
nO = 2;
nP = nO*nA;

# Define a random transition kernel and instantaneous reward function
α = zeros((nS, nS, nA));
α[:,1,:] = Matrix(I, 2, 2);
α[:,2,:] = [0 1; 1 0];
γ = 0.9;
μ = [0.8, 0.2]
r = [1. 3.; 2. 0];

# Compute the optimal reward
rewards_det = zeros(2,2)
ηDet = zeros(2, 2, 2, 2)
ηDet_proj = zeros(2, 2, 3)
advantages = zeros(2, 2, 2, 2)
Bas = [1. 1 1; -1 -1 1; -1 1 -1; 1 -1 -1];
for i in 1:2
    for j in 1:2
    π_det = transpose([i-1 2-i; j-1 2-j])
    print(π_det)
    rewards_det[i,j] = R(π_det, α, γ, μ, r)
    ηDet[i,j,:,:] = stateActionFrequency(π_det, α, γ, μ)
    ηDet_proj[i, j, :] = transpose(Bas) * vec(ηDet[i,j,:,:]) 
    V = valueFunction(π_det, α, γ, r)
    Q = [(1-γ)*r[s,a] + γ * α[:,s,a]'*V for s in 1:nS, a in 1:nA]
    local A = [Q[s,a] - V[s] for s in 1:nS, a in 1:nA]
    advantages[i,j,:,:] = A
    end
end
R_opt = maximum(rewards_det)
R_min = minimum(rewards_det)
ηDet_proj = reshape(ηDet_proj, (4, 3))

# Define a random transition kernel and instantaneous reward function

# Begin the plot
begin
    p_state_action_polytope = Plots.scatter(ηDet_proj[:,1], ηDet_proj[:,2], ηDet_proj[:,3], seriestype=:scatter, markersize=3, c=:black)
    p_state_action_polytope = plot(p_state_action_polytope, Bas[[1, 2, 3, 4, 1, 3],1], Bas[[1, 2, 3, 4, 1, 3],2], Bas[[1, 2, 3, 4, 1, 3],3], 
    color=:black, label=false, width=1.2, linestyle=:dash)
    # Computing the state-action frequencies
    k=2*10^2
    ηAll = zeros(k + 1, k+1, 3)
    for i in 1:(k + 1)
        for j in 1:(k + 1)
        π_plot = transpose([(i-1)/k (k-i+1) / k; (j-1)/k (k-j+1)/k])
        η_plot = stateActionFrequency(π_plot, α, γ, μ)
        ηAll[i, j, :] = transpose(Bas) * vec(η_plot) #
        end
    end
    ηAll = reshape(ηAll, ((k+1)^2, 3))
    p_state_action_polytope = plot(p_state_action_polytope, ηAll[:,1], ηAll[:,2], ηAll[:,3],linewidth = 2, label=false, color="black", alpha=0.6)
end;

