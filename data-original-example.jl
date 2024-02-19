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
r = [1. 0.; 2. 0];

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

# compute the optimal state-action distribution and policy 
i = findall(==(maximum(rewards_det)), rewards_det)[1]
η_opt = ηDet[i,:,:]
π_opt = transpose([i[1]-1 2-i[1]; i[2]-1 2-i[2]])

