using LinearAlgebra
using ForwardDiff

# Reward via the expression as determinantal rational function
function R_det(π, α, γ, μ, r)
    pπ = [π'[:,s_old]' * α[s_new,s_old,:] for s_new in 1:nS, s_old in 1:nS]
    rπ = diag(r * π')  # Compute the one step reward
    counter = det(I - γ * pπ + μ * transpose(rπ))
    denominator = det(I - γ * pπ)
    return (1-γ) * (counter/denominator - 1)
end

# Reward of a policy
function R(π, α, γ, μ, r)
    pπ = [π[:,s_old]' * α[s_new,s_old,:] for s_new in 1:nS, s_old in 1:nS]
    rπ = diag(r * π)  # Compute the one step reward
    Vπ = (I-γ*transpose(pπ))\((1-γ)*rπ) # Compute the state value function via Bellman's equation
    return μ'*Vπ
end

# Value of a policy
function valueFunction(π, α, γ, r)
    (nS, nA) = size(r)
    pπ = [π[s_old,:]' * α[s_new,s_old,:] for s_new in 1:nS, s_old in 1:nS]
    rπ = diag(r * π')  # Compute the one step reward
    Vπ = (I-γ*transpose(pπ))\((1-γ)*rπ)
    return Vπ
end

# State action frequency for a policy
function stateActionFrequency(π, α, γ, μ, r)
    τ = β * π 
    pπ = [τ[:,s_old]' * α[s_new,s_old,:] for s_new in 1:nS, s_old in 1:nS]
    ρ = (I-γ*pπ) \ ((1-γ)*μ)
    η = Diagonal(ρ) * τ'
    return η
end


### Tabular softmax parametrization

# Define the tabular softmax policy parametrization
function softmaxPolicy(θ)
    θ = reshape(θ, (nA, nO))
    π = exp.(θ)
    for o in 1:nO
        π[:,o] = π[:,o] / sum(π[:,o])
    end
    return π
end

# Reward function for the softmax model
function softmaxReward(θ, α, γ, μ, r)
    π = softmaxPolicy(θ)
    return R(π, α, γ, μ, r)
end

function softmaxStateActionFrequency(θ, α, γ, μ, r)
    π = softmaxPolicy(θ)
    τ = π * β
    pπ = [transpose(τ[:,s_old]) * α[s_new,s_old,:] for s_new in 1:nS, s_old in 1:nS]
    ρ = (I-γ*pπ)\((1-γ)*μ)
    η = Diagonal(ρ) * transpose(π)
    return η
    #return softmaxStateActionFrequency(π, α, γ, μ, r)
end

function softmaxStateFrequency(θ, α, γ, μ, r)
    π = softmaxPolicy(θ)
    τ = π * β
    pπ = [transpose(τ[:,s_old]) * α[s_new,s_old,:] for s_new in 1:nS, s_old in 1:nS]
    ρ = (I-γ*pπ)\((1-γ)*μ)
    return ρ
    #return softmaxStateActionFrequency(π, α, γ, μ, r)
end

### Functions for policy gradient methods

# Implement variants of the natural gradient
saf(θ) = softmaxStateActionFrequency(θ, α, γ, μ, r)
jacobianStateActionFrequencies = θ -> ForwardDiff.jacobian(saf, θ)

# Define the conditioner of the Kakade natural gradient
logLikelihoods(θ) = log.(softmaxPolicy(θ))
jacobianLogLikelihoods = θ -> ForwardDiff.jacobian(logLikelihoods, θ)

jacobianSoftmax = θ -> ForwardDiff.jacobian(softmaxPolicy, θ)

function kakadeConditioner(θ)
    η = reshape(softmaxStateActionFrequency(θ, α, γ, μ, r)', nS*nA)
    J = jacobianLogLikelihoods(θ)
    G = [sum(J[:, i].*J[:, j].*η) for i in 1:nP, j in 1:nP]
    return G
end

function kakadePenalty(θ)
    π = softmaxPolicy(θ)
    τ = π * β
    η = softmaxStateActionFrequency(θ, α, γ, μ, r)
    return -sum(log.(τ).*transpose(η))
end

# Define the conditioner of the Morimura natural gradient
logLikelihoodsSAF(θ) = log.(softmaxStateActionFrequency(θ, α, γ, μ, r))
jacobianLogLikelihoodsSAF = θ -> ForwardDiff.jacobian(logLikelihoodsSAF, θ)

function morimuraConditioner(θ)
    η = reshape(softmaxStateActionFrequency(θ, α, γ, μ, r), nS*nA)
    J = jacobianLogLikelihoodsSAF(θ)
    G = [sum(J[:, i].*J[:, j].*η) for i in 1:nP, j in 1:nP]
    return G
end

function morimuraPenalty(θ)
    η = softmaxStateActionFrequency(θ, α, γ, μ, r)
    return -sum(log.(η).*η)
end

# Define the σ-conditioner and corresponding penalization

function sigmaConditioner(θ, σ=1)
    η = reshape(saf(θ), nS*nA)  
    J = jacobianStateActionFrequencies(θ)
    G = transpose(J) * diagm(η.^-σ) * J
    return G
end

function sigmaPenalty(θ, σ=1)
    η = reshape(saf(θ), nS*nA)  
    if σ == 1
        return -sum(log.(η).*η)
    elseif σ == 0
        return -sum(log.(η))
    else
        return -sum(η.^-σ)
    end
end

logLikelihoodsSAF(θ) = log.(softmaxStateFrequency(θ, α, γ, μ, r)) 
jacobianLogLikelihoodsSF = θ -> ForwardDiff.jacobian(logLikelihoodsSF, θ)


function stateFIM(θ)
    J = jacobianLogLikelihoodsSF(θ)
    ρ = softmaxStateFrequency(θ, α, γ, μ, r)
    G = J' * diagm(ρ) * J
    return G
end


logLikelihoodsSAF(θ) = log.(softmaxStateActionFrequency(θ, α, γ, μ, r))
jacobianLogLikelihoodsSAF = θ -> ForwardDiff.jacobian(logLikelihoodsSAF, θ)

function morimuraConditioner(θ)
    η = reshape(softmaxStateActionFrequency(θ, α, γ, μ, r)', nS*nA)
    J = jacobianLogLikelihoodsSAF(θ)
    G = [sum(J[:, i].*J[:, j].*η) for i in 1:nP, j in 1:nP]
    return G
end

function stateActionFrequency(π, α, γ, μ, r)
    τ = β * π
    pπ = [τ[s_old,:]' * α[s_new,s_old,:] for s_new in 1:nS, s_old in 1:nS]
    ρ = (I-γ*pπ) \ ((1-γ)*μ)
    η = Diagonal(ρ) * τ
    return η
end

# Cardinality of the state, action, observation and parameter space
nS = 2;
nA = 2;
nO = 2;
nP = nO*nA;

# Define a random transition kernel and instantaneous reward function
α = zeros((nS, nS, nA));
α[:,1,:] = Matrix(I, 2, 2);
α[:,2,:] = [0 1; 1 0];
β = [1 0; 0 1];
γ = 0.9;
μ = [0.8, 0.2]
r = [1. 0.; 2. 0.];

#Define the parameter policy gradient
reward(θ) = R(softmaxPolicy(θ), α, γ, μ, r)
∇R = θ -> ForwardDiff.gradient(reward, θ)

# Define a random transition kernel and instantaneous reward function
    
# Compute the optimal reward
rewards_det = zeros(2,2)
for i in 1:2
    for j in 1:2
    π_det = transpose([i-1 2-i; j-1 2-j])
    rewards_det[i,j] = R(π_det, α, γ, μ, r)
    end
end
R_opt = maximum(rewards_det)
R_min = minimum(rewards_det)
#Define the corners of the probability simplex required for the plotting
Bas = [1. 1 1; -1 -1 1; -1 1 -1; 1 -1 -1];

ηDet = zeros(2, 2, 3)
for i in 1:2
    for j in 1:2
    π_det = transpose([i-1 2-i; j-1 2-j])
    η_det = stateActionFrequency(π_det, α, γ, μ, r)
    ηDet[i, j, :] = transpose(Bas) * vec(η_det) #
    end
end

ηDet = reshape(ηDet, (4, 3))

# Begin the plot
p_state_action_polytope = Plots.plot(ηDet[:,1], ηDet[:,2], ηDet[:,3], seriestype=:scatter, markersize = 3, color="black")

Plots.plot!(Bas[[1, 2, 3, 4, 1, 3],1], Bas[[1, 2, 3, 4, 1, 3],2], Bas[[1, 2, 3, 4, 1, 3],3], 
color="black", label=false, width=1.2, linestyle=:dash)

# Computing the state-action frequencies
k=2*10^2

ηAll = zeros(k + 1, k+1, 3)
for i in 1:(k + 1)
    for j in 1:(k + 1)
    π_plot = transpose([(i-1)/k (k-i+1) / k; (j-1)/k (k-j+1)/k])
    η_plot = stateActionFrequency(π_plot, α, γ, μ, r)
    ηAll[i, j, :] = transpose(Bas) * vec(η_plot) #
    end
end
ηAll = reshape(ηAll, ((k+1)^2, 3))

p_state_action_polytope = Plots.plot!(ηAll[:,1], ηAll[:,2], ηAll[:,3],linewidth = 2, label=false, color="black", alpha=0.6)

title_fontsize, tick_fontsize, legend_fontsize, guide_fontsize = 18, 14, 14, 14;

r = [1. 0.; 2. 0.];

nTrajectories = 30;
θ₀ = randn(nTrajectories, nA*nS);
nIterations = 3*10^3;

### Morimura NPG
Δt = 10^-2;
@elapsed begin
    # Allocate the space for the training trajectories
    time_Morimura = zeros(nTrajectories, nIterations);
    rewardTrajectories_Morimura = zeros(nTrajectories, nIterations);
    policyTrajectories_Morimura = zeros(nTrajectories, nIterations, nA);
    ηTrajectories_Morimura = zeros(nIterations, nTrajectories, 3);
    #Optimize using Kakade natural gradient trajectories
    for i in 1:nTrajectories
        θ = θ₀[i,:]
        for k in 1:nIterations
            π = softmaxPolicy(θ)
            policyTrajectories_Morimura[i, k,:] = π[1, :]
            rewardTrajectories_Morimura[i, k] = R(π, α, γ, μ, r)
            η = stateActionFrequency(π, α, γ, μ, r)
            ηTrajectories_Morimura[k, i, :] = transpose(Bas) * vec(η)
            G = morimuraConditioner(θ)
            #G = kakadeConditioner(θ)
            Δθ = pinv(G) * ∇R(θ)
            stepsize = Δt / norm(Δθ)
            θ += stepsize * Δθ
            if k < nIterations
                time_Morimura[i, k+1] =  time_Morimura[i, k] + stepsize
            end
        end
    end
end

title_fontsize, tick_fontsize, legend_fontsize, guide_fontsize = 18, 18, 18, 18;

begin
    p = plot(p_state_action_polytope, ηTrajectories_Morimura[:,:,1], ηTrajectories_Morimura[:,:,2], ηTrajectories_Morimura[:,:,3], width=1.5,
    #title = titles[i], 
    fontfamily="Computer Modern", camera = (30, 0), showaxis=false, ticks=false, legend=false, 
    titlefontsize=title_fontsize, tickfontsize=tick_fontsize, legendfontsize=legend_fontsize, 
    guidefontsize=guide_fontsize, size = (400, 400), ylims=(-1.,1.), zlims=(-1.,1.), margin=-2cm)
    Plots.plot!(p, ηDet[[1, 2, 4, 3, 1],1], ηDet[[1, 2, 4, 3, 1],2], ηDet[[1, 2, 4, 3, 1],3], color="black", width=1.2)
    Plots.plot!(p, Bas[[2, 4],1], Bas[[2, 4],2], Bas[[2, 4],3], color="black", width=1.2, linestyle=:dash)
end


function stateActionFrequency(π, α, γ, μ, r)
    τ = β * π
    pπ = [τ[s_old,:]' * α[s_new,s_old,:] for s_new in 1:nS, s_old in 1:nS]
    ρ = (I-γ*pπ) \ ((1-γ)*μ)
    η = Diagonal(ρ) * τ
    return η
end

begin
    π1 = [1 0; 0 1]
    R1 = R_det(π1, α, γ, μ, r)
    η1 = stateActionFrequency(π1, α, γ, μ, r)
    π2 = [1 0; 1 0]
    R2 = R_det(π2, α, γ, μ, r)
    R(π2, α, γ, μ, r)
    η2 = stateActionFrequency(π2, α, γ, μ, r)
    π3 = [0 1; 0 1]
    R3 = R_det(π3, α, γ, μ, r)
    R(π3, α, γ, μ, r)
    η3 = stateActionFrequency(π3, α, γ, μ, r)
    π4 = [0 1; 1 0]
    R4 = R_det(π4, α, γ, μ, r)
    η4 = stateActionFrequency(π4, α, γ, μ, r)
    V4 = valueFunction(π4, α, γ, r)
    Q4 = [(1-γ)*r[s,a] + γ * α[:,s,a]'*V4 for s in 1:nS, a in 1:nA]
    A4 = [Q4[s,a] - V4[s] for s in 1:nS, a in 1:nA]
end;


δ = min(R4-R1, R4-R2, R4-R3)
Δ = 2 * min((R4-R2) / sum(abs.(η4 - η2)), (R4-R3) / sum(abs.(η4 - η3)))
Δ_M = -(1-γ)^-1 * maximum(A4[.! isapprox.(A4, 0; atol=10^-10)])

time_Morimura[:,1] = minimum(time_Morimura[:,2:end])*ones(nTrajectories)    
gap = R_opt*ones(size(transpose(rewardTrajectories_Morimura[:,:,1])))-transpose(rewardTrajectories_Morimura[:,:,1]);
# State-action plot
p = plot(transpose(time_Morimura[:,:]), gap, linewidth=1.5)  
plot!(p, legend = false, linewidth=1., size=(400,300), fontfamily="Computer Modern", 
titlefontsize=title_fontsize, tickfontsize=tick_fontsize, legendfontsize=legend_fontsize, guidefontsize=guide_fontsize,
framestyle=:box, yaxis=:log,
ylims=(minimum(gap[10^3,:]),1.2*maximum(gap)), xlims=(minimum(time_Morimura[:,1:10^3]), maximum(time_Morimura[:,10^3])))


