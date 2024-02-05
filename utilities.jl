using LinearAlgebra
using ForwardDiff

function valueFunction(π, α, γ, r)
    (nS, nA) = size(r)
    pπ = [π[:,s_old]' * α[s_new,s_old,:] for s_new in 1:nS, s_old in 1:nS]
    rπ = diag(r * π)  # Compute the one step reward
    Vπ = (I-γ*transpose(pπ))\((1-γ)*rπ)
    return Vπ
end

# Reward of a policy
function R(π, α, γ, μ, r)
    pπ = [π[:,s_old]' * α[s_new,s_old,:] for s_new in 1:nS, s_old in 1:nS]
    rπ = diag(r * π)  # Compute the one step reward
    Vπ = (I-γ*transpose(pπ))\((1-γ)*rπ) # Compute the state value function via Bellman's equation
    return μ'*Vπ
end

# State action frequency for a policy
function stateActionFrequency(π, α, γ, μ)
    pπ = [π[:,s_old]' * α[s_new,s_old,:] for s_new in 1:nS, s_old in 1:nS]
    ρ = (I-γ*pπ) \ ((1-γ)*μ)
    η = Diagonal(ρ) * π'
    return η
end

# Define the tabular softmax policy parametrization
function softmaxPolicy(θ)
    θ = reshape(θ, (nA, nS))
    π = exp.(θ)
    for s in 1:nS
        π[:,s] = π[:,s] / sum(π[:,s])
    end
    return π
end

# Reward function for the softmax model
function softmaxReward(θ, α, γ, μ, r)
    π = softmaxPolicy(θ)
    return R(π, α, γ, μ, r)
end

function softmaxStateActionFrequency(θ, α, γ, μ)
    π = softmaxPolicy(θ)
    pπ = [π[:,s_old]' * α[s_new,s_old,:] for s_new in 1:nS, s_old in 1:nS]
    ρ = (I-γ*pπ) \ ((1-γ)*μ)
    η = Diagonal(ρ) * π'
    return η
end

logLikelihoodsSAF(θ) = log.(vec(softmaxStateActionFrequency(θ, α, γ, μ)))
jacobianLogLikelihoodsSAF = θ -> ForwardDiff.jacobian(logLikelihoodsSAF, θ)

function morimuraConditioner(θ)
    η = vec(softmaxStateActionFrequency(θ, α, γ, μ))
    J = jacobianLogLikelihoodsSAF(θ)
    G = [sum(J[:, i].*J[:, j].*η) for i in 1:nP, j in 1:nP]
    return G
end


logLikelihoods(θ) = log.(softmaxPolicy(θ))
jacobianLogLikelihoods = θ -> ForwardDiff.jacobian(logLikelihoods, θ)

function kakadeConditioner(θ)
    η = vec(softmaxStateActionFrequency(θ, α, γ, μ)')
    J = jacobianLogLikelihoods(θ)
    G = [sum(J[:, i].*J[:, j].*η) for i in 1:nP, j in 1:nP]
    return G
end
