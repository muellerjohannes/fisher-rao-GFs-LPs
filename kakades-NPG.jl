using LinearAlgebra
using ForwardDiff
using Plots
using PlotlySave
include("utilities.jl")
include("data-MDP.jl")


#Define the parameter policy gradient
reward(θ) = R(softmaxPolicy(θ), α, γ, μ, r)
∇R = θ -> ForwardDiff.gradient(reward, θ)

nTrajectories = 30;
θ₀ = randn(nTrajectories, nA*nS);
nIterations = 10^4;

### Kakade's NPG
Δt = 10^-2;
@elapsed begin
    # Allocate the space for the training trajectories
    time_kakade = zeros(nTrajectories, nIterations);
    rewardTrajectories_kakade = zeros(nTrajectories, nIterations);
    policyTrajectories_kakade = zeros(nTrajectories, nIterations, nA);
    ηTrajectories_kakade = zeros(nIterations, nTrajectories, 3);
    #Optimize using Kakade natural gradient trajectories
    for i in 1:nTrajectories
        θ = θ₀[i,:]
        for k in 1:nIterations
            π = softmaxPolicy(θ)
            policyTrajectories_kakade[i, k,:] = π[1, :]
            rewardTrajectories_kakade[i, k] = R(π, α, γ, μ, r)
            η = stateActionFrequency(π, α, γ, μ)
            ηTrajectories_kakade[k, i, :] = transpose(Bas) * vec(η)
            G = kakadeConditioner(θ)
            Δθ = pinv(G) * ∇R(θ)
            stepsize = Δt / norm(Δθ)
            θ += stepsize * Δθ
            if k < nIterations
                time_kakade[i, k+1] =  time_kakade[i, k] + stepsize
            end
        end
    end
end

title_fontsize, tick_fontsize, legend_fontsize, guide_fontsize = 18, 18, 18, 18;

begin
    p = plot(p_state_action_polytope, ηTrajectories_kakade[:,:,1], ηTrajectories_kakade[:,:,2], ηTrajectories_kakade[:,:,3], width=1.5,
        camera = (30, 0), showaxis=false, ticks=false, legend=false, 
        titlefontsize=title_fontsize, tickfontsize=tick_fontsize, legendfontsize=legend_fontsize, 
        guidefontsize=guide_fontsize, size = (400, 400), ylims=(-1.,1.), zlims=(-1.,1.)#, margin=-2cm
    )
    p = plot(p, ηDet_proj[[1, 2, 4, 3, 1],1], ηDet_proj[[1, 2, 4, 3, 1],2], ηDet_proj[[1, 2, 4, 3, 1],3], color="black", width=1.2)
    p = plot(p, Bas[[2, 4],1], Bas[[2, 4],2], Bas[[2, 4],3], color="black", width=1.2, linestyle=:dash)
    save("graphics/kakade-state-action.pdf", p)
end

begin
    δ = R_opt - maximum(rewards_det[rewards_det .< maximum(rewards_det)])
    println("δ = ", δ)
    i = findall(==(maximum(rewards_det)), rewards_det)[1]
    scaledOptimalityGap = 2 * [(R_opt-rewards_det[j]) / sum(abs.(ηDet[i,:,:] - ηDet[j,:,:])) for j in CartesianIndex.([(1,1), (2,2)])]
    Δ = minimum(scaledOptimalityGap)
    println("Δ = ", Δ)
    # Δ = 2 * min((R_opt-rewards_det[j]) / sum(abs.(ηDet[i,:,:] - ηDet[j,:,:])), (R_opt-rewards_det[2,2]) / sum(abs.(ηDet[i,:,:] - ηDet[2,2,:,:])))
    A_opt = advantages[i,:,:]
    Δ_M = -(1-γ)^-1 * maximum(A_opt[.! isapprox.(A_opt, 0; atol=10^-10)])
    println("Δ_M = ", Δ_M)
end

begin
    time_kakade[:,1] = minimum(time_kakade[:,2:end])*ones(nTrajectories)    
    gap = R_opt*ones(size(transpose(rewardTrajectories_kakade[:,:,1])))-transpose(rewardTrajectories_kakade[:,:,1]);
    # State-action plot
    p = plot(transpose(time_kakade[:,:]), gap, linewidth=1.5);  
    p = plot(p, legend = false, linewidth=1., size=(400,300), fontfamily="Computer Modern", 
        titlefontsize=title_fontsize, tickfontsize=tick_fontsize, legendfontsize=legend_fontsize, guidefontsize=guide_fontsize,
        framestyle=:box, yaxis=:log,
        ylims=(minimum(gap[10^3,:]),1.2*maximum(gap)), xlims=(minimum(time_kakade), 0.8*maximum(time_kakade))
    );
    t = range(minimum(time_kakade), maximum(time_kakade), 10)
    p = plot(p, t, exp.(- Δ * t), linewidth = 4, color="black", alpha = 0.5, linestyle=:dash)
    save("graphics/kakade-gap.pdf", p)
end