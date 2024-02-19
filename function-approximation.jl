using LinearAlgebra
using ForwardDiff
using Plots
using PlotlySave
include("utilities.jl")
include("data-original-example.jl")

pp()

#Define the parameter policy gradient
reward(θ) = R(softmaxPolicy(θ), α, γ, μ, r)
∇R = θ -> ForwardDiff.gradient(reward, θ)

nTrajectories = 30;
θ₀ = randn(nTrajectories, nA*nS);
nIterations = 3*10^3;

### Run state-action NPG
Δt = 10^-2;
@elapsed begin
    # Allocate the space for the training trajectories
    time_Morimura = zeros(nTrajectories, nIterations);
    rewardTrajectories_Morimura = zeros(nTrajectories, nIterations);
    policyTrajectories_Morimura = zeros(nTrajectories, nIterations, nA);
    ηTrajectories_Morimura = zeros(nIterations, nTrajectories, 3);
    #Optimize using the state-action NPG
    for i in 1:nTrajectories
        θ = θ₀[i,:]
        for k in 1:nIterations
            π = softmaxPolicy(θ)
            policyTrajectories_Morimura[i, k,:] = π[1, :]
            rewardTrajectories_Morimura[i, k] = R(π, α, γ, μ, r)
            η = stateActionFrequency(π, α, γ, μ)
            ηTrajectories_Morimura[k, i, :] = transpose(Bas) * vec(η)
            G = morimuraConditioner(θ)
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

# Export state-action distribution plot
begin
    p = plot(p_state_action_polytope, ηTrajectories_Morimura[:,:,1], ηTrajectories_Morimura[:,:,2], ηTrajectories_Morimura[:,:,3], width=1.5,
        camera = (30, 0), showaxis=false, ticks=false, legend=false, 
        titlefontsize=title_fontsize, tickfontsize=tick_fontsize, legendfontsize=legend_fontsize, 
        guidefontsize=guide_fontsize, size = (400, 400), ylims=(-1.,1.), zlims=(-1.,1.)#, margin=-2cm
    )
    p = plot(p, ηDet_proj[[1, 2, 4, 3, 1],1], ηDet_proj[[1, 2, 4, 3, 1],2], ηDet_proj[[1, 2, 4, 3, 1],3], color="black", width=1.2)
    p = plot(p, Bas[[2, 4],1], Bas[[2, 4],2], Bas[[2, 4],3], color="black", width=1.2, linestyle=:dash)
    save("graphics/morimura-state-action.pdf", p)
end

# Compute different exponents
begin
    δ = R_opt - maximum(rewards_det[rewards_det .< maximum(rewards_det)])
    println("δ = ", δ)
    i = findall(==(maximum(rewards_det)), rewards_det)[1]
    scaledOptimalityGap = 2 * [(R_opt-rewards_det[j]) / sum(abs.(ηDet[i,:,:] - ηDet[j,:,:])) for j in CartesianIndex.([(1,1), (2,2)])]
    Δ = minimum(scaledOptimalityGap)
    println("Δ = ", Δ)
    A_opt = advantages[i,:,:]
    Δ_M = -(1-γ)^-1 * maximum(A_opt[.! isapprox.(A_opt, 0; atol=10^-10)])
    println("Δ_M = ", Δ_M)
end

# Export optimality gap plot
begin
    time_Morimura[:,1] = minimum(time_Morimura[:,2:end])*ones(nTrajectories)    
    gap = R_opt*ones(size(transpose(rewardTrajectories_Morimura[:,:,1])))-transpose(rewardTrajectories_Morimura[:,:,1]);
    # State-action plot
    p = plot(transpose(time_Morimura[:,:]), gap, linewidth=1.5);  
    p = plot(p, legend = false, linewidth=1., size=(400,300), fontfamily="Computer Modern", 
        titlefontsize=title_fontsize, tickfontsize=tick_fontsize, legendfontsize=legend_fontsize, guidefontsize=guide_fontsize,
        framestyle=:box, yaxis=:log,
        ylims=(minimum(gap[10^3,:]),1.2*maximum(gap)), xlims=(minimum(time_Morimura), 0.8*maximum(time_Morimura))
    );
    t = range(minimum(time_Morimura), maximum(time_Morimura), 10)
    p = plot(p, t, exp.(- Δ * t), linewidth = 4, color="black", alpha = 0.5, linestyle=:dash)
    p = plot(p, t, exp.(- Δ_M * t), linewidth = 4, color="black", alpha = 0.5, linestyle=:dot)
    save("graphics/morimura-gap.pdf", p)
end
