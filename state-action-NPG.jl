using LinearAlgebra
using ForwardDiff
using Plots
using PlotlySave
include("utilities.jl")
# Examples: original and gap
#ex = "original-example"
ex = "gap-example"
include(string("data-", ex, ".jl"))

#Define the parameter policy gradient
reward(θ) = R(softmaxPolicy(θ), α, γ, μ, r)
∇R = θ -> ForwardDiff.gradient(reward, θ)

#nTrajectories = 30;
#θ₀ = randn(nTrajectories, nA*nS);
nIterations = 3*10^3;

### Run state-action NPG
Δt = 10^-2;
@elapsed begin
    # Allocate the space for the training trajectories
    times_morimura = zeros(nTrajectories, nIterations);
    rewards_morimura = zeros(nTrajectories, nIterations);
    KLs_morimura = zeros(nTrajectories, nIterations);
    cKLs_morimura = zeros(nTrajectories, nIterations);
    #Optimize using the state-action NPG
    for i in 1:nTrajectories
        θ = θ₀[i,:]
        for k in 1:nIterations
            π = softmaxPolicy(θ)
            rewards_morimura[i, k] = R(π, α, γ, μ, r)
            cKLs_morimura[i, k] = cKL(π_opt, π, α, γ, μ)
            η = stateActionFrequency(π, α, γ, μ)
            KLs_morimura[i, k] = KL(η_opt, η)
            G = morimuraConditioner(θ)
            Δθ = pinv(G) * ∇R(θ)
            stepsize = Δt / norm(Δθ)
            θ += stepsize * Δθ
            if k < nIterations
                times_morimura[i, k+1] =  times_morimura[i, k] + stepsize
            end
        end
    end
end

title_fontsize, tick_fontsize, legend_fontsize, guide_fontsize = 18, 18, 18, 18;

times_morimura[:,1] = minimum(times_morimura[:,2:end])*ones(nTrajectories)    

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

# Plot the KL-divergence to the optimal state-action distribution 
begin
    p = plot(transpose(times_morimura[:,:]), KLs_morimura', linewidth=1.5);  
    t = range(minimum(times_morimura), maximum(times_morimura), 10)
    p = plot(p, t, exp.(- Δ * t), linewidth = 4, color="black", alpha = 0.5, linestyle=:dash, label="\$e^{-Δt}\$")
    p = plot(p, t, 10*exp.(- Δ_K * t), linewidth = 4, color="black", alpha = 0.5, linestyle=:dot)
    p = plot(p, legend=false, linewidth=1., size=(400,300), fontfamily="Computer Modern", 
        titlefontsize=title_fontsize, tickfontsize=tick_fontsize, legendfontsize=legend_fontsize, guidefontsize=guide_fontsize,
        framestyle=:box, yaxis=:log, ylims=(minimum(KLTrajectories_Morimura[:,10^3]),1.2*maximum(KLTrajectories_Morimura)), 
        xlims=(minimum(times_morimura), 0.5*maximum(times_morimura))
    )
    save(string("graphics/morimura-KL-", ex, ".pdf"), p)
end

# Plot the conditional KL-divergence to the optimal policy
begin
    p = plot(transpose(times_morimura[:,:]), cKLs_morimura', linewidth=1.5);  
    t = range(minimum(times_morimura), maximum(times_morimura), 10)
    p = plot(p, t, exp.(- Δ * t), linewidth = 4, color="black", alpha = 0.5, linestyle=:dash, label="\$e^{-Δt}\$")
    #p = plot(p, t, 10*exp.(- Δ_K * t), linewidth = 4, color="black", alpha = 0.5, linestyle=:dot)
    p = plot(p, legend=false, linewidth=1., size=(400,300), fontfamily="Computer Modern", 
        titlefontsize=title_fontsize, tickfontsize=tick_fontsize, legendfontsize=legend_fontsize, guidefontsize=guide_fontsize,
        framestyle=:box, yaxis=:log, ylims=(minimum(KLTrajectories_Morimura[:,10^3]),1.2*maximum(KLTrajectories_Morimura)), 
        xlims=(minimum(times_morimura), 0.8*maximum(times_morimura))
    )
    save(string("graphics/morimura-cKL-", ex, ".pdf"), p)
end

# Plot the sub-optimality gap 
begin
    gap = R_opt*ones(size(transpose(rewards_morimura[:,:,1])))-transpose(rewards_morimura[:,:,1]);
    p = plot(transpose(times_morimura[:,:]), gap, linewidth=1.5);  
    t = range(minimum(times_morimura), maximum(times_morimura), 10)
    p = plot(p, t, exp.(- Δ * t), linewidth = 4, color="black", alpha = 0.5, linestyle=:dash, label="\$e^{-Δt}\$")
    p = plot(p, t, 10*exp.(- Δ_K * t), linewidth = 4, color="black", alpha = 0.5, linestyle=:dot)
    p = plot(p, legend=false, linewidth=1., size=(400,300), fontfamily="Computer Modern", 
        titlefontsize=title_fontsize, tickfontsize=tick_fontsize, legendfontsize=legend_fontsize, guidefontsize=guide_fontsize,
        framestyle=:box, yaxis=:log,
        ylims=(minimum(KLTrajectories_Morimura[:,10^3]),1.2*maximum(KLTrajectories_Morimura)), 
        xlims=(minimum(times_morimura), 0.5*maximum(times_morimura))
    );
    save(string("graphics/morimura-gap-", ex, ".pdf"), p)
end
