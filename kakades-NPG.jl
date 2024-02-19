using LinearAlgebra
using ForwardDiff
using Plots
using PlotlySave
include("utilities.jl")
# Examples: original and gap
# ex = "gap-example"
ex = "original-example"
include(string("data-", ex, ".jl"))

#Define the parameter policy gradient
reward(θ) = R(softmaxPolicy(θ), α, γ, μ, r)
∇R = θ -> ForwardDiff.gradient(reward, θ)

# nTrajectories = 30;
# θ₀ = randn(nTrajectories, nA*nS);
nIterations = 10^4;

### Kakade's NPG
Δt = 10^-2;
@elapsed begin
    # Allocate the space for the training trajectories
    times_kakade = zeros(nTrajectories, nIterations);
    rewardTrajectories_kakade = zeros(nTrajectories, nIterations);
    KLs_kakade = zeros(nTrajectories, nIterations);
    cKLs_kakade = zeros(nTrajectories, nIterations);
    #Optimize using Kakade natural gradient trajectories
    for i in 1:nTrajectories
        θ = θ₀[i,:]
        for k in 1:nIterations
            π = softmaxPolicy(θ)
            policies_kakade[i, k,:] = π[1, :]
            rewardTrajectories_kakade[i, k] = R(π, α, γ, μ, r)
            cKLs_kakade[i, k] = cKL(π_opt, π, α, γ, μ)
            η = stateActionFrequency(π, α, γ, μ)
            KLs_kakade[i, k] = KL(η_opt, η)
            G = kakadeConditioner(θ)
            Δθ = pinv(G) * ∇R(θ)
            stepsize = Δt / norm(Δθ)
            θ += stepsize * Δθ
            if k < nIterations
                times_kakade[i, k+1] =  times_kakade[i, k] + stepsize
            end
        end
    end
end

title_fontsize, tick_fontsize, legend_fontsize, guide_fontsize = 18, 18, 18, 18;

times_kakade[:,1] = minimum(times_kakade[:,2:end])*ones(nTrajectories)    

begin
    δ = R_opt - maximum(rewards_det[rewards_det .< maximum(rewards_det)])
    println("δ = ", δ)
    i = findall(==(maximum(rewards_det)), rewards_det)[1]
    scaledOptimalityGap = 2 * [(R_opt-rewards_det[j]) / sum(abs.(ηDet[i,:,:] - ηDet[j,:,:])) for j in CartesianIndex.([(1,1), (2,2)])]
    Δ = minimum(scaledOptimalityGap)
    println("Δ = ", Δ)
    # Δ = 2 * min((R_opt-rewards_det[j]) / sum(abs.(ηDet[i,:,:] - ηDet[j,:,:])), (R_opt-rewards_det[2,2]) / sum(abs.(ηDet[i,:,:] - ηDet[2,2,:,:])))
    A_opt = advantages[i,:,:]
    Δ_K = -(1-γ)^-1 * maximum(A_opt[.! isapprox.(A_opt, 0; atol=10^-10)])
    println("Δ_K = ", Δ_K)
end

# Plot the KL-divergence to the optimal state-action distribution 
begin
    p = plot(transpose(times_kakade[:,:]), KLs_kakade', linewidth=1.5);  
    t = range(minimum(times_kakade), maximum(times_kakade), 10)
    # p = plot(p, t, .4 *exp.(- Δ_K * t), linewidth = 4, color="black", alpha = 0.5, linestyle=:dot)
    p = plot(p, t, .4 *exp.(- Δ * t), linewidth = 4, color="black", alpha = 0.5, linestyle=:dash)
    p = plot(p, legend=false, linewidth=1., size=(400,300), fontfamily="Computer Modern", framestyle=:box, 
        titlefontsize=title_fontsize, tickfontsize=tick_fontsize, legendfontsize=legend_fontsize, guidefontsize=guide_fontsize,
        yaxis=:log, ylims=(minimum(KLTrajectories_Morimura[:,10^3]),1.2*maximum(KLTrajectories_Morimura)), 
        xlims=(minimum(times_morimura), 0.8*maximum(times_morimura))
    )
    save(string("graphics/kakade-KL-", ex, ".pdf"), p)
end

# Plot the conditional KL-divergence to the optimal policy
begin
    p = plot(transpose(times_kakade[:,:]), cKLs_kakade', linewidth=1.5);  
    t = range(minimum(times_kakade), maximum(times_kakade), 10)
    # p = plot(p, t, .4 *exp.(- Δ_K * t), linewidth = 4, color="black", alpha = 0.5, linestyle=:dot)
    p = plot(p, t, .4 *exp.(- Δ * t), linewidth = 4, color="black", alpha = 0.5, linestyle=:dash)
    p = plot(p, legend=false, linewidth=1., size=(400,300), fontfamily="Computer Modern", framestyle=:box, 
        titlefontsize=title_fontsize, tickfontsize=tick_fontsize, legendfontsize=legend_fontsize, guidefontsize=guide_fontsize,
        yaxis=:log, ylims=(minimum(KLTrajectories_Morimura[:,10^3]),1.2*maximum(KLTrajectories_Morimura)), 
        xlims=(minimum(times_morimura), 0.8*maximum(times_morimura))
    )
    save(string("graphics/kakade-cKL-", ex, ".pdf"), p)
end

plot(p, legend=false, linewidth=1., size=(400,300), fontfamily="Computer Modern", framestyle=:box, 
            titlefontsize=title_fontsize, tickfontsize=tick_fontsize, legendfontsize=legend_fontsize, guidefontsize=guide_fontsize,
            yaxis=:log, ylims=(minimum(KLTrajectories_Morimura[:,10^3]),1.2*maximum(KLTrajectories_Morimura)), 
            xlims=(minimum(times_morimura), 0.8*maximum(times_morimura))
        )

begin
    gap = R_opt*ones(size(transpose(rewardTrajectories_kakade[:,:,1])))-transpose(rewardTrajectories_kakade[:,:,1]);
    # State-action plot
    p = plot(transpose(times_kakade[:,:]), gap, linewidth=1.5);  
    t = range(minimum(times_kakade), maximum(times_kakade), 10)
    p = plot(p, t, 1 *exp.(- Δ_K * t), linewidth = 4, color="black", alpha = 0.5, linestyle=:dot)
    #p = plot(p, t, .3 *exp.(- Δ * t), linewidth = 4, color="black", alpha = 0.5, linestyle=:dash)
    p = plot(p, legend = false, linewidth=1., size=(400,300), fontfamily="Computer Modern", 
        titlefontsize=title_fontsize, tickfontsize=tick_fontsize, legendfontsize=legend_fontsize, guidefontsize=guide_fontsize,
        framestyle=:box, yaxis=:log,
        ylims=(minimum(KLTrajectories_Morimura[:,10^3]),1.2*maximum(KLTrajectories_Morimura)), 
        xlims=(minimum(times_morimura), 0.5*maximum(times_morimura))
    );
    #save("graphics/kakade-gap.pdf", p)
    save(string("graphics/kakade-gap-", ex, ".pdf"), p)
end
