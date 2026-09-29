using EasyHybrid, Lux, NNlib, Optimisers, CSV, DataFrames, JLD2

include(joinpath(@__DIR__, "dropout_uq.jl"))

const scalers = Dict(:SOCconc => 0.151, :CF => 0.263, :BD => 0.529, :SOCdensity => 0.167)

function SOCD_model(; SOCconc, CF, oBD, mBD)
    ϵ = 1.0e-7
    soct = (exp.(SOCconc ./ scalers[:SOCconc]) .- 1) ./ 1000
    soct = clamp.(soct, ϵ, Inf)
    cft = (exp.(CF ./ scalers[:CF]) .- 1) ./ 100
    cft = clamp.(cft, 0, 0.99)
    som = 1.724f0 .* soct
    som = clamp.(som, 0, 1)
    denom = som .* mBD .+ (1.0f0 .- som) .* oBD
    BD = (oBD .* mBD) ./ denom
    BD = clamp.(BD, ϵ, Inf)
    SOCdensity = soct .* 1000 .* BD .* (1 .- cft)
    SOCdensity = clamp.(SOCdensity, 1, Inf)
    SOCdensity = log.(SOCdensity) .* scalers[:SOCdensity]
    BD = BD .* scalers[:BD]
    return (; BD, SOCconc, CF, SOCdensity, oBD, mBD)
end

const parameters = (
    SOCconc = (0.01f0, 0.0f0, 1.0f0),
    CF = (0.15f0, 0.0f0, 1.0f0),
    oBD = (0.20f0, 0.05f0, 0.40f0),
    mBD = (1.20f0, 0.75f0, 2.0f0),
)
const neural_param_names = [:SOCconc, :CF, :mBD, :oBD]
const targets = [:BD, :SOCconc, :SOCdensity, :CF]

study = study_grid()
run_study(study.df, study; testid = "mend_hybridNN_dropout", prefix = "SiNN_",
    targets = targets, latents = [:oBD, :mBD], monitor = [:oBD, :mBD]) do cfg
    constructHybridModel(study.predictors, Symbol[], targets, SOCD_model, parameters,
        neural_param_names, Symbol[];
        hidden_layers = make_dropout_chain(cfg.h, cfg.act, cfg.p),
        activation = cfg.act, scale_nn_outputs = true, input_batchnorm = false,
        start_from_default = true)
end
