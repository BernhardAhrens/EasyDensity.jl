using EasyHybrid, Lux, NNlib, Optimisers, CSV, DataFrames, JLD2

include(joinpath(@__DIR__, "dropout_uq.jl"))

const targets = [:BD, :SOCconc, :SOCdensity, :CF]

study = study_grid()
run_study(study.df, study; testid = "mend_hybridNN_dropout", prefix = "SiNN_",
    targets = targets, latents = SINN_LATENTS, monitor = SINN_LATENTS) do cfg
    build_sinn(study.predictors, targets, cfg)
end
