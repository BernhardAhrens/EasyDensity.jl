using EasyHybrid, Lux, NNlib, Optimisers, CSV, DataFrames, JLD2

include(joinpath(@__DIR__, "dropout_uq.jl"))

const targets = [:BD, :SOCconc, :SOCdensity, :CF]

study = study_grid()
run_study(study.df, study; testid = "02_multiNN_dropout", prefix = "MultiNN_", targets = targets) do cfg
    build_multinn(study.predictors, targets, cfg)
end
