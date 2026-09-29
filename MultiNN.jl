using EasyHybrid, Lux, NNlib, Optimisers, CSV, DataFrames, JLD2

include(joinpath(@__DIR__, "dropout_uq.jl"))

const targets = [:BD, :SOCconc, :SOCdensity, :CF]

study = study_grid()
run_study(study.df, study; testid = "02_multiNN_dropout", prefix = "MultiNN_", targets = targets) do cfg
    constructNNModel(study.predictors, targets;
        hidden_layers = make_dropout_chain(cfg.h, cfg.act, cfg.p),
        activation = cfg.act, scale_nn_outputs = true, input_batchnorm = false)
end
