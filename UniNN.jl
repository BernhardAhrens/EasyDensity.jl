using EasyHybrid, Lux, NNlib, Optimisers, CSV, DataFrames, JLD2

include(joinpath(@__DIR__, "dropout_uq.jl"))

const targets = [:BD, :SOCconc, :CF, :SOCdensity]

study = study_grid()
keys = filter(k -> hasproperty(study.df, k), ROW_KEYS)
global dest = select(study.df, keys)
for (i, tgt) in enumerate(targets)
    global dest = run_study(dropmissing(study.df, tgt), study;
        testid = "mend_uniNN_dropout", prefix = "UniNN_", targets = [tgt],
        into = dest, write = i == length(targets), target = tgt,
        append_params = i > 1) do cfg
        constructNNModel(study.predictors, [tgt];
            hidden_layers = make_dropout_chain(cfg.h, cfg.act, cfg.p),
            activation = cfg.act, scale_nn_outputs = true, input_batchnorm = false)
    end
end
