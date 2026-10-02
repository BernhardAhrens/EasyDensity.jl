# Retrain the selected MultiNN with seeds 1:20. Every seed is kept.
# The test fold is predicted only. Spread is computed in physical units.
# multinn_model, fit, and predict_test are defined in cv/common.jl.

include(joinpath(@__DIR__, "common.jl"))

study = load_study()
params = read_hyperparams(study.root, "MultiNN")
folds = assign_folds(study.df, study.k)
pieces = DataFrame[]
members = DataFrame[]

for test_fold in 1:study.k
    # Hold out this test fold. The next fold is validation; the other three train.
    roles = fold_roles(study.df, folds, test_fold, study.k, TARGETS, study.predictors)
    cfg = saved_config(params, test_fold)
    cfg.val_fold == roles.val_fold || error("validation fold mismatch for fold $test_fold")
    seeds = collect(Int, study.seeds)
    @info "MultiNN ensemble" test_fold seeds h = cfg.h
    # Ensemble happens here. Hyperparameters stay fixed; only the seed changes.
    # Every member predicts the test fold. None is discarded.
    slots = Vector{Any}(undef, length(seeds))
    blas = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    try
        Threads.@threads for i in eachindex(seeds)
            model = multinn_model(study.predictors, TARGETS, cfg)
            trained = fit(model, roles, study, cfg; seed = seeds[i])
            trained === nothing && error("seed $(seeds[i]) produced no model")
            slots[i] = predict_test(model, trained, roles.test_df, TARGETS)
        end
    finally
        BLAS.set_num_threads(blas)
    end
    push!(pieces, spread_frame(roles.test_df.row_id, "MultiNN", slots, TARGETS))
    push!(members, member_frame(roles.test_df.row_id, "MultiNN", slots, TARGETS, study.seeds))
end

write_table(ensemble_path(study.root, "MultiNN"), vcat(pieces...))
write_table(members_path(study.root, "MultiNN"), vcat(members...))
