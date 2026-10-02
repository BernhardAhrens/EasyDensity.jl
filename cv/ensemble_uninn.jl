# Retrain each selected UniNN with seeds 1:20. Every seed is kept.
# The test fold is predicted only. Spread is computed in physical units.
# uninn_model, fit, and predict_test are defined in cv/common.jl.

include(joinpath(@__DIR__, "common.jl"))

study = load_study()
params = read_hyperparams(study.root, "UniNN")
folds = assign_folds(study.df, study.k)
per_target = DataFrame[]
members = DataFrame[]

for target in TARGETS
    pieces = DataFrame[]
    for test_fold in 1:study.k
        # Hold out this test fold. The next fold is validation; the other three train.
        # Only rows with this target are used for fitting.
        roles = fold_roles(study.df, folds, test_fold, study.k, [target], study.predictors)
        cfg = saved_config(params, test_fold; target)
        cfg.val_fold == roles.val_fold || error("validation fold mismatch for $target fold $test_fold")
        seeds = collect(Int, study.seeds)
        @info "UniNN ensemble" target test_fold seeds h = cfg.h
        # Ensemble happens here. Hyperparameters stay fixed; only the seed changes.
        # Every member predicts the test fold. None is discarded.
        slots = Vector{Any}(undef, length(seeds))
        blas = BLAS.get_num_threads()
        BLAS.set_num_threads(1)
        try
            Threads.@threads for i in eachindex(seeds)
                model = uninn_model(study.predictors, target, cfg)
                trained = fit(model, roles, study, cfg; seed = seeds[i])
                trained === nothing && error("seed $(seeds[i]) produced no model")
                slots[i] = predict_test(model, trained, roles.test_df, [target])
            end
        finally
            BLAS.set_num_threads(blas)
        end
        push!(pieces, spread_frame(roles.test_df.row_id, "UniNN", slots, [target]))
        push!(members, member_frame(roles.test_df.row_id, "UniNN", slots, [target], study.seeds))
    end
    push!(per_target, vcat(pieces...))
end

write_table(ensemble_path(study.root, "UniNN"), join_on_row(per_target))
write_table(members_path(study.root, "UniNN"), vcat(members...))
