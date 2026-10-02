# One network per target. Test fold held out. Next fold validates. The rest train.
# uninn_model and fit are defined in cv/common.jl. This script does not train the ensemble.

include(joinpath(@__DIR__, "common.jl"))

study = load_study()
folds = assign_folds(study.df, study.k)
rows = NamedTuple[]

for target in TARGETS
    for test_fold in 1:study.k
        # Hold out this test fold. The next fold is validation; the other three train.
        # Only rows with this target are used for fitting.
        roles = fold_roles(study.df, folds, test_fold, study.k, [target], study.predictors)
        @info "UniNN tuning" target test_fold val_fold = roles.val_fold train_folds = roles.train_folds rows = nrow(roles.fit_df) configs = length(study.configs)

        # Tuning happens here: every width, batch size, learning rate, and activation
        # is trained on the three training folds. The validation fold picks the winner.
        losses = fill(Inf, length(study.configs))
        epochs = fill(0, length(study.configs))
        blas = BLAS.get_num_threads()
        BLAS.set_num_threads(1)
        try
            Threads.@threads for i in eachindex(study.configs)
                cfg = study.configs[i]
                try
                    model = uninn_model(study.predictors, target, cfg)
                    trained = fit(model, roles, study, cfg; seed = TUNING_SEED)
                    trained === nothing && continue
                    losses[i] = Float64(trained.best_loss)
                    epochs[i] = Int(trained.best_epoch)
                catch err
                    @warn "configuration failed" target h = cfg.h bs = cfg.bs lr = cfg.lr act = act_name(cfg.act) exception = (err, catch_backtrace())
                end
            end
        finally
            BLAS.set_num_threads(blas)
        end
        i = argmin(losses)
        isfinite(losses[i]) || error("every configuration failed for $target, test fold $test_fold")
        cfg = study.configs[i]
        best = (; h = cfg.h, bs = cfg.bs, lr = cfg.lr, act = cfg.act, val_mse = losses[i], best_epoch = epochs[i])
        @info "UniNN selected" target test_fold h = best.h bs = best.bs lr = best.lr act = act_name(best.act) val_mse = best.val_mse
        push!(rows, remember(target, test_fold, roles.val_fold, best))
    end
end

write_table(hyperparams_path(study.root, "UniNN"), rows)
