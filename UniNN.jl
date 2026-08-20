using Pkg
# Pkg.activate(".")
using AxisKeys
using Revise
using EasyHybrid
using Lux
using Optimisers
using Random
using LuxCore
using CSV, DataFrames
using EasyHybrid.MLUtils
using Statistics
using Plots
using Flux
using NNlib 
using JLD2

testid = "mend_uniNN"
version = "v20251209";
results_dir = joinpath(@__DIR__, "eval");
target_names = [:BD, :SOCconc, :CF, :SOCdensity];

# input
df = CSV.read(joinpath(@__DIR__, "data/lucas_preprocessed_v20251125.csv"), DataFrame; normalizenames=true)

# scales
scalers = Dict(
    :SOCconc   => 0.151, # g/kg, log(x+1)*0.151
    :CF        => 0.263, # percent, log(x+1)*0.263
    :BD        => 0.529, # g/cm3, x*0.529
    :SOCdensity => 0.167, # kg/m3, log(x)*0.167
);

# predictor
predictors = Symbol.(names(df))[18:end-6]; # CHECK EVERY TIME 
nf = length(predictors)

# search space
hidden_configs = [ 
    (512, 256, 128, 64, 32, 16),
    (512, 256, 128, 64, 32), 
    (256, 128, 64, 32, 16),
    (256, 128, 64, 32),
    (256, 128, 64),
    (128, 64, 32, 16),
    (128, 64, 32),
    (64, 32, 16)
];
batch_sizes = [128, 256, 512];
lrs = [1e-3, 5e-4, 1e-4];
activations = [relu, swish, gelu];

configs = [(h=h, bs=bs, lr=lr, act=act)
    for h in hidden_configs
    for bs in batch_sizes
    for lr in lrs
    for act in activations]

println(length(configs))
# cross-validation
k = 5;
# folds_df = CSV.read("folds_assignment.csv", DataFrame)
# folds = folds_df.fold
# folds = collect(folds)
Random.seed!(42);
folds = make_folds(df, k = k, shuffle = true);
# d = load("$(testid)_3folds_results.jld")
# a_list_param = d["rlt_list_param"]
# a_list_pred  = d["rlt_list_pred"]
rlt_list_param = Vector{DataFrame}(undef, k)
rlt_list_pred = Vector{DataFrame}(undef, k)  

@info "Threads: $(Threads.nthreads())"

# Per-target hyperparameter search and 5-fold cross-validation training

@time for test_fold in 1:k 
    @info "Fold $test_fold"

    train_folds = setdiff(1:k, test_fold)
    train_idx = findall(in(train_folds), folds)
    train_df = df[train_idx, :]
    test_idx  = findall(==(test_fold), folds)
    test_df_full = df[test_idx, :]

    fold_params = DataFrame()
    preds_this_fold = Dict{Symbol, DataFrame}()

    for tgt in target_names
        @info "Target $tgt"

        # dropmissing for train one by one
        train_df_t = dropmissing(train_df, tgt)
        if nrow(train_df_t) == 0
            @warn "No training rows for $tgt — filling NaN"
            test_df_full[!, Symbol("pred_", tgt)] = fill(NaN32, nrow(test_df_full))
            continue
        end

        lk = ReentrantLock()
        best_loss   = Inf
        best_cfg    = nothing
        best_rlt    = nothing      
        best_model  = nothing    
        best_model_path = nothing

        ########################
        # hyperparam search
        ########################
        Threads.@threads for i in 1:length(configs)
            cfg = configs[i]

            h  = cfg.h
            bs = cfg.bs
            lr = cfg.lr
            act = cfg.act
            println("Testing h=$h, bs=$bs, lr=$lr, activation=$act")

            nn_local = constructNNModel(
                predictors, [tgt];
                hidden_layers = collect(h),
                activation = act,
                scale_nn_outputs = true,
                input_batchnorm = false
            )

            rlt = train(
                nn_local, train_df_t, ();
                nepochs = 200,
                batchsize = bs,
                opt = AdamW(lr),
                training_loss = :mse,
                loss_types = [:mse, :r2],
                shuffleobs = true,
                file_name = "$(testid)_$(tgt)_config$(i)_fold$(test_fold).jld2",
                patience = 15,
                return_model = :best,  
                plotting = false,
                show_progress = false,
                hybrid_name = "$(testid)_$(tgt)_config$(i)_fold$(test_fold)"
            )

            lock(lk) do
                if rlt.best_loss < best_loss
                    best_loss = rlt.best_loss
                    best_cfg  = (h=h, bs=bs, lr=lr, act=act)
                    best_rlt  = rlt              
                    best_model = nn_local     
                    best_model_path = "best_model_$(testid)_$(tgt)_config$(i)_fold$(test_fold)"
                end
            end
        end

        # if all hyperparameter fails
        if best_rlt === nothing
            @warn "All configs failed for target=$tgt on fold=$test_fold — recording NaN"

            push!(fold_params, (
                fold       = test_fold,
                target     = String(tgt),
                h          = "nothing",
                bs         = NaN,
                lr         = NaN,
                act        = "nothing",
                r2         = NaN,
                mse        = NaN,
                best_epoch = NaN,
                best_model_path = "none"
            ))

            test_df_full[!, Symbol("pred_", tgt)] = fill(NaN32, nrow(test_df_full))
            continue
        end

        ########################
        # get the best hyper combi
        ########################
        agg = :sum
        r2s  = map(vh -> getproperty(vh, agg), best_rlt.val_history.r2)
        mses = map(vh -> getproperty(vh, agg), best_rlt.val_history.mse)
        be = max(best_rlt.best_epoch, 1)

        push!(fold_params, (
            fold       = test_fold,
            target     = String(tgt),
            h          = string(best_cfg.h),
            bs         = best_cfg.bs,
            lr         = best_cfg.lr,
            act        = string(best_cfg.act),
            r2         = r2s[be],
            mse        = mses[be],
            best_epoch = be,
            best_model_path = best_model_path
        ))

        ########################
        # predict
        ########################
        ps, st = best_rlt.ps, best_rlt.st     
        # println("full test", size(test_df_full))

        x_test = to_keyedArray(Float32.(test_df_full[!, predictors]))
        # test_df_t = filter(tgt => x -> !isnan(x), test_df_full)
        # # println("test_df_t", size(test_df_t))
        # x_test, _ = prepare_data(best_model, test_df_t)
        # println("prepared data", size(x_test))
        
        ŷ_test, _ = best_model(x_test, ps, LuxCore.testmode(st))
        pred_df = DataFrame(
            row_id = test_df_full.row_id,
            pred = ŷ_test[tgt]
        )
        preds_this_fold[tgt] = pred_df

        ############
        ## clean tmp files....tooo much
        ##########
        cp(joinpath("output_tmp", best_model_path * ".jld2"), joinpath("model", best_model_path * ".jld2"); force=true)
        for f in filter(x -> isfile(x), readdir("output_tmp"; join=true))
            rm(f; force=true)
        end

    end

    for tgt in target_names
        test_df_full = leftjoin(test_df_full, preds_this_fold[tgt], on=:row_id)
        rename!(test_df_full, :pred => Symbol("pred_", tgt))
    end
    
    rlt_list_pred[test_fold] = test_df_full
    rlt_list_param[test_fold] = fold_params
end

rlt_param = vcat(rlt_list_param...)
rlt_pred  = vcat(rlt_list_pred...)


# folds_df = DataFrame(fold = folds)
# CSV.write("folds_assignment.csv", folds_df)
# jldsave("$(testid)_3folds_results.jld";
#     rlt_list_param = rlt_list_param,
#     rlt_list_pred  = rlt_list_pred
# )

CSV.write(joinpath(results_dir, "$(testid)_cv.pred_$version.csv"), rlt_pred)
CSV.write(joinpath(results_dir, "$(testid)_hyperparams_$version.csv"), rlt_param)

# Scatter plots: predicted vs true values

for tgt in ["BD", "SOCconc", "CF", "SOCdensity"]

    true_vals = rlt_pred[:, Symbol(tgt)]
    pred_vals = rlt_pred[:, Symbol("pred_", tgt)]

    # 过滤掉 invalid 值（避免 NaN 出图报错）
    mask = map(!isnan, true_vals) .& map(!isnan, pred_vals)
    x = true_vals[mask]
    y = pred_vals[mask]

    println("Plotting $tgt: valid points = ", length(x))

    plt = histogram2d(
        x, y;
        nbins = (30, 30),
        cbar = true,
        xlab = tgt,
        ylab = "pred_$tgt",
        color = cgrad(:bamako, rev=true),
        normalize = false,
        size = (460, 400),
    )

    display(plt)
end

# Histograms of predicted variables

for col in ["pred_BD", "pred_SOCconc", "pred_CF", "pred_SOCdensity"]

    vals = rlt_pred[:, col]

    # 有效值（非 missing 且非 NaN）
    valid_vals = filter(x -> !ismissing(x) && !isnan(x), vals)

    n_valid = length(valid_vals)
    vmin = minimum(valid_vals)
    vmax = maximum(valid_vals)

    println("Variable: $col")
    println("  Valid count = $n_valid")
    println("  Min = $vmin")
    println("  Max = $vmax\n")

    histogram(
        vals;
        bins = 50,
        xlabel = col,
        ylabel = "Frequency",
        title = "Histogram of $col",
        lw = 1,
        legend = false
    )
    display(current())
end

# Result dimensions

println(size(rlt_pred))
