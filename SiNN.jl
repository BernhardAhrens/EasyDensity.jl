using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()
package_path = joinpath(abspath(joinpath(@__DIR__, "..")), "EasyHybrid_Porosity") # EasyDensity.jl and EasyHybrid_Porosity must live in the same parent folder
Pkg.develop(path=package_path)
using EasyHybrid
using Lux
using Optimisers
using Random
using LuxCore
using CSV, DataFrames
using EasyHybrid.MLUtils
using Statistics
using Plots
using JLD2
using Base.Threads: nthreads
using ProgressMeter
@info "Number of threads: $(nthreads())"

# 03 - flexiable BD, both oBD and mBD will be learnt by NN

testid = "mend_hybridNN";
version = "v20251209"
results_dir = joinpath(@__DIR__, "eval");
model_dir = joinpath(@__DIR__, "model");
output_tmp_dir = joinpath(@__DIR__, "output_tmp");
plots_dir = joinpath(results_dir, "plots");
mkpath(results_dir); mkpath(model_dir); mkpath(output_tmp_dir); mkpath(plots_dir);
target_names = [:BD, :SOCconc, :CF, :SOCdensity];

# input
df = CSV.read(joinpath(@__DIR__, "data/lucas_preprocessed_v20251125.csv"), DataFrame; normalizenames=true)
println(size(df))

# scales
scalers = Dict(
    :SOCconc   => 0.151, # g/kg, log(x+1)*0.151
    :CF        => 0.263, # percent, log(x+1)*0.263
    :BD        => 0.529, # g/cm3, x*0.529
    :SOCdensity => 0.167, # kg/m3, log(x)*0.167
);

# mechanistic model
function SOCD_model(; SOCconc, CF, oBD, mBD)
    ϵ = 1e-7

    # invert transforms
    soct = (exp.(SOCconc ./ scalers[:SOCconc]) .- 1) ./ 1000
    soct = clamp.(soct, ϵ, Inf)
    
    cft = (exp.(CF ./ scalers[:CF]) .- 1) ./ 100
    cft = clamp.(cft, 0, 0.99)

    # compute BD safely
    som = 1.724f0 .* soct
    som = clamp.(som, 0, 1) # test!!!!!!!!
    
    denom = som .* mBD .+ (1f0 .- som) .* oBD
    # denom = clamp.(denom, ϵ, Inf)

    BD = (oBD .* mBD) ./ denom
    BD = clamp.(BD, ϵ, Inf)

    # SOCdensity
    SOCdensity = soct .* 1000 .* BD .* (1 .- cft)
    SOCdensity = clamp.(SOCdensity, 1, Inf)

    # scale
    SOCdensity = log.(SOCdensity) .* scalers[:SOCdensity]
    BD = BD .* scalers[:BD]

    return (; BD, SOCconc, CF, SOCdensity, oBD, mBD)
end


# param bounds
parameters = (
    SOCconc = (0.01f0, 0.0f0, 1.0f0),   # fraction
    CF      = (0.15f0, 0.0f0, 1.0f0),   # fraction,
    oBD     = (0.20f0, 0.05f0, 0.40f0),  # also NN learnt, g/cm3
    mBD     = (1.20f0, 0.75f0, 2.0f0),  # NN leanrt
)

# define param for hybrid model
neural_param_names = [:SOCconc, :CF, :mBD, :oBD]
# global_param_names = [:oBD]
forcing = Symbol[]
targets = [:BD, :SOCconc, :SOCdensity, :CF]       # SOCconc is both a param and a target

# predictor
predictors = Symbol.(names(df))[18:end-6]; # CHECK EVERY TIME 
nf = length(predictors)

# hyperparameters
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

Random.seed!(42);
k = 5;
folds = make_folds(df, k = k, shuffle = true);
rlt_list_param = Vector{DataFrame}(undef, k)
rlt_list_pred = Vector{DataFrame}(undef, k)  

@info "Threads available: $(Threads.nthreads())"

# Base hybrid model. `tune` reconstructs a model from this base plus the
# per-config overrides (hidden_layers, activation) in the loop below, so the
# architecture passed here only provides the defaults.
hm_base = constructHybridModel(
    predictors, forcing, targets, SOCD_model,
    parameters, neural_param_names, [];
    hidden_layers = collect(configs[1].h), activation = configs[1].act,
    scale_nn_outputs = true, input_batchnorm = false, start_from_default = true
)

# Warm-up: force compilation of tune/train once on the master before the
# threaded sweep, so all worker threads hit already-compiled code paths.
cfg = configs[1]
    train(
        hm_base, df;
        hidden_layers = collect(cfg.h), activation = cfg.act,
        nepochs = 2, batchsize = cfg.bs, opt = AdamW(cfg.lr),
        training_loss = :nseLoss, loss_types = [:nse, :r2], shuffleobs = true,
        model_name = "warmup", output_folder = output_tmp_dir,
        random_seed = 42, patience = 15, yscale = identity,
        monitor_names = [:oBD, :mBD],return_model = :best,
        show_progress = false, plotting = false, agg = mean
    )


# Hyperparameter search and 5-fold cross-validation training

    # disable logging in loop, repetitive output is annoying
using Logging
Logging.disable_logging(Logging.Warn)
@showprogress for test_fold in 1:k
    @info "Training outer fold $test_fold of $k"

    train_folds = setdiff(1:k, test_fold)
    train_idx = findall(in(train_folds), folds)
    train_df = df[train_idx, :]
    test_idx  = findall(==(test_fold), folds)
    test_df = df[test_idx, :]

    # track best config for this outer fold
    lk = ReentrantLock()
    best_val_loss = Inf
    best_config = nothing
    best_result = nothing
    best_model_path = nothing
    best_model = nothing

# This is like a normal for loop, but it uses threads.
# The :greedy option means that as soon as one task finishes its work, it takes the next value from the iterator. https://docs.julialang.org/en/v1/base/multi-threading/
# The @showprogress macro is used to show a progress bar.
# The Threads.@threads macro is used to run the loop in parallel.

Threads.@threads :greedy for i in 1:length(configs)
        try
            cfg = configs[i]
        
            h  = cfg.h
            bs = cfg.bs
            lr = cfg.lr
            act = cfg.act
            println("Testing h=$h, bs=$bs, lr=$lr, activation=$act")

            # `tune` reconstructs the model from hm_base with these per-config
            # overrides and then trains it, replacing the manual
            # constructHybridModel + train pair.
            rlt = tune(
                hm_base, train_df;
                hidden_layers = collect(h),
                activation = act,
                nepochs = 1000,
                batchsize = bs,
                opt = AdamW(lr),
                training_loss = :nseLoss,
                loss_types = [:r2, :mse],
                shuffleobs = true,
                model_name = "$(testid)_config$(i)_fold$(test_fold)",
                output_folder = output_tmp_dir,
                patience = 100,
                yscale = identity,
                monitor_names = [:oBD, :mBD],
                return_model = :best,
                show_progress = false,
                plotting = false,
                agg = mean
            )
    
            lock(lk)
            if rlt.best_loss < best_val_loss
                best_val_loss = rlt.best_loss
                best_config = cfg
                best_result = rlt
                best_model_path = "trained_model_$(testid)_config$(i)_fold$(test_fold).jld2"
            end
            unlock(lk)
        catch err
            @error "Thread $i crashed" exception = err
            @error sprint(showerror, err)
        end

    end

    # register best hyper paramets
    agg_name = Symbol("mean")
    r2s  = map(vh -> getproperty(vh, agg_name), best_result.val_history.r2)
    mses = map(vh -> getproperty(vh, agg_name), best_result.val_history.mse)
    best_epoch = max(best_result.best_epoch, 1)

    local_results_param = DataFrame(
        h = string(best_config.h),
        bs = best_config.bs,
        lr = best_config.lr,
        act = string(best_config.act),
        r2 = r2s[best_epoch],
        mse = mses[best_epoch],
        best_epoch = best_epoch,
        test_fold = test_fold,
        path = best_model_path,
    )
    rlt_list_param[test_fold] = local_results_param

    # move best models and then remove tmp files
    cp(joinpath(output_tmp_dir, best_model_path), joinpath(model_dir, best_model_path * ".jld2"); force=true)
    for f in readdir(output_tmp_dir; join=true)
        rm(f; force=true, recursive=true)
    end

    # rebuild the winning architecture (tune builds the model internally and
    # only returns TrainResults) to run test-set predictions with its ps/st.
    best_model = constructHybridModel(
        predictors, forcing, targets, SOCD_model,
        parameters, neural_param_names, [];
        hidden_layers = collect(best_config.h), activation = best_config.act,
        scale_nn_outputs = true, input_batchnorm = false, start_from_default = true
    )

    ps, st = best_result.ps, best_result.st
    (x_test,  y_test)  = prepare_data(best_model, test_df)
    ŷ_test, st_test = best_model(x_test, ps, LuxCore.testmode(st))
    # println(propertynames(ŷ_test))
    # println(propertynames(ŷ_test.parameters))

    for var in [:BD, :SOCconc, :CF, :SOCdensity, :oBD, :mBD]
        if hasproperty(ŷ_test, var)
            val = getproperty(ŷ_test, var)

            if val isa AbstractVector && length(val) == nrow(test_df)
                test_df[!, Symbol("pred_", var)] = val # per row

            elseif (val isa Number) || (val isa AbstractVector && length(val) == 1)
                test_df[!, Symbol("pred_", var)] = fill(Float32(val isa AbstractVector ? first(val) : val), nrow(test_df))
            end


        end
    end
    
    rlt_list_pred[test_fold] = test_df

end


# enable logging again
Logging.disable_logging(Logging.BelowMinLevel)


rlt_param = vcat(rlt_list_param...)
rlt_pred = vcat(rlt_list_pred...)

CSV.write(joinpath(results_dir, "$(testid)_cv.pred_$version.csv"), rlt_pred)
CSV.write(joinpath(results_dir, "$(testid)_hyperparams_$version.csv"), rlt_param)

# Plausibility check: negative SOCdensity predictions

bad_rows = rlt_pred[rlt_pred.pred_SOCdensity .< 0, [:ocd,:pred_SOCdensity]]

# Plausibility check: negative oBD predictions

bad_rows = rlt_pred[rlt_pred.pred_oBD .< 0, [:ocd,:pred_SOCdensity]]

# Plausibility check: negative mBD predictions

bad_rows = rlt_pred[rlt_pred.pred_mBD .< 0, [:ocd,:pred_SOCdensity]]

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

    savefig(plt, joinpath(plots_dir, "scatter_$(tgt)_$(version).png"))
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

    plt = histogram(
        valid_vals;
        bins = 50,
        xlabel = col,
        ylabel = "Frequency",
        title = "Histogram of $col",
        lw = 1,
        legend = false
    )
    savefig(plt, joinpath(plots_dir, "hist_$(col)_$(version).png"))
end
