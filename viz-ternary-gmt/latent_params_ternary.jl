using Parquet2, Tables, CSV
using Statistics
using DataFrames
using GMT: ternary, colorbar!, makecpt, rich, superscript # for the ternary plot

include(joinpath(@__DIR__, "../viz-makie/uq_columns.jl"))

#  "CET-L19"
cmap_list = ["#abdda4", "#ffffbf", "#fdae61", "#d7191c"]

version = "v20251216"
# targets = ["SOCconc", "CF", "BD", "SOCdensity"]
# labels = ["SOC content", "CF", "BD", "SOC density"]
models_raw = ["UniNN", "MultiNN", "SiNN"]

function compute_r2_mse(y_pred, y_target)
    _mse = mean((y_target .- y_pred).^2)
    denom = sum((y_target .- mean(y_target)).^2)
    _r2 = 1 - sum((y_target .- y_pred).^2)/denom
    _bias = mean(y_pred .- y_target)
    return _r2, _mse, _bias
end

ds = Parquet2.Dataset(joinpath(@__DIR__, "../eval/all_cv.pred_with.lc_$(version).pq"))
df = attach_uq(DataFrame(ds; copycols=true), joinpath(@__DIR__, ".."))

# ? why, from where are these coming from?
scalers = Dict(
    "SOCconc"=> 0.151,
    "CF" => 0.263,
    "BD"=> 0.529,
    "SOCdensity"=> 0.167)

function draw_ternary(df, method)
add_scaled_predictions!(df, method, models_raw, scalers)
vars_to_check = ["$(mod)_ocd_$(method)" for mod in models_raw]
obd = "pred_oBD_$(method)"
mbd = "pred_mBD_$(method)"

df_filtered = copy(df)

# ---- 1. COUNT IDS BY NUMBER OF TIME POINTS ----
id_counts = combine(groupby(df_filtered, :id), nrow => :count)

num_ids_all3  = count(==(3), id_counts.count)
num_ids_not3  = count(!=(3), id_counts.count)

println("IDs with all 3 time points: ", num_ids_all3)
println("IDs WITHOUT all 3 time points: ", num_ids_not3)

# ---- 2. KEEP ONLY IDS THAT HAVE ALL 3 ROWS ----
valid_ids = id_counts.id[id_counts.count .== 3]

df3 = filter(row -> row.id in valid_ids, df_filtered)

println("Filtered dataframe shape: ", size(df3))

# ---- 3. TEMPORAL STABILITY USING IQR AND RANGE ----
# Compute min and max per id for each variable
stability = combine(
    groupby(df3, :id),
    [
        var => minimum => Symbol(var * "_min") for var in vars_to_check
    ]...,
    [
        var => maximum => Symbol(var * "_max") for var in vars_to_check
    ]...
)

# Compute range (max - min)
for var in vars_to_check
    stability[!, Symbol(var * "_range")] =
        stability[!, Symbol(var * "_max")] .-
        stability[!, Symbol(var * "_min")]
end

val_labels = ["UniNN","MultiNN","SiNN"]

# ---- 1. FILTER: silt > 0 (missing-safe) ----
df_filter = filter(row -> !ismissing(row.silt) && row.silt > 0, df)

# ---- 2. GROUPBY clay / silt / sand, mean of numeric columns ----
gcols = [:clay, :silt, :sand]

gdf = DataFrames.groupby(df_filter, gcols)

select_names = ["clay", "silt", "sand", obd, mbd]

df_filter_mean = combine(
    gdf,
    select_names .=> (x -> mean(skipmissing(x)))   # compute mean per group
)

# ---- 3. POINTS (Nx3 matrix, like .values in NumPy) ----
points_css = Matrix(df_filter_mean[:, [:clay, :silt, :sand]])

# ---- 4. COLORS ----
colors_oBD = df_filter_mean[!, "$(obd)_function"]
colors_mBD = df_filter_mean[!, "$(mbd)_function"]

# ---- 5. QUANTILE LIMITS ----
vmin_o = quantile(skipmissing(colors_oBD), 0.05)
vmax_o = quantile(skipmissing(colors_oBD), 0.95)

@show vmin_o, vmax_o

vmin_m = quantile(skipmissing(colors_mBD), 0.05)
vmax_m = quantile(skipmissing(colors_mBD), 0.95)


mkpath(joinpath(@__DIR__, "../figures/"))

points_css = Matrix(df_filter_mean[:, [:clay, :silt, :sand, Symbol("$(obd)_function")]])
no_mss = replace(points_css, missing=>NaN)

C = makecpt(cmap=:hot, range=(vmin_o, vmax_o), reverse=true, bg=true);

# this only works in the REPL, it fails in vs-code (panel issues)
ternary(no_mss, marker=:p, cmap=C, image=true, figsize=10,
    frame = (
        annot=:auto,
        grid=:a,
        ticks=:a,
        alabel="Sand",
        blabel="Silt",
        clabel="Clay",
        suffix=" %",
        fill=:grey95,
        ),
    par=(FONT_ANNOT_PRIMARY=12,
        FONT_LABEL=14,
        MAP_LABEL_OFFSET="8p",
        MAP_ANNOT_OFFSET_PRIMARY="6p"
        )
        )
colorbar!(pos=(paper=true, anchor=(11,0), size=(8,0.5), justify=:BL, vertical=true, triangles=true),
    par=(FONT_ANNOT_PRIMARY=14, FONT_LABEL=14,),
    frame=(
        annot=:auto,
        ticks=:auto,
        xlabel=rich("oBD (g / cm",  superscript("3"), ")")),
        show=false, # true, only works in the REPL, it fails in vs-code (panel issues)
        savefig=joinpath(@__DIR__, "../figures/latent_oBD_2_$(method).pdf")
        )

#! now for mBD
points_mBD = Matrix(df_filter_mean[:, [:clay, :silt, :sand, Symbol("$(mbd)_function")]])
no_mBD = replace(points_mBD, missing=>NaN)

C = makecpt(cmap=:viridis, range=(vmin_m, vmax_m), reverse=true, bg=true);

ternary(no_mBD, marker=:p, cmap=C, ms=0.1, image=true, figsize=10,
    frame = (
        annot=:auto,
        grid=:a,
        ticks=:a,
        alabel="Sand",
        blabel="Silt",
        clabel="Clay",
        suffix=" %",
        fill=:grey95,),
    par=(FONT_ANNOT_PRIMARY=12,
        FONT_LABEL=14,
        MAP_LABEL_OFFSET="8p",
        MAP_ANNOT_OFFSET_PRIMARY="6p"
        )
        )
colorbar!(pos=(paper=true, anchor=(11,0), size=(8,0.5), justify=:BL, vertical=true, triangles=true),
    par=(FONT_ANNOT_PRIMARY=14, FONT_LABEL=14,),
    frame=(
        annot=:auto,
        ticks=:auto,
        xlabel=rich("mBD (g /cm",  superscript("3"), ")")),
        show=false, # true, only works in the REPL, it fails in vs-code (panel issues)
        savefig=joinpath(@__DIR__, "../figures/latent_mBD_2_$(method).pdf")
        )
end

for method in present_methods(df, method -> [pred_col(mod, target, method) for mod in models_raw for target in ("SOCconc", "CF", "BD", "SOCdensity")] ∪ ["pred_oBD_$(method)", "pred_mBD_$(method)"])
    draw_ternary(copy(df), method)
end

# Executing from the REPL
# cd viz-ternary-gmt
# julia
# pkg > activate .
# include("latent_params_ternary.jl")
