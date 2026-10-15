# ══════════════════════════════════════════════════════════════════════════════
# POISONING ATTACK BENCHMARK — Datasets for Logistic Regression
# ══════════════════════════════════════════════════════════════════════════════
#
# This script loads, cleans, and binarizes 7 datasets from two sources
# (RDatasets.jl, UCI ML Repository) into a uniform Dict of (X, y) matrices
# suitable for evaluating data poisoning attacks on logistic regression.
# The eighth dataset of the paper, Spambase, is added by spambase_loader.jl.
# UCI files are downloaded into data/ on first use.
#
# Output: benchmarks::Dict{String, NamedTuple}
#   benchmarks["name"].X  → Matrix{Float64} (n × p), feature matrix
#   benchmarks["name"].y  → Vector{Float64} (n,) ∈ {0.0, 1.0}, binary target
#   benchmarks["name"].n  → Int, number of observations after cleaning
#   benchmarks["name"].p  → Int, number of features
#   benchmarks["name"].balance → Float64, proportion of positive class
#   benchmarks["name"].label   → String, human-readable description
#
# Requirements: RDatasets, DataFrames, Statistics, CSV, Downloads, Printf
# ══════════════════════════════════════════════════════════════════════════════

using RDatasets, DataFrames, Statistics, CSV, Downloads, Printf

benchmarks = Dict{String, NamedTuple}()


"""
    store!(key, X, y, label)

Validate and store a processed dataset into the global `benchmarks` Dict.

Performs two assertions before storing:
  1. The cleaned dataset must retain at least 100 observations.
  2. The target vector must be strictly binary ({0.0, 1.0}).

If both pass, the dataset is stored as a NamedTuple with fields
`X`, `y`, `n`, `p`, `balance`, and `label`, and a summary line is printed.
"""
function store!(key::String, X::Matrix{Float64}, y::Vector{Float64}, label::String)
    n, p = size(X)
    bal = round(mean(y), digits=3)
    n0, n1 = Int(sum(y .== 0)), Int(sum(y .== 1))
    @assert n >= 100 "Dataset $key: n=$n < 100 after cleaning"
    @assert sort(unique(y)) == [0.0, 1.0] "Dataset $key: target is not binary {0,1}"
    benchmarks[key] = (X=X, y=y, n=n, p=p, balance=bal, label=label)
    println("  ✓ n=$n  p=$p  P(y=1)=$bal  classes=($n0, $n1)")
end


"""
    select_numeric_features(df, target; min_unique=1)

Programmatically select feature columns from a DataFrame.

Returns a `Vector{Symbol}` of all columns that are (a) numeric,
(b) not the target column, and (c) have more than `min_unique`
distinct values.
"""
function select_numeric_features(df::DataFrame, target::Symbol; min_unique::Int=1)
    return [Symbol(n) for n in names(df)
            if Symbol(n) != target &&
               nonmissingtype(eltype(df[!, n])) <: Number &&
               length(unique(skipmissing(df[!, n]))) > min_unique]
end


"""
    load_uci_csv(url, local_name)

Download a CSV from the UCI ML Repository into `data/` (if not already
cached) and return it as a headerless DataFrame.
"""
function load_uci_csv(url::String, local_name::String)
    datadir = joinpath(@__DIR__, "..", "data")
    mkpath(datadir)
    filepath = joinpath(datadir, local_name)
    isfile(filepath) || Downloads.download(url, filepath)
    return CSV.read(filepath, DataFrame; header=false)
end


# ══════════════════════════════════════════════════════════════════════════════
# DATASET LOADING & PROCESSING
# ══════════════════════════════════════════════════════════════════════════════


# ──────────────────────────────────────────────────────────────────────────────
## 1. MASS::biopsy (Wisconsin Breast Cancer — Cell Measurements)
# ──────────────────────────────────────────────────────────────────────────────
# Threat Model: Compromised Cancer Diagnosis.
# An insider at a pathology lab injects targeted noise into cell measurements
# (V1–V9: clump thickness, uniformity, marginal adhesion). Goal: induce False
# Negatives in the classifier, delaying malignant cancer diagnoses.
# ──────────────────────────────────────────────────────────────────────────────
begin
    println("\n[1/7] MASS::biopsy")
    raw = dataset("MASS", "biopsy")
    @info "Columns: $(names(raw))  Rows: $(nrow(raw))"

    features = [:V1, :V2, :V3, :V4, :V5, :V6, :V7, :V8, :V9]
    sub = dropmissing(raw[:, [features..., :Class]])

    X = Matrix{Float64}(sub[:, features])
    y = Float64.(sub.Class .== "malignant")

    store!("biopsy", X, y, "Breast Cancer Wisconsin (Healthcare)")
end


# ──────────────────────────────────────────────────────────────────────────────
## 2. ISLR::Smarket (S&P 500 Daily Returns — Quantitative Trading)
# ──────────────────────────────────────────────────────────────────────────────
# Threat Model: Quantitative Trading Model Manipulation.
# An adversary with access to a quant fund's data feed poisons lagged returns
# (Lag1–Lag5) and historical Volume. Goal: make the logistic model predict
# Direction="Up" when the market is falling, inducing wrong-direction trades.
# ──────────────────────────────────────────────────────────────────────────────
begin
    println("\n[2/7] ISLR::Smarket")
    raw = dataset("ISLR", "Smarket")
    @info "Columns: $(names(raw))  Rows: $(nrow(raw))"

    features = [:Lag1, :Lag2, :Lag3, :Lag4, :Lag5, :Volume]
    sub = dropmissing(raw[:, [features..., :Direction]])

    X = Matrix{Float64}(sub[:, features])
    y = Float64.(sub.Direction .== "Up")

    store!("Smarket", X, y, "S&P 500 Returns (Finance)")
end


# ──────────────────────────────────────────────────────────────────────────────
## 3. MASS::fgl (Forensic Glass Fragments — Criminal Investigation)
# ──────────────────────────────────────────────────────────────────────────────
# Threat Model: Forensic Evidence Tampering.
# A criminal with access to the forensic lab system manipulates chemical
# composition measurements (RI, Na, Mg, Al, Si, K, Ca, Ba, Fe) of glass
# fragments. Goal: make the classifier categorize crime-scene window glass
# (WinF/WinNF) as non-window glass, weakening the forensic evidence.
# Binarization: type ∈ {"WinF","WinNF"} → 1, else → 0.
# ──────────────────────────────────────────────────────────────────────────────
begin
    println("\n[3/7] MASS::fgl")
    raw = dataset("MASS", "fgl")
    @info "Columns: $(names(raw))  Rows: $(nrow(raw))"

    features = [:RI, :Na, :Mg, :Al, :Si, :K, :Ca, :Ba, :Fe]
    sub = dropmissing(raw[:, [features..., :Type]])

    X = Matrix{Float64}(sub[:, features])
    y = Float64.(string.(sub.Type) .∈ Ref(["WinF", "WinNF"]))

    store!("fgl", X, y, "Forensic Glass (Criminal)")
end


# ──────────────────────────────────────────────────────────────────────────────
## 4. ISLR::College (University Classification — Ranking Manipulation)
# ──────────────────────────────────────────────────────────────────────────────
# Threat Model: University Ranking Manipulation.
# An attacker with access to a university's reporting system alters continuous
# metrics (Apps, Accept, Outstate, Expend, Grad.Rate) so that a classification
# model mislabels a public university as private (or vice-versa), manipulating
# federal funding allocation or regulatory status.
# ──────────────────────────────────────────────────────────────────────────────
begin
    println("\n[4/7] ISLR::College")
    raw = dataset("ISLR", "College")
    @info "Columns: $(names(raw))  Rows: $(nrow(raw))"

    target_col = :Private
    feature_cols = select_numeric_features(raw, target_col)
    @info "Selected features: $feature_cols ($(length(feature_cols)))"

    sub = dropmissing(raw[:, [feature_cols..., target_col]])

    X = Matrix{Float64}(sub[:, feature_cols])
    y = Float64.(sub[:, target_col] .== "Yes")

    store!("College", X, y, "University Private/Public (Education)")
end


# ──────────────────────────────────────────────────────────────────────────────
## 5. ISLR::OJ (Orange Juice Purchase — Consumer Choice)
# ──────────────────────────────────────────────────────────────────────────────
# Threat Model: Consumer Preference Model Manipulation.
# A retailer poisons training data of a brand preference model by altering
# price/discount/loyalty features. Goal: make the model systematically favor
# one brand (CH) over the other (MM), biasing pricing strategies.
# ──────────────────────────────────────────────────────────────────────────────
begin
    println("\n[5/7] ISLR::OJ")
    raw = dataset("ISLR", "OJ")
    @info "Columns: $(names(raw))  Rows: $(nrow(raw))"

    target_col = :Purchase
    feature_cols = select_numeric_features(raw, target_col; min_unique=10)
    @info "Selected features: $feature_cols ($(length(feature_cols)))"

    sub = dropmissing(raw[:, [feature_cols..., target_col]])

    X = Matrix{Float64}(sub[:, feature_cols])
    y = Float64.(sub[:, target_col] .== "CH")

    store!("OJ", X, y, "Orange Juice Brand Choice (Retail)")
end


# ──────────────────────────────────────────────────────────────────────────────
## 6. UCI Banknote Authentication (Counterfeit Detection)
# ──────────────────────────────────────────────────────────────────────────────
# Threat Model: Counterfeit Detector Sabotage.
# An attacker corrupts training data of an ATM banknote authentication system
# by altering wavelet-derived features (variance, skewness, kurtosis, entropy
# of the banknote image). Goal: shift the decision boundary so that forged
# notes are classified as authentic, enabling large-scale money laundering.
# 1372 obs, 4 continuous features, binary target (0=authentic, 1=forged).
# ──────────────────────────────────────────────────────────────────────────────
begin
    println("\n[6/7] UCI::Banknote")
    raw = load_uci_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt",
        "banknote.csv"
    )
    @info "Columns: $(names(raw))  Rows: $(nrow(raw))"

    features = [:Column1, :Column2, :Column3, :Column4]
    sub = dropmissing(raw)

    X = Matrix{Float64}(sub[:, features])
    y = Float64.(sub.Column5)

    store!("Banknote", X, y, "Banknote Authentication (Counterfeit)")
end


# ──────────────────────────────────────────────────────────────────────────────
## 7. UCI Ionosphere (Radar Signal Classification)
# ──────────────────────────────────────────────────────────────────────────────
# Threat Model: Radar Surveillance Sabotage.
# An adversary poisons training data of an ionospheric radar return classifier
# (34 continuous pulse features). Goal: make "good" signals (showing structure
# in the ionosphere) be classified as "bad" (no structure), effectively
# blinding the aerial surveillance system.
# Binarization: "g" → 1, "b" → 0.
# 351 obs, 34 features. Column2 is constant (all zeros) and is removed.
# ──────────────────────────────────────────────────────────────────────────────
begin
    println("\n[7/7] UCI::Ionosphere")
    raw = load_uci_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data",
        "ionosphere.csv"
    )
    @info "Columns: $(names(raw))  Rows: $(nrow(raw))"

    target_col = Symbol(names(raw)[end])
    drop_cols = Set([string(target_col)])

    # Remove zero-variance columns (e.g. Column2 is all zeros)
    for n in names(raw)
        Symbol(n) == target_col && continue
        col = raw[!, n]
        if nonmissingtype(eltype(col)) <: Number && length(unique(skipmissing(col))) <= 1
            push!(drop_cols, n)
            @info "  Constant column removed: $n"
        end
    end

    feature_cols = [Symbol(n) for n in names(raw)
                    if n ∉ drop_cols && nonmissingtype(eltype(raw[!, n])) <: Number]
    @info "Features retained: $(length(feature_cols))"

    sub = dropmissing(raw)
    X = Matrix{Float64}(sub[:, feature_cols])
    y = Float64.(sub[:, target_col] .== "g")

    store!("Ionosphere", X, y, "Ionosphere Radar (Surveillance)")
end


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════
println("\n", "═"^70)
@printf("%-20s %6s %4s %8s %10s   %s\n",
        "Dataset", "n", "p", "P(y=1)", "n/p", "Label")
println("─"^70)
for (key, b) in sort(collect(benchmarks), by=x->x.second.n)
    ratio = round(b.n / b.p, digits=1)
    @printf("%-20s %6d %4d %8.3f %10.1f   %s\n",
            key, b.n, b.p, b.balance, ratio, b.label)
end
println("═"^70)
println("Total: $(length(benchmarks)) datasets loaded.")
println()
println("Access: benchmarks[\"biopsy\"].X, benchmarks[\"biopsy\"].y, etc.")