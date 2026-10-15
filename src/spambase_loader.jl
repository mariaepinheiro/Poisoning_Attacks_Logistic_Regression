# ==============================================================================
# spambase_loader.jl — adds "Spambase" to `benchmarks`
#   UCI Spambase: 4601 e-mails, 57 numeric features, label 1 = spam.
#   Downloads the raw file on first use (no header; last column = label).
# ==============================================================================
using CSV, DataFrames, Downloads

function load_spambase!(benchmarks, path::String)
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
    if !isfile(path)
        mkpath(dirname(path))
        @info "Downloading Spambase ..."
        Downloads.download(url, path)
    end
    df = CSV.read(path, DataFrame; header = false)
    dropmissing!(df)
    X = Matrix{Float64}(df[:, 1:end-1])
    y = Float64.(df[:, end])
    keep = [j for j in 1:size(X, 2) if length(unique(view(X, :, j))) > 1]
    benchmarks["Spambase"] = (X = X[:, keep], y = y)
    @info "Spambase: $(size(X,1)) obs, $(length(keep)) features, P(y=1) = $(round(sum(y)/length(y), digits=2))"
end
