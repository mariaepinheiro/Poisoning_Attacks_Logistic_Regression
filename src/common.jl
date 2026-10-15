# ==============================================================================
# common.jl — split, poisoning labels, metrics (identical to the first round)
# ==============================================================================
using Random

# 30 % of each class, rounded to the nearest integer
function stratified_split(X, y, ratio; rng)
    i0 = findall(y .== 0.0);  i1 = findall(y .== 1.0)
    n0 = max(1, round(Int, length(i0) * ratio))
    n1 = max(1, round(Int, length(i1) * ratio))
    shuffle!(rng, i0);  shuffle!(rng, i1)
    tr = vcat(i0[1:n0], i1[1:n1]);  te = vcat(i0[n0+1:end], i1[n1+1:end])
    shuffle!(rng, tr);  shuffle!(rng, te)
    return X[tr, :], y[tr], X[te, :], y[te]
end

# q = max(1, round(ρ n))
poison_budget(n, ρ) = max(1, round(Int, n * ρ))

# ⌈q/2⌉ labels 0 and ⌊q/2⌋ labels 1, alternating
poison_labels(q) = Float64[mod(i - 1, 2) for i in 1:q]

# accuracy and MCC; Xb includes the intercept column
function metrics(β, Xb, y)
    any(isnan, β) && return (accuracy = NaN, mcc = NaN)
    pred = Float64.(1.0 ./ (1.0 .+ exp.(-(Xb * β))) .>= 0.5)
    tp = sum((pred .== 1) .& (y .== 1));  tn = sum((pred .== 0) .& (y .== 0))
    fp = sum((pred .== 1) .& (y .== 0));  fn = sum((pred .== 0) .& (y .== 1))
    den = sqrt(Float64(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (accuracy = (tp + tn) / length(y),
            mcc = den == 0 ? 0.0 : (tp * tn - fp * fn) / den)
end
