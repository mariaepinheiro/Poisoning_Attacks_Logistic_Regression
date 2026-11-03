# ==============================================================================
# bounds.jl — Valid bounds (Section 3.4)
# ==============================================================================
#   mode = :prop3  Bounds from Proposition 3 only (used by PA_G and CAR_0):
#                  β ∈ [-R,R]^d, |θ_i^tr| ≤ R‖x_i^tr‖, |z^p| ≤ R√d,
#                  α from σ(θ), h ∈ [0, ᾱ].
#   mode = :full   All bounds of Section 3.4 (used by CAR):
#                  Prop. 3 (R), Prop. 4 (β^0), convex problems (31) for β and
#                  (32) for θ^tr over B, inexpensive bounds (33)-(34) for z^p
#                  (with x_{k,1} = 1), then (35), (38)-(39), (40).
#
# Convex problems are solved with Ipopt; each value v is relaxed by
# 1e-6 (1 + |v|) and intersected with the closed-form bound.
# All matrices X include the intercept column (X[:,1] == 1).
# ==============================================================================

using JuMP, Ipopt, LinearAlgebra

_σ(z::Real) = z >= 0 ? 1 / (1 + exp(-z)) : (e = exp(z); e / (1 + e))
_φ(z::Real) = z > 0 ? z + log1p(exp(-z)) : log1p(exp(z))
_margin(v::Real) = 1e-6 * (1 + abs(v))

struct ModelBounds
    mode   :: Symbol
    R      :: Float64
    β_lo   :: Vector{Float64};  β_up   :: Vector{Float64}
    θtr_lo :: Vector{Float64};  θtr_up :: Vector{Float64}
    zp_lo  :: Float64;          zp_up  :: Float64
    θp_lo  :: Vector{Float64};  θp_up  :: Vector{Float64}
    αtr_lo :: Vector{Float64};  αtr_up :: Vector{Float64}
    αp_lo  :: Vector{Float64};  αp_up  :: Vector{Float64}
    time   :: Float64
    info   :: Dict{String,Any}
end

# ── Proposition 3: largest root of H(R) = N log 2 (bisection) ────────────────
function R_prop3(X::Matrix{Float64}, q::Int, λ::Real)
    n, d = size(X)
    c  = (n + q) * log(2)
    nx = [norm(view(X, i, :)) for i in 1:n]
    sd = sqrt(d)
    H(r) = 2λ * r^2 + sum(_φ(-r * v) for v in nx) + q * _φ(-r * sd)
    hi = sqrt(c / (2λ))
    while H(hi) <= c
        hi *= 2
    end
    lo = 0.0
    for _ in 1:200
        mid = (lo + hi) / 2
        H(mid) <= c ? (lo = mid) : (hi = mid)
        hi - lo <= 1e-10 * (1 + hi) && break
    end
    return hi                      # H(hi) > c  ⇒  hi ≥ R  (safe side)
end

# ── Proposition 4: componentwise bounds ──────────────────────────────────────
function beta_box(X, y, yp, λ, R)
    n, d = size(X)
    q0 = count(==(0.0), yp);  q1 = count(==(1.0), yp)
    nx = [norm(view(X, i, :)) for i in 1:n]
    sR = _σ(R * sqrt(d))
    C0 = zeros(d);  C1 = zeros(d)
    for i in 1:n
        si = _σ(R * nx[i])
        for j in 1:d
            v = X[i, j] * si
            y[i] == 0.0 ? (C0[j] += v) : (C1[j] += v)
        end
    end
    C0 .+= q0 * sR;  C1 .+= q1 * sR
    return -min.(R, C0 ./ (2λ)), min.(R, C1 ./ (2λ))
end

# ── (35), (38)-(39) ──────────────────────────────────────────────────────────
function _finish(mode, R, βlo, βup, θlo, θhi, zlo, zhi, yp, ε, t0, info)
    sp   = 1 .- 2 .* yp
    θplo = [s > 0 ?  zlo : -zhi for s in sp]
    θphi = [s > 0 ?  zhi : -zlo for s in sp]
    n_clamped = 0
       function αint(a, b)
        lo, hi = _σ(a), _σ(b)
        (lo < ε || hi > 1 - ε) && (n_clamped += 1)
        lo, hi = clamp(lo, ε, 1 - ε), clamp(hi, ε, 1 - ε)
        lo > hi && ((lo, hi) = (hi, lo))
        return lo, hi
    end
    αtr = [αint(θlo[i], θhi[i]) for i in eachindex(θlo)]
    αp  = [αint(θplo[k], θphi[k]) for k in eachindex(θplo)]
    info["n_alpha_clamped"] = n_clamped
    return ModelBounds(mode, R, βlo, βup, θlo, θhi, zlo, zhi, θplo, θphi,
                       first.(αtr), last.(αtr), first.(αp), last.(αp),
                       time() - t0, info)
end

# ── The convex set B (30) ────────────────────────────────────────────────────
function _B_model(X, s, q, λ, R, βlo, βup; sub_time = 60.0)
    n, d = size(X)
    m = Model(optimizer_with_attributes(Ipopt.Optimizer,
              "print_level" => 0, "sb" => "yes", "tol" => 1e-9,
              "acceptable_tol" => 1e-7, "max_iter" => 3000,
              "max_cpu_time" => sub_time))
    @variable(m, βlo[j] <= β[j = 1:d] <= βup[j], start = 0.0)
    rhs = (n + q) * log(2) - q * _φ(-R * sqrt(d))
    @constraint(m, sum(β[j]^2 for j in 1:d) <= R^2)
    @expression(m, a[i = 1:n],
                s[i] * sum(X[i, j] * β[j] for j in 1:d if X[i, j] != 0))
    @constraint(m, sum(log(1 + exp(a[i])) for i in 1:n) +
                   2λ * sum(β[j]^2 for j in 1:d) <= rhs)
    return m, β
end

_ok(m) = termination_status(m) in (MOI.LOCALLY_SOLVED, MOI.OPTIMAL,
                                   MOI.ALMOST_LOCALLY_SOLVED) &&
         primal_status(m) in (MOI.FEASIBLE_POINT, MOI.NEARLY_FEASIBLE_POINT)

function _minmax!(m, β, c)
    idx = [j for j in eachindex(c) if c[j] != 0]
    @objective(m, Min, sum(c[j] * β[j] for j in idx))
    optimize!(m)
    vmin = _ok(m) ? objective_value(m) : -Inf
    set_objective_sense(m, MOI.MAX_SENSE)
    optimize!(m)
    vmax = _ok(m) ? objective_value(m) : Inf
    return vmin, vmax
end

# ── Main entry point ─────────────────────────────────────────────────────────
function compute_bounds(X::Matrix{Float64}, y::Vector{Float64},
                        yp::Vector{Float64}, λ::Real;
                        mode::Symbol = :full, ε::Float64 = 1e-8,
                        theta::Bool = true, sub_time::Float64 = 60.0,
                        budget::Float64 = 3600.0)
    t0 = time()
    n, d = size(X);  q = length(yp);  sd = sqrt(d)
    s  = 1 .- 2 .* y
    nx = [norm(view(X, i, :)) for i in 1:n]
    info = Dict{String,Any}()
    R = R_prop3(X, q, λ)

    if mode == :prop3
        return _finish(:prop3, R, fill(-R, d), fill(R, d),
                       -R .* nx, R .* nx, -R * sd, R * sd, yp, ε, t0, info)
    end
    mode == :full || error("unknown mode $mode")

    # Proposition 4
    β0lo, β0up = beta_box(X, y, yp, λ, R)

    # (31): β_j over B
    n_fail = 0
    t1 = time()
    m, β = _B_model(X, s, q, λ, R, β0lo, β0up; sub_time)
    βlo, βup = copy(β0lo), copy(β0up)
    for j in 1:d
        time() - t0 > budget && (info["budget_hit"] = "beta"; break)
        c = zeros(d);  c[j] = 1.0
        vmin, vmax = _minmax!(m, β, c)
        isfinite(vmin) ? (βlo[j] = max(β0lo[j], vmin - _margin(vmin))) : (n_fail += 1)
        isfinite(vmax) ? (βup[j] = min(β0up[j], vmax + _margin(vmax))) : (n_fail += 1)
        βlo[j] > βup[j] && ((βlo[j], βup[j]) = (β0lo[j], β0up[j]))
    end
    info["t_beta"] = time() - t1

    # θ^tr: interval from the box, then (32) over B
    θlo = zeros(n);  θhi = zeros(n)
    for i in 1:n
        a = max(dot(βlo, view(X, i, :)), -R * nx[i])
        b = min(dot(βup, view(X, i, :)),  R * nx[i])
        θlo[i], θhi[i] = s[i] > 0 ? (a, b) : (-b, -a)
    end
    t2 = time()
    if theta
        for i in 1:n
            time() - t0 > budget && (info["budget_hit"] = "theta"; break)
            vmin, vmax = _minmax!(m, β, s[i] .* X[i, :])
            isfinite(vmin) ? (θlo[i] = max(θlo[i], vmin - _margin(vmin))) : (n_fail += 1)
            isfinite(vmax) ? (θhi[i] = min(θhi[i], vmax + _margin(vmax))) : (n_fail += 1)
        end
    end
    info["t_theta"] = time() - t2
    info["n_fail"] = n_fail

    # (33)-(34) with x_{k,1} = 1
    zlo = max(-R * sd, βlo[1] + sum(min(0.0, βlo[j]) for j in 2:d))
    zhi = min( R * sd, βup[1] + sum(max(0.0, βup[j]) for j in 2:d))

    return _finish(:full, R, βlo, βup, θlo, θhi, zlo, zhi, yp, ε, t0, info)
end

# ── Diagnostics ──────────────────────────────────────────────────────────────
function bounds_violation(B::ModelBounds, X, y, β; Xp = nothing)
    v = maximum(max.(B.β_lo .- β, β .- B.β_up, 0.0))
    θ = (1 .- 2 .* y) .* (X * β)
    v = max(v, maximum(max.(B.θtr_lo .- θ, θ .- B.θtr_up, 0.0)))
    if Xp !== nothing && !isempty(Xp)
        z = Xp * β
        v = max(v, maximum(max.(B.zp_lo .- z, z .- B.zp_up, 0.0)))
    end
    return v
end

bounds_summary(B::ModelBounds) = (
    R            = B.R,
    beta_width   = sum(B.β_up .- B.β_lo) / length(B.β_lo),
    theta_width  = sum(B.θtr_up .- B.θtr_lo) / length(B.θtr_lo),
    zp_width     = B.zp_up - B.zp_lo,
    alphap_width = sum(B.αp_up .- B.αp_lo) / max(1, length(B.αp_lo)),
    time         = B.time)
