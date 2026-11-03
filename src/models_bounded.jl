# ==============================================================================
# models_bounded.jl — PA_G (P2 as written) and CAR / CAR_0 (P3)
# ==============================================================================
# Requires models_global_regularization.jl (get_optimizer, _extract,
# with_log_file) and bounds.jl (ModelBounds, _φ).  All matrices include the
# intercept column.
#
# Changes in solve_CAR (review notes 13 and 18):
#   - softplus terms log(1+e^θ) are replaced by epigraph variables u with
#     e^(θ−u) + e^(−u) ≤ 1, a form SCIP recognizes as convex;
#   - α is no longer clipped to [1e-8, 1−1e-8]; the complements 1−α are
#     separate variables, so both entropy terms have the form x·log(x),
#     which SCIP handles as its `entropy` operator (with 0·log 0 = 0).
# solve_PAG is unchanged.
# ==============================================================================

using JuMP, Random

# ──────────────────────────────────────────────────────────────────────────────
# PA_G / PA_L: problem (P2) exactly as written, exact stationarity, with the bounds
#       of Proposition 3 on β and on the inner products.   (UNCHANGED)
# ──────────────────────────────────────────────────────────────────────────────
function solve_PAG(X, y, yp, λ, B::ModelBounds;
                   solver::Symbol = :SCIP, time_limit::Float64 = 3600.0,
                   silent::Bool = true, log_file::Union{String,Nothing} = nothing,
                   rng::Union{AbstractRNG,Nothing} = nothing)
    n, d = size(X);  q = length(yp)
    s_tr = 1 .- 2 .* y
    zlo = [s_tr[i] > 0 ?  B.θtr_lo[i] : -B.θtr_up[i] for i in 1:n]
    zhi = [s_tr[i] > 0 ?  B.θtr_up[i] : -B.θtr_lo[i] for i in 1:n]

    model = Model(get_optimizer(solver; time_limit, silent, log_file))

    @variable(model, B.β_lo[j] <= β[j = 1:d] <= B.β_up[j])
    @variable(model, Xp[1:q, 1:d])
    for k in 1:q
        fix(Xp[k, 1], 1.0; force = true)
        for j in 2:d
            set_lower_bound(Xp[k, j], 0.0);  set_upper_bound(Xp[k, j], 1.0)
        end
    end
    @variable(model, zlo[i] <= z_t[i = 1:n] <= zhi[i])
    @variable(model, B.zp_lo <= z_p[1:q] <= B.zp_up)
    @constraint(model, [i = 1:n], z_t[i] == sum(X[i, j] * β[j] for j in 1:d))
    @constraint(model, [k = 1:q], z_p[k] == sum(Xp[k, j] * β[j] for j in 1:d))

    @objective(model, Max, sum(log(1 + exp(z_t[i])) - y[i] * z_t[i] for i in 1:n))

    @constraint(model, [j = 1:d],
        sum(X[i, j] * (1 / (1 + exp(-z_t[i])) - y[i]) for i in 1:n) +
        sum(Xp[k, j] * (1 / (1 + exp(-z_p[k])) - yp[k]) for k in 1:q) +
        2λ * β[j] == 0)

    if rng !== nothing
        β0  = clamp.(0.1 .* randn(rng, d), B.β_lo, B.β_up)
        Xp0 = rand(rng, q, d);  Xp0[:, 1] .= 1.0
        for j in 1:d;  set_start_value(β[j], β0[j]);  end
        for k in 1:q, j in 2:d;  set_start_value(Xp[k, j], Xp0[k, j]);  end
        zt0 = X * β0;  zp0 = Xp0 * β0
        for i in 1:n;  set_start_value(z_t[i], clamp(zt0[i], zlo[i], zhi[i]));  end
        for k in 1:q;  set_start_value(z_p[k], clamp(zp0[k], B.zp_lo, B.zp_up));  end
    end

    solver == :SCIP ? with_log_file(() -> optimize!(model), log_file) : optimize!(model)
    r = _extract(model)
    return (beta = r.has_solution ? value.(β)  : fill(NaN, d),
            X_p  = r.has_solution ? value.(Xp) : fill(NaN, q, d),
            objective = r.objective, objective_bound = r.objective_bound,
            relative_gap = r.relative_gap, solve_time = r.solve_time,
            status = r.status, has_solution = r.has_solution)
end

# ──────────────────────────────────────────────────────────────────────────────
# CAR / CAR_0: problem (P3).  The two differ only in the ModelBounds passed.
# ──────────────────────────────────────────────────────────────────────────────
function solve_CAR(X, y, yp, λ, B::ModelBounds;
                   solver::Symbol = :SCIP, time_limit::Float64 = 3600.0,
                   silent::Bool = true, log_file::Union{String,Nothing} = nothing)
    n, d = size(X);  q = length(yp)
    s_tr = 1 .- 2 .* y;  s_p = 1 .- 2 .* yp

    model = Model(get_optimizer(solver; time_limit, silent, log_file))
    # NEW: SCIP keeps arguments of log() at a distance ≥ minzerodistance from 0
    # (default 1e-9, which would act as a clipping of α).  1e-100 lies below
    # σ(−R√d) ≥ 1e-80 in all instances, so it excludes no feasible point.
    solver == :SCIP && set_attribute(model, "expr/log/minzerodistance", 1e-100)

    @variable(model, B.β_lo[j]   <= β[j = 1:d]   <= B.β_up[j])
    @variable(model, B.θtr_lo[i] <= θ_t[i = 1:n] <= B.θtr_up[i])
    @variable(model, B.θp_lo[k]  <= θ_p[k = 1:q] <= B.θp_up[k])
    @variable(model, B.αtr_lo[i] <= α_t[i = 1:n] <= B.αtr_up[i])
    @variable(model, B.αp_lo[k]  <= α_p[k = 1:q] <= B.αp_up[k])
    @variable(model, 0.0 <= h[k = 1:q, j = 1:d] <= B.αp_up[k])

    # (P3b)
    @constraint(model, [i = 1:n], θ_t[i] == s_tr[i] * sum(X[i, j] * β[j] for j in 1:d))
    # (P3c) and intercept h_{k,1} = α_k^p
    @constraint(model, [k = 1:q, j = 2:d], h[k, j] <= α_p[k])
    @constraint(model, [k = 1:q], h[k, 1] == α_p[k])
    # (P3d) bilinear
    @constraint(model, [k = 1:q],
        α_p[k] * θ_p[k] == s_p[k] * sum(β[j] * h[k, j] for j in 1:d))
    # (P3e) linear stationarity
    @constraint(model, [j = 1:d],
        sum(s_tr[i] * X[i, j] * α_t[i] for i in 1:n) +
        sum(s_p[k] * h[k, j] for k in 1:q) + 2λ * β[j] == 0)

    # NEW (note 18): complements c = 1 − α as variables, so that
    # ϕ*(α) = α log α + c log c consists of two x·log(x) terms (SCIP: entropy)
    # bounds of c = 1 − α = σ(−θ), computed from θ directly (1 − σ(θ) would
    # round to 0 for large θ)
    @variable(model, _σ(-B.θtr_up[i]) <= c_t[i = 1:n] <= _σ(-B.θtr_lo[i]))
    @variable(model, _σ(-B.θp_up[k])  <= c_p[k = 1:q] <= _σ(-B.θp_lo[k]))
    @constraint(model, [i = 1:n], c_t[i] == 1 - α_t[i])
    @constraint(model, [k = 1:q], c_p[k] == 1 - α_p[k])
    ent(a, c) = a * log(a) + c * log(c)

    # NEW (note 13): softplus in epigraph form,
    #   log(1 + e^θ) ≤ u   ⇔   e^(θ−u) + e^(−u) ≤ 1
    @variable(model, _φ(B.θtr_lo[i]) <= u_t[i = 1:n] <= _φ(B.θtr_up[i]))
    @variable(model, _φ(B.θp_lo[k])  <= u_p[k = 1:q] <= _φ(B.θp_up[k]))
    @constraint(model, [i = 1:n], exp(θ_t[i] - u_t[i]) + exp(-u_t[i]) <= 1)
    @constraint(model, [k = 1:q], exp(θ_p[k] - u_p[k]) + exp(-u_p[k]) <= 1)

    # (P3f) convex Fenchel-gap constraint
    @constraint(model,
        sum(u_t[i] + ent(α_t[i], c_t[i]) for i in 1:n) +
        sum(u_p[k] + ent(α_p[k], c_p[k]) for k in 1:q) +
        2λ * sum(β[j]^2 for j in 1:d) <= 0)
    # (P3a) concave objective
    @objective(model, Max,
        - sum(ent(α_t[i], c_t[i]) for i in 1:n)
        - sum(u_p[k] + ent(α_p[k], c_p[k]) for k in 1:q)
        - 2λ * sum(β[j]^2 for j in 1:d))

    solver == :SCIP ? with_log_file(() -> optimize!(model), log_file) : optimize!(model)
    r = _extract(model)

    Xp = fill(NaN, q, d)
    if r.has_solution
        hv = value.(h);  av = value.(α_p)
        for k in 1:q, j in 1:d
            Xp[k, j] = av[k] > 0 ? hv[k, j] / av[k] : NaN     # NEW: guard α = 0
        end
    end
    return (beta = r.has_solution ? value.(β) : fill(NaN, d), X_p = Xp,
            objective = r.objective, objective_bound = r.objective_bound,
            relative_gap = r.relative_gap, solve_time = r.solve_time,
            status = r.status, has_solution = r.has_solution)
end