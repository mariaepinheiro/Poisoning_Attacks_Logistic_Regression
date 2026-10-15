# ==============================================================================
# models_bounded.jl — PA_G (P2 as written) and CAR / CAR_0 (P3)
# ==============================================================================
# Requires models_global_regularization.jl (get_optimizer, _extract,
# with_log_file) and bounds.jl.  All matrices include the intercept column.
# ==============================================================================

using JuMP, Random

# ──────────────────────────────────────────────────────────────────────────────
# PA_G / PA_L: problem (P2) exactly as written, exact stationarity, with the bounds
#       of Proposition 3 on β and on the inner products.
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
    # inner products β'x (named, so that their bounds can be imposed)
    @variable(model, zlo[i] <= z_t[i = 1:n] <= zhi[i])
    @variable(model, B.zp_lo <= z_p[1:q] <= B.zp_up)
    @constraint(model, [i = 1:n], z_t[i] == sum(X[i, j] * β[j] for j in 1:d))
    @constraint(model, [k = 1:q], z_p[k] == sum(Xp[k, j] * β[j] for j in 1:d))

    # (P2a): L(D^tr, β) = Σ log(1 + e^{z_i}) − y_i z_i
    @objective(model, Max, sum(log(1 + exp(z_t[i])) - y[i] * z_t[i] for i in 1:n))

    # (P2c): X'(m_β(X) − y) + 2λβ = 0
    @constraint(model, [j = 1:d],
        sum(X[i, j] * (1 / (1 + exp(-z_t[i])) - y[i]) for i in 1:n) +
        sum(Xp[k, j] * (1 / (1 + exp(-z_p[k])) - yp[k]) for k in 1:q) +
        2λ * β[j] == 0)

    # random starting point (multistart): β ~ N(0, 0.1² I), x^p ~ U[0,1]
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
    # (P3f) convex Fenchel-gap constraint
    ϕs(a) = a * log(a) + (1 - a) * log(1 - a)
    @constraint(model,
        sum(log(1 + exp(θ_t[i])) + ϕs(α_t[i]) for i in 1:n) +
        sum(log(1 + exp(θ_p[k])) + ϕs(α_p[k]) for k in 1:q) +
        2λ * sum(β[j]^2 for j in 1:d) <= 0)
    # (P3a) concave objective
    @objective(model, Max,
        - sum(ϕs(α_t[i]) for i in 1:n)
        - sum(log(1 + exp(θ_p[k])) + ϕs(α_p[k]) for k in 1:q)
        - 2λ * sum(β[j]^2 for j in 1:d))

    solver == :SCIP ? with_log_file(() -> optimize!(model), log_file) : optimize!(model)
    r = _extract(model)

    Xp = fill(NaN, q, d)
    if r.has_solution
        hv = value.(h);  av = value.(α_p)
        for k in 1:q, j in 1:d
            Xp[k, j] = hv[k, j] / av[k]
        end
    end
    return (beta = r.has_solution ? value.(β) : fill(NaN, d), X_p = Xp,
            objective = r.objective, objective_bound = r.objective_bound,
            relative_gap = r.relative_gap, solve_time = r.solve_time,
            status = r.status, has_solution = r.has_solution)
end
