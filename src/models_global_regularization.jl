
using JuMP, Ipopt, Random, Statistics
try using Gurobi catch; @info "Gurobi not available (only needed for solver = :Gurobi)" end
using SCIP


# ==============================================================================
# 1. SOLVER CONFIGURATION 
# ==============================================================================

function get_optimizer(solver::Symbol; time_limit::Float64=300.0,
                       silent::Bool=true, threads::Int=1, mip_gap::Float64=1e-4,
                       log_file::Union{String,Nothing}=nothing)
    if solver == :Ipopt
        attrs = Pair{String,Any}[
            "print_level"  => (silent && log_file === nothing ? 0 : 5),
            "max_iter"     => 10_000,
            "max_cpu_time" => time_limit,
            "tol"          => 1e-6,
        ]
        log_file !== nothing && push!(attrs, "output_file" => log_file)
        return optimizer_with_attributes(Ipopt.Optimizer, attrs...)

    elseif solver == :Gurobi
        attrs = Pair{String,Any}[
            "OutputFlag"     => (silent && log_file === nothing ? 0 : 1),
            "LogToConsole"   => (silent ? 0 : 1),
            "TimeLimit"      => time_limit,
            "NonConvex"      => 2,
            "FuncNonlinear"  => 1,
            "Threads"        => threads,
            "MIPGap"         => mip_gap,
            "NumericFocus"   => 2,
        ]
        log_file !== nothing && push!(attrs, "LogFile" => log_file)
        return optimizer_with_attributes(Gurobi.Optimizer, attrs...)

    elseif solver == :SCIP
        return optimizer_with_attributes(
            SCIP.Optimizer,
            "display/verblevel" => (silent && log_file === nothing ? 0 : 4),
            "limits/time"       => time_limit,
            "limits/gap"        => mip_gap,
        )

    else
        error("Unknown solver: $solver. Use :Ipopt, :Gurobi or :SCIP.")
    end
end


function with_log_file(f::Function, path::Union{String,Nothing})
    path === nothing && return f()
    mkpath(dirname(path))
    open(path, "w") do io
        redirect_stdout(io) do
            result = f()
            flush(io)
            return result
        end
    end
end


# ==============================================================================
# 2. HELPERS 
# ==============================================================================

function min_max_scale(X_train::Matrix{Float64}, X_test::Matrix{Float64};
                       scale_factor::Float64=1.0)
    min_vals = minimum(X_train, dims=1)
    max_vals = maximum(X_train, dims=1)
    denom = max_vals .- min_vals
    denom[denom .== 0.0] .= 1.0
    X_tr = (X_train .- min_vals) ./ denom .* scale_factor
    X_te = clamp.((X_test .- min_vals) ./ denom .* scale_factor, 0.0, scale_factor)
    return X_tr, X_te
end

add_bias(X::Matrix{Float64}) = hcat(ones(size(X, 1), 1), X)


function _extract(model::Model)
    st       = termination_status(model)
    has_sol  = has_values(model)
    obj      = has_sol ? objective_value(model) : NaN
    bound    = try objective_bound(model)  catch; NaN end
    gap      = try relative_gap(model)     catch; NaN end
    stime    = try JuMP.solve_time(model)  catch; NaN end
    return (status=st, has_solution=has_sol, objective=obj,
            objective_bound=bound, relative_gap=gap, solve_time=stime)
end


# ==============================================================================
# 3. CLEAN BASELINE 
# ==============================================================================

function logistic_regression_model(X_train::Matrix{Float64}, y_train::Vector{Float64};
                                   solver::Symbol=:Ipopt, time_limit::Float64=300.0,
                                   beta_bound::Union{Float64,Nothing}=nothing,
                                   l2_reg::Float64,
                                   threads::Int=1, silent::Bool=true)
    n, d  = size(X_train)
    model = Model(get_optimizer(solver; time_limit, threads, silent))

    if beta_bound === nothing
        @variable(model, beta[1:d])
    else
        @variable(model, -beta_bound <= beta[1:d] <= beta_bound)
    end

    @expression(model, z[i=1:n], sum(X_train[i, j] * beta[j] for j in 1:d))
    @objective(model, Min,
        sum(log(1 + exp(z[i])) - y_train[i] * z[i] for i in 1:n) +
        l2_reg * sum(beta[j]^2 for j in 1:d))

    optimize!(model)
    r = _extract(model)
    return (beta = r.has_solution ? value.(beta) : fill(NaN, d),
            solve_time = r.solve_time, status = r.status, objective = r.objective)
end


function logistic_regression(X_train::Matrix{Float64}, y_train::Vector{Float64},
                             X_test::Matrix{Float64}, y_test::Vector{Float64};
                             solver::Symbol=:Ipopt, time_limit::Float64=300.0,
                             beta_bound::Union{Float64,Nothing}=nothing,
                             l2_reg::Float64)
    X_tr_bias = add_bias(X_train)
    X_te_bias = add_bias(X_test)
    res = logistic_regression_model(X_tr_bias, y_train; solver, time_limit, beta_bound, l2_reg)

    prob = 1.0 ./ (1.0 .+ exp.(-(X_te_bias * res.beta)))
    accuracy = sum(Float64.(prob .>= 0.5) .== y_test) / length(y_test)

    return (accuracy=accuracy, beta=res.beta, solve_time=res.solve_time,
            status=res.status, objective=res.objective,
            beta_max_abs = maximum(abs, res.beta))
end


# ==============================================================================
# 3b. ANALYTICAL BOUNDS ON β AND z (from λ)
# ==============================================================================
#

"""
    analytical_beta_bound(n, q, l2_reg)

Exact upper bound on |β*_j| for the regularised lower-level problem.
`n` = training points, `q` = poisoning points, `l2_reg` = λ > 0.
Returns M = √((n+q) log(2) / λ).
"""
function analytical_beta_bound(n::Int, q::Int, l2_reg::Float64)
    l2_reg > 0 || error("analytical_beta_bound requires l2_reg > 0")
    return sqrt((n + q) * log(2) / l2_reg)
end

"""
    analytical_z_bound(n, q, d, l2_reg)

Exact upper bound on |β'x| for x ∈ [0,1]^d via Cauchy-Schwarz.
Returns Z = M √d where M = analytical_beta_bound(n, q, l2_reg).
`d` includes the bias column.
"""
function analytical_z_bound(n::Int, q::Int, d::Int, l2_reg::Float64)
    M = analytical_beta_bound(n, q, l2_reg)
    return M * sqrt(d)
end


# ==============================================================================
# 4. POISONING ATTACK (BILEVEL, KKT REFORMULATION) — with ridge regularization
# ==============================================================================

"""
    PA_logistic_regression_model(X_train, y_train, y_poisoning; l2_reg, kwargs...)

`X_train` must already include the bias column.
`l2_reg` (λ > 0) is REQUIRED — no default.

Both `beta_bound` (M) and `z_bound` (Z) are derived analytically from λ:
  M = √((n+q) log(2) / λ)       (ridge bound on ‖β*‖₂)
  Z = M √d                       (Cauchy-Schwarz on |β'x|)
Pass explicit values to override (e.g. for tighter solver envelopes).

Upper-level objective: L(D^tr, β̂) — pure training loss, no penalty.
KKT stationarity: ∇L + 2λβ = 0.
"""
function PA_logistic_regression_model(X_train::Matrix{Float64}, y_train::Vector{Float64},
                                      y_poisoning::Vector{Float64};
                                      solver::Symbol=:Ipopt,
                                      scale_factor::Float64=1.0,
                                      time_limit::Float64=300.0,
                                      rng::Union{AbstractRNG,Nothing}=nothing,
                                      beta_bound::Union{Float64,Nothing}=nothing,
                                      z_bound::Union{Float64,Nothing}=nothing,
                                      l2_reg::Float64,
                                      sigmoid_form::Symbol=:mult_aux,
                                      t_form::Symbol=:softplus,
                                      tol_kkt::Float64=1e-4,
                                      threads::Int=1,
                                      silent::Bool=true,
                                      log_file::Union{String,Nothing}=nothing)

    n, d = size(X_train)
    q    = length(y_poisoning)

    l2_reg > 0 || error("l2_reg must be > 0 (ridge regularization is required)")

    # Analytical bounds if not provided
    if beta_bound === nothing
        beta_bound = analytical_beta_bound(n, q, l2_reg)
    end
    if z_bound === nothing
        z_bound = analytical_z_bound(n, q, d, l2_reg)
    end

    sigmoid_form in (:division, :mult, :mult_aux) ||
        error("sigmoid_form must be :division, :mult or :mult_aux")
    t_form in (:softplus, :exp_aux) ||
        error("t_form must be :softplus or :exp_aux")

    model = Model(get_optimizer(solver; time_limit, threads, silent, log_file))

    # ── decision variables ────────────────────────────────────────────────────
    @variable(model, -beta_bound <= beta[1:d] <= beta_bound)

    @variable(model, X_p[1:q, 1:d])
    for j in 1:q
        fix(X_p[j, 1], 1.0; force=true)
        for k in 2:d
            set_lower_bound(X_p[j, k], 0.0)
            set_upper_bound(X_p[j, k], scale_factor)
        end
    end

    # ── inner products and sigmoids ────────────────────────────────────────────
    @variable(model, -z_bound <= z_tr[1:n] <= z_bound)
    @variable(model, -z_bound <= z_p[1:q]  <= z_bound)
    @variable(model, 0.0 <= s_tr[1:n] <= 1.0)
    @variable(model, 0.0 <= s_p[1:q]  <= 1.0)

    @constraint(model, def_z_tr[i=1:n],
        z_tr[i] == sum(X_train[i, k] * beta[k] for k in 1:d))
    @constraint(model, def_z_p[j=1:q],
        z_p[j] == sum(X_p[j, k] * beta[k] for k in 1:d))

    # ── sigmoid definitions (switchable) ───────────────────────────────────────
    if sigmoid_form == :division
        @constraint(model, def_s_tr[i=1:n], s_tr[i] == 1 / (1 + exp(-z_tr[i])))
        @constraint(model, def_s_p[j=1:q],  s_p[j]  == 1 / (1 + exp(-z_p[j])))

    elseif sigmoid_form == :mult
        @constraint(model, def_s_tr[i=1:n], s_tr[i] + s_tr[i] * exp(-z_tr[i]) == 1)
        @constraint(model, def_s_p[j=1:q],  s_p[j]  + s_p[j]  * exp(-z_p[j]) == 1)

    else # :mult_aux
        @variable(model, exp(-z_bound) <= u_tr[1:n] <= exp(z_bound))
        @variable(model, exp(-z_bound) <= u_p[1:q]  <= exp(z_bound))
        @constraint(model, def_u_tr[i=1:n], u_tr[i] == exp(-z_tr[i]))
        @constraint(model, def_u_p[j=1:q],  u_p[j]  == exp(-z_p[j]))
        @constraint(model, def_s_tr[i=1:n], s_tr[i] + s_tr[i] * u_tr[i] == 1)
        @constraint(model, def_s_p[j=1:q],  s_p[j]  + s_p[j]  * u_p[j]  == 1)
    end

    # ── softplus / t definition (switchable) ──────────────────────────────────
    if t_form == :softplus
        @variable(model, 0.0 <= t[1:n] <= z_bound + log(2))
        @constraint(model, def_t[i=1:n], t[i] == log(1 + exp(z_tr[i])))

    else # :exp_aux
        @variable(model, exp(-z_bound) <= w_tr[1:n] <= exp(z_bound))
        @variable(model, 1 + exp(-z_bound) <= v_tr[1:n] <= 1 + exp(z_bound))
        @variable(model, 0.0 <= t[1:n] <= z_bound + log(2))
        @constraint(model, def_w_tr[i=1:n], w_tr[i] == exp(z_tr[i]))
        @constraint(model, def_v_lin[i=1:n], v_tr[i] == 1 + w_tr[i])
        @constraint(model, def_v_exp[i=1:n], v_tr[i] == exp(t[i]))
    end

    # ── UPPER LEVEL: maximise training NLL (no regularisation). ───────────────
    @objective(model, Max, sum(t[i] - y_train[i] * z_tr[i] for i in 1:n))

    # ── LOWER LEVEL: KKT stationarity  ∇L + 2λβ = 0. ─────────────────────────
    #    The +2λβ_k term is LINEAR — it does not add any nonconvexity.
    @expression(model, grad_k[k=1:d],
        sum(X_train[i, k] * (s_tr[i] - y_train[i]) for i in 1:n) +
        sum(X_p[j, k]     * (s_p[j]  - y_poisoning[j]) for j in 1:q) +
        2 * l2_reg * beta[k])

    @constraint(model, kkt_up[k=1:d],  grad_k[k] <=  tol_kkt)
    @constraint(model, kkt_lo[k=1:d],  grad_k[k] >= -tol_kkt)

    # ── multi-start (Ipopt only) ──────────────────────────────────────────────
    if rng !== nothing
        b0 = randn(rng, d) .* 0.1
        Xp0 = zeros(q, d); Xp0[:, 1] .= 1.0
        for j in 1:q, k in 2:d
            Xp0[j, k] = rand(rng) * scale_factor
        end
        for k in 1:d; set_start_value(beta[k], b0[k]); end
        for j in 1:q, k in 2:d; set_start_value(X_p[j, k], Xp0[j, k]); end

        ztr0 = clamp.(X_train * b0, -z_bound, z_bound)
        zp0  = clamp.(Xp0 * b0,     -z_bound, z_bound)

        for i in 1:n
            set_start_value(z_tr[i], ztr0[i])
            set_start_value(s_tr[i], 1 / (1 + exp(-ztr0[i])))
            if sigmoid_form == :mult_aux
                set_start_value(u_tr[i], exp(-ztr0[i]))
            end
            if t_form == :softplus
                set_start_value(t[i], log(1 + exp(ztr0[i])))
            else
                set_start_value(w_tr[i], exp(ztr0[i]))
                set_start_value(v_tr[i], 1 + exp(ztr0[i]))
                set_start_value(t[i], log(1 + exp(ztr0[i])))
            end
        end
        for j in 1:q
            set_start_value(z_p[j], zp0[j])
            set_start_value(s_p[j], 1 / (1 + exp(-zp0[j])))
            if sigmoid_form == :mult_aux
                set_start_value(u_p[j], exp(-zp0[j]))
            end
        end
    end

    if solver == :SCIP
        with_log_file(log_file) do
            optimize!(model)
        end
    else
        optimize!(model)
    end

    r = _extract(model)

    beta_val = r.has_solution ? value.(beta) : fill(NaN, d)
    Xp_val   = r.has_solution ? value.(X_p)  : fill(NaN, q, d)
    z_val    = r.has_solution ? vcat(value.(z_tr), value.(z_p)) : [NaN]

    bound_active = if !r.has_solution
        missing
    else
        (beta = maximum(abs, beta_val) > 0.999 * beta_bound,
         z    = maximum(abs, z_val) > 0.999 * z_bound)
    end

    return (beta            = beta_val,
            X_p             = Xp_val,
            solve_time      = r.solve_time,
            status          = r.status,
            has_solution    = r.has_solution,
            objective       = r.objective,
            objective_bound = r.objective_bound,
            relative_gap    = r.relative_gap,
            beta_max_abs    = r.has_solution ? maximum(abs, beta_val) : NaN,
            z_max_abs       = r.has_solution ? maximum(abs, z_val)    : NaN,
            bound_active    = bound_active,
            sigmoid_form    = sigmoid_form,
            t_form          = t_form,
            l2_reg          = l2_reg,
            beta_bound_used = beta_bound,
            z_bound_used    = z_bound)
end


function PA_logistic_regression(X_train::Matrix{Float64}, y_train::Vector{Float64},
                                X_test::Matrix{Float64}, y_test::Vector{Float64},
                                y_poisoning::Vector{Float64};
                                solver::Symbol=:Ipopt,
                                scale_factor::Float64=1.0,
                                time_limit::Float64=300.0,
                                rng::Union{AbstractRNG,Nothing}=nothing,
                                beta_bound::Union{Float64,Nothing}=nothing,
                                z_bound::Union{Float64,Nothing}=nothing,
                                l2_reg::Float64,
                                sigmoid_form::Symbol=:mult_aux,
                                t_form::Symbol=:softplus,
                                tol_kkt::Float64=1e-4,
                                threads::Int=1,
                                silent::Bool=true,
                                log_file::Union{String,Nothing}=nothing)

    X_tr_bias = add_bias(X_train)
    X_te_bias = add_bias(X_test)

    res = PA_logistic_regression_model(X_tr_bias, y_train, y_poisoning;
              solver, scale_factor, time_limit, rng,
              beta_bound, z_bound, l2_reg, sigmoid_form, t_form, tol_kkt,
              threads, silent, log_file)

    accuracy = if res.has_solution
        prob = 1.0 ./ (1.0 .+ exp.(-(X_te_bias * res.beta)))
        sum(Float64.(prob .>= 0.5) .== y_test) / length(y_test)
    else
        NaN
    end

    return merge((accuracy = accuracy,), res)
end


# ==============================================================================
# 5. QUICK A/B HARNESS
# ==============================================================================

function benchmark_sigmoid_forms(X_train, y_train, X_test, y_test, y_poisoning;
                                 solver::Symbol=:Gurobi, time_limit::Float64=300.0,
                                 beta_bound::Union{Float64,Nothing}=nothing,
                                 z_bound::Union{Float64,Nothing}=nothing,
                                 l2_reg::Float64,
                                 threads::Int=1,
                                 log_dir::Union{String,Nothing}=nothing)
    results = Dict{Symbol,Any}()
    for form in (:division, :mult, :mult_aux)
        log_file = log_dir === nothing ? nothing : joinpath(log_dir, "sigmoid_$(form).log")
        results[form] = PA_logistic_regression(X_train, y_train, X_test, y_test, y_poisoning;
                             solver, time_limit, beta_bound, z_bound, l2_reg,
                             sigmoid_form=form, threads, silent=true, log_file)
    end
    return results
end