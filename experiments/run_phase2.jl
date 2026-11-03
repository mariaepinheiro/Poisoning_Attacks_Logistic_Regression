# ==============================================================================
# run_phase2.jl — Phase 2: PA_G vs CAR_0 vs CAR  (SCIP, 3600 s, partition 1)
# ==============================================================================
#   Step 1 (parallel, per dataset × ρ):
#     bounds :prop3 (PA_G and CAR_0) and :full (CAR), timed and saved;
#     Ipopt multistart on P2 as a sanity check: its β must lie inside both
#     sets of bounds.
#   Step 2 (parallel): 3 SCIP runs per dataset × ρ.
#
#   Usage:  julia run_phase2.jl          (all files in the same folder)
# ==============================================================================

const SEED        = 42                    # = partition 1 of Phase 1
const TRAIN_RATIO = 0.30
const Q_RATIOS    = [0.05, 0.10, 0.15, 0.20]
const L2_REG      = 1.0
const TIME_LIMIT  = 3600.0
const DATASETS    = ["biopsy", "Smarket", "fgl", "College",
                     "OJ", "Banknote", "Ionosphere", "Spambase"]
const N_STARTS    = 5
const EPS_ALPHA   = 1e-8
const OBBT        = (theta = true, sub_time = 60.0, budget = 3600.0)
const N_WORKERS   = min(6, max(Sys.CPU_THREADS - 1, 1))

const ROOT = normpath(joinpath(@__DIR__, ".."))   # repository root
const SRC  = joinpath(ROOT, "src")
const DIR  = ROOT
const SPAMBASE = joinpath(DIR, "data", "spambase.data")
const OUT      = joinpath(DIR, "results", "phase2")
const LOGS     = joinpath(OUT, "solver_logs")

# ── workers ──────────────────────────────────────────────────────────────────
using Distributed
nworkers() == 1 && addprocs(N_WORKERS)
@everywhere begin
    include(joinpath($SRC, "models_global_regularization.jl"))
    include(joinpath($SRC, "bounds.jl"))
    include(joinpath($SRC, "models_bounded.jl"))
    include(joinpath($SRC, "common.jl"))
end

using CSV, DataFrames, Printf, Serialization, Random
include(joinpath(SRC, "datasets.jl"))          # YOUR datasets.jl
include(joinpath(SRC, "spambase_loader.jl"))
"Spambase" in DATASETS && load_spambase!(benchmarks, SPAMBASE)
include(joinpath(SRC, "common.jl"))
mkpath(LOGS)

# ── data (same split as before) ──────────────────────────────────────────────
splits = Dict{String,NamedTuple}()
for name in DATASETS
    haskey(benchmarks, name) || (@warn "Dataset $name not found — skipped"; continue)
    D = benchmarks[name]
    Xtr, ytr, Xte, yte = stratified_split(D.X, D.y, TRAIN_RATIO; rng = MersenneTwister(SEED))
    Xtr_s, Xte_s = min_max_scale(Xtr, Xte)
    splits[name] = (Xb = add_bias(Xtr_s), y = ytr, Xb_te = add_bias(Xte_s), y_te = yte,
                    n = size(Xtr_s, 1), d = size(Xtr_s, 2) + 1)
    @info "$name: n=$(splits[name].n), d=$(splits[name].d) (with intercept)"
end
names_ok = sort(collect(keys(splits)))

# ── Step 1 worker ────────────────────────────────────────────────────────────
@everywhere function prepare(name, Xb, y, ρ, q, yp, λ, nstarts, seed, ε, opts)
    Bp = compute_bounds(Xb, y, yp, λ; mode = :prop3, ε)
    Bf = compute_bounds(Xb, y, yp, λ; mode = :full, ε,
                        theta = opts.theta, sub_time = opts.sub_time, budget = opts.budget)
    best = nothing
    for s in 1:nstarts
        r = try
            PA_logistic_regression_model(Xb, y, yp; solver = :Ipopt, time_limit = 300.0,
                rng = MersenneTwister(seed * 1000 + s), l2_reg = λ,
                sigmoid_form = :division, t_form = :softplus, silent = true)
        catch
            nothing
        end
        r === nothing && continue
        if r.has_solution && (best === nothing || r.objective > best.objective)
            best = r
        end
    end
    vp = best === nothing ? NaN : bounds_violation(Bp, Xb, y, best.beta; Xp = best.X_p)
    vf = best === nothing ? NaN : bounds_violation(Bf, Xb, y, best.beta; Xp = best.X_p)
    return (name = name, ρ = ρ, q = q, Bp = Bp, Bf = Bf,
            z_ipopt = best === nothing ? NaN : best.objective,
            viol_prop3 = vp, viol_full = vf)
end

println("\n", "═"^90, "\n  STEP 1: bounds + sanity check\n", "═"^90)
tasks = [(n, ρ) for n in names_ok for ρ in Q_RATIOS]
t1 = time()
prep = pmap(tasks) do (n, ρ)
    S = splits[n];  q = poison_budget(S.n, ρ)
    prepare(n, S.Xb, S.y, ρ, q, poison_labels(q), L2_REG, N_STARTS, SEED, EPS_ALPHA, OBBT)
end
@info "Step 1 done in $(round((time() - t1) / 60, digits = 1)) min"

bnd = DataFrame()
for P in prep
    serialize(joinpath(OUT, "bounds_$(P.name)_q$(round(Int, 100P.ρ)).jls"), (Bp = P.Bp, Bf = P.Bf))
    sp, sf = bounds_summary(P.Bp), bounds_summary(P.Bf)
    push!(bnd, (dataset = P.name, q_ratio = P.ρ, q = P.q, R = sp.R,
        beta_w_prop3 = sp.beta_width,   beta_w_full = sf.beta_width,
        theta_w_prop3 = sp.theta_width, theta_w_full = sf.theta_width,
        zp_w_prop3 = sp.zp_width,       zp_w_full = sf.zp_width,
        alphap_w_prop3 = sp.alphap_width, alphap_w_full = sf.alphap_width,
        bounds_time_full = sf.time,
        n_fail_full = get(P.Bf.info, "n_fail", 0),
        budget_hit = get(P.Bf.info, "budget_hit", ""),
        alpha_clamped_prop3 = P.Bp.info["n_alpha_clamped"],
        alpha_clamped_full  = P.Bf.info["n_alpha_clamped"],
        z_ipopt = P.z_ipopt, viol_prop3 = P.viol_prop3, viol_full = P.viol_full))
    for (lab, v) in (("prop3", P.viol_prop3), ("full", P.viol_full))
        !isnan(v) && v > 1e-3 &&
            @warn "Ipopt solution OUTSIDE the $lab bounds for $(P.name), ρ=$(P.ρ): violation $(round(v, sigdigits = 3))"
    end
end
CSV.write(joinpath(OUT, "bounds_summary.csv"), bnd)

@printf("\n  %-11s %4s %4s %8s  %9s %9s  %9s %9s  %7s  %7s %8s\n", "Dataset", "ρ%", "q", "R",
        "β_w P3", "β_w full", "zp_w P3", "zp_w full", "t_full", "clamp", "viol")
for r in eachrow(bnd)
    @printf("  %-11s %4.0f %4d %8.2f  %9.3f %9.3f  %9.2f %9.2f  %6.0fs  %7d %8.1e\n",
            r.dataset, 100r.q_ratio, r.q, r.R, r.beta_w_prop3, r.beta_w_full,
            r.zp_w_prop3, r.zp_w_full, r.bounds_time_full, r.alpha_clamped_full, r.viol_full)
end

# ── Step 2: SCIP runs ────────────────────────────────────────────────────────
@everywhere struct Job
    name::String;  Xb::Matrix{Float64};  y::Vector{Float64}
    n::Int;  d::Int;  ρ::Float64;  q::Int;  yp::Vector{Float64}
    method::String;  B::ModelBounds;  λ::Float64;  tl::Float64;  log::String
end

@everywhere function run_job(J::Job)
    r = try
        startswith(J.method, "PA_G") ?
            solve_PAG(J.Xb, J.y, J.yp, J.λ, J.B; time_limit = J.tl, log_file = J.log) :
            solve_CAR(J.Xb, J.y, J.yp, J.λ, J.B; time_limit = J.tl, log_file = J.log)
    catch e
        (objective = NaN, objective_bound = NaN, relative_gap = NaN,
         solve_time = NaN, status = "ERROR: " * sprint(showerror, e), has_solution = false)
    end
    num(x) = x isa Number ? Float64(x) : NaN
    return Dict{String,Any}("dataset" => J.name, "n" => J.n, "d" => J.d,
        "q_ratio" => J.ρ, "q" => J.q, "method" => J.method,
        "primal" => num(r.objective), "dual" => num(r.objective_bound),
        "scip_gap" => num(r.relative_gap), "solve_time" => num(r.solve_time),
        "status" => string(r.status), "bounds_time" => J.B.time)
end

jobs = Job[]
for P in prep
    S = splits[P.name];  yp = poison_labels(P.q)
    for (meth, B) in (("PA_G", P.Bp), ("CAR_0", P.Bp), ("CAR", P.Bf))
        lf = joinpath(LOGS, "$(P.name)_q$(round(Int, 100P.ρ))_$(meth).log")
        push!(jobs, Job(P.name, S.Xb, S.y, S.n, S.d, P.ρ, P.q, yp, meth, B, L2_REG, TIME_LIMIT, lf))
    end
end

nj = length(jobs)
println("\n", "═"^90)
@printf("  STEP 2: %d SCIP runs on %d workers → at most ≈ %.0f h\n",
        nj, nworkers(), ceil(nj / nworkers()) * TIME_LIMIT / 3600)
println("═"^90)

ch = RemoteChannel(() -> Channel{Dict}(nj))
@everywhere run_tracked(J, ch) = (r = run_job(J); put!(ch, r); r)
t2 = time()
mon = @async for c in 1:nj
    r = take!(ch)
    println(@sprintf("  [%2d/%2d %6.0fs] %-11s ρ=%2.0f%% %-5s primal=%10.3f dual=%10.3f %s",
            c, nj, time() - t2, r["dataset"], 100r["q_ratio"], r["method"],
            r["primal"], r["dual"], r["status"]))
end
raw = pmap(J -> run_tracked(J, ch), jobs; on_error = identity)
wait(mon)

serialize(joinpath(OUT, "raw_phase2.jls"), raw)
good = [r for r in raw if !(r isa Exception)]
foreach(r -> @warn("Job failed: $r"), filter(r -> r isa Exception, raw))
res = DataFrame([c => [r[c] for r in good] for c in keys(good[1])])
CSV.write(joinpath(OUT, "phase2_results.csv"), res)
@info "Done. Results in $OUT.  Next: julia tables_phase2.jl"
