# ==============================================================================
# run_phase1.jl — Phase 1: LR and PA_L on all datasets
#   PA_L = problem (P2) exactly as written, bounds of Proposition 3,
#          Ipopt from 5 random starts, 300 s each; 5 partitions (seeds 42–46).
#
#   Each instance is saved to results/phase1/jobs/ as soon as it finishes;
#   re-running skips finished instances.  At the end, everything is collected
#   into results/phase1/phase1_results.csv.
#
#   julia run_phase1.jl            run what is missing, then collect
#   julia run_phase1.jl collect    only collect
# ==============================================================================

const PARTITION_SEEDS = [42, 43, 44, 45, 46]
const N_STARTS    = 5
const TRAIN_RATIO = 0.30
const Q_RATIOS    = [0.05, 0.10, 0.15, 0.20]
const L2_REG      = 1.0
const TLIM        = 300.0      # (not TIME_LIMIT: that name is exported by JuMP)
const DATASETS    = ["biopsy", "Smarket", "fgl", "College",
                     "OJ", "Banknote", "Ionosphere", "Spambase"]
const N_WORKERS   = min(6, max(Sys.CPU_THREADS - 1, 1))

const ROOT = normpath(joinpath(@__DIR__, ".."))   # repository root
const SRC  = joinpath(ROOT, "src")
const DIR  = ROOT
const SPAMBASE = joinpath(DIR, "data", "spambase.data")
const OUT      = joinpath(DIR, "results", "phase1")
const JOBS     = joinpath(OUT, "jobs")
mkpath(JOBS)

using Serialization, CSV, DataFrames, Printf, Random
tag(name, pid, ρ) = "$(name)_p$(pid)_q$(round(Int, 100ρ))"

function collect_results()
    files = filter(f -> endswith(f, ".jls"), readdir(JOBS))
    isempty(files) && (println("  No finished instances yet."); return)
    df = DataFrame([deserialize(joinpath(JOBS, f)) for f in files])
    df.drop_ac  = df.ac_lr  .- df.ac_pal
    df.drop_mcc = df.mcc_lr .- df.mcc_pal
    sort!(df, [:dataset, :q_ratio, :partition])
    CSV.write(joinpath(OUT, "phase1_results.csv"), df)
    println("  Collected $(nrow(df)) / $(length(DATASETS) * 5 * length(Q_RATIOS)) instances → ",
            joinpath(OUT, "phase1_results.csv"))
    println("  Starts that converged: $(sum(df.starts_ok)) / $(N_STARTS * nrow(df))")
end

if "collect" in ARGS
    collect_results();  exit()
end

using Distributed
nworkers() == 1 && addprocs(N_WORKERS)
@everywhere begin
    include(joinpath($SRC, "models_global_regularization.jl"))
    include(joinpath($SRC, "bounds.jl"))
    include(joinpath($SRC, "models_bounded.jl"))
    include(joinpath($SRC, "common.jl"))
    using Serialization
end
include(joinpath(SRC, "models_global_regularization.jl"))
include(joinpath(SRC, "common.jl"))
include(joinpath(SRC, "datasets.jl"))            # YOUR datasets.jl
include(joinpath(SRC, "spambase_loader.jl"))
"Spambase" in DATASETS && load_spambase!(benchmarks, SPAMBASE)

@everywhere function one_instance(t, name, pid, pseed, Xtr, ytr, Xte, yte, ρ,
                                  λ, nstarts, tl, jobs_dir)
    Xb = add_bias(Xtr);  Xb_te = add_bias(Xte)
    lr = logistic_regression_model(Xb, ytr; solver = :Ipopt, l2_reg = λ)
    m_lr = metrics(lr.beta, Xb_te, yte)

    q = poison_budget(size(Xb, 1), ρ);  yp = poison_labels(q)
    B = compute_bounds(Xb, ytr, yp, λ; mode = :prop3)
    best = nothing;  t_total = 0.0;  n_ok = 0;  n_err = 0;  first_err = ""
    for s in 1:nstarts
        r = try
            solve_PAG(Xb, ytr, yp, λ, B; solver = :Ipopt, time_limit = tl,
                      rng = MersenneTwister(pseed * 1000 + s))
        catch e
            n_err += 1
            first_err == "" && (first_err = sprint(showerror, e))
            nothing
        end
        r === nothing && continue
        t_total += isnan(r.solve_time) ? 0.0 : r.solve_time
        if r.has_solution && string(r.status) in ("LOCALLY_SOLVED", "ALMOST_LOCALLY_SOLVED")
            n_ok += 1
            (best === nothing || r.objective > best.objective) && (best = r)
        end
    end
    m_pa = best === nothing ? (accuracy = NaN, mcc = NaN) : metrics(best.beta, Xb_te, yte)
    row = (dataset = name, partition = pid, seed = pseed, n = size(Xb, 1), d = size(Xb, 2),
           q_ratio = ρ, q = q,
           ac_lr = m_lr.accuracy, mcc_lr = m_lr.mcc, time_lr = lr.solve_time,
           ac_pal = m_pa.accuracy, mcc_pal = m_pa.mcc,
           obj_pal = best === nothing ? NaN : best.objective,
           time_pal = t_total, starts_ok = n_ok, n_err = n_err, first_err = first_err)
    n_err < nstarts && serialize(joinpath(jobs_dir, "$t.jls"), row)   # not if every start crashed
    return row
end

# ── tasks still to do (largest first) ────────────────────────────────────────
tasks = []
for name in DATASETS
    haskey(benchmarks, name) || (@warn "Dataset $name not found — skipped"; continue)
    D = benchmarks[name]
    for (pid, ps) in enumerate(PARTITION_SEEDS)
        Xtr, ytr, Xte, yte = stratified_split(D.X, D.y, TRAIN_RATIO; rng = MersenneTwister(ps))
        Xtr_s, Xte_s = min_max_scale(Xtr, Xte)
        for ρ in Q_RATIOS
            t = tag(name, pid, ρ)
            isfile(joinpath(JOBS, "$t.jls")) && continue
            push!(tasks, (t, name, pid, ps, Xtr_s, ytr, Xte_s, yte, ρ))
        end
    end
end
sort!(tasks, by = x -> -size(x[5], 1) * size(x[5], 2) * x[9])
println("  Instances to do: $(length(tasks)) × $N_STARTS starts on $(nworkers()) workers")

λ_, ns_, tl_, jd_ = L2_REG, N_STARTS, TLIM, JOBS            # plain values for the workers
ch = RemoteChannel(() -> Channel{Any}(max(1, length(tasks))))
t0 = time()
mon = @async for c in 1:length(tasks)
    r = take!(ch)
    msg = r.n_err > 0 ? "   ERROR ($(r.n_err)×): $(first(r.first_err, 120))" : ""
    println(@sprintf("  [%3d/%3d %6.0fs] %-11s p%d ρ=%2.0f%%  AC %.3f → %.3f  ok=%d  %.0fs%s",
            c, length(tasks), time() - t0, r.dataset, r.partition, 100r.q_ratio,
            r.ac_lr, r.ac_pal, r.starts_ok, r.time_pal, msg))
end
pmap(tasks; on_error = identity) do t
    r = one_instance(t..., λ_, ns_, tl_, jd_)
    put!(ch, r);  r
end
wait(mon)
collect_results()
