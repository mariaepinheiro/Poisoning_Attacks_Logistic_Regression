# ==============================================================================
# run_phase2_pagfull.jl — PA_G with the FULL bounds (the ones CAR used)
# ==============================================================================
#   - Loads the bounds saved by run_phase2.jl (results/phase2/bounds_*.jls),
#     so PA_G gets exactly the same bounds as CAR.  Nothing is recomputed.
#   - Each run is saved to results/phase2_pagfull/jobs/<tag>.jls as soon as it
#     finishes.  Re-running the script skips the runs that already exist.
#   - At the end, all finished runs are collected into phase2_pagfull.csv.
#
#   julia run_phase2_pagfull.jl            run what is missing, then collect
#   julia run_phase2_pagfull.jl collect    only collect the finished runs
#
#   Stopping: press Ctrl+C right AFTER a batch finishes (a progress line
#   appears roughly every hour); runs still in progress are lost.
# ==============================================================================

const SEED        = 42
const TRAIN_RATIO = 0.30
const Q_RATIOS    = [0.05, 0.10, 0.15, 0.20]
const L2_REG      = 1.0
const TLIM        = 3600.0     # (not TIME_LIMIT: that name is exported by JuMP)
const DATASETS    = ["biopsy", "Smarket", "fgl", "College",
                     "OJ", "Banknote", "Ionosphere", "Spambase"]
const N_WORKERS   = min(6, max(Sys.CPU_THREADS - 1, 1))

const ROOT = normpath(joinpath(@__DIR__, ".."))   # repository root
const SRC  = joinpath(ROOT, "src")
const DIR  = ROOT
const SPAMBASE = joinpath(DIR, "data", "spambase.data")
const BND_DIR  = joinpath(DIR, "results", "phase2")           # bounds of the CAR run
const OUT      = joinpath(DIR, "results", "phase2_pagfull")
const JOBS     = joinpath(OUT, "jobs")
const LOGS     = joinpath(OUT, "solver_logs")
mkpath(JOBS);  mkpath(LOGS)

using Serialization, CSV, DataFrames, Printf, Random
tag(name, ρ) = "$(name)_q$(round(Int, 100ρ))"

function collect_results()
    files = filter(f -> endswith(f, ".jls"), readdir(JOBS))
    isempty(files) && (println("  No finished runs yet."); return)
    rows = [deserialize(joinpath(JOBS, f)) for f in files]
    df = DataFrame([c => [r[c] for r in rows] for c in keys(rows[1])])
    sort!(df, [:dataset, :q_ratio])
    CSV.write(joinpath(OUT, "phase2_pagfull.csv"), df)
    println("  Collected $(nrow(df)) / $(length(DATASETS) * length(Q_RATIOS)) runs → ",
            joinpath(OUT, "phase2_pagfull.csv"))
end

if "collect" in ARGS
    collect_results();  exit()
end

# ── workers ──────────────────────────────────────────────────────────────────
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
include(joinpath(SRC, "bounds.jl"))                  # ModelBounds, for deserialize
include(joinpath(SRC, "common.jl"))
include(joinpath(SRC, "datasets.jl"))
include(joinpath(SRC, "spambase_loader.jl"))
"Spambase" in DATASETS && load_spambase!(benchmarks, SPAMBASE)

# ── jobs still to do ─────────────────────────────────────────────────────────
jobs = []
for name in DATASETS
    haskey(benchmarks, name) || (@warn "Dataset $name not found — skipped"; continue)
    D = benchmarks[name]
    Xtr, ytr, _, _ = stratified_split(D.X, D.y, TRAIN_RATIO; rng = MersenneTwister(SEED))
    Xs, _ = min_max_scale(Xtr, Xtr);  Xb = add_bias(Xs);  n, d = size(Xb)
    for ρ in Q_RATIOS
        t = tag(name, ρ)
        isfile(joinpath(JOBS, "$t.jls")) && continue                 # already done
        bf = joinpath(BND_DIR, "bounds_$t.jls")
        isfile(bf) || (@warn "Bounds file missing: $bf — skipped"; continue)
        Bf = deserialize(bf).Bf                                      # same bounds as CAR
        q  = poison_budget(n, ρ)
        push!(jobs, (t, name, Xb, ytr, n, d, ρ, q, poison_labels(q), Bf))
    end
end
println("  Runs to do: $(length(jobs))   (≈ $(ceil(Int, length(jobs) / nworkers())) h with $(nworkers()) workers)")

@everywhere function run_one(t, name, Xb, y, n, d, ρ, q, yp, B, λ, tl, jobs_dir, logs_dir)
    r = try
        solve_PAG(Xb, y, yp, λ, B; time_limit = tl, log_file = joinpath(logs_dir, "$t.log"))
    catch e
        (objective = NaN, objective_bound = NaN, relative_gap = NaN,
         solve_time = NaN, status = "ERROR: " * sprint(showerror, e), has_solution = false)
    end
    num(x) = x isa Number ? Float64(x) : NaN
    row = Dict{String,Any}("dataset" => name, "n" => n, "d" => d, "q_ratio" => ρ, "q" => q,
        "method" => "PA_G_full", "primal" => num(r.objective), "dual" => num(r.objective_bound),
        "scip_gap" => num(r.relative_gap), "solve_time" => num(r.solve_time),
        "status" => string(r.status))
    startswith(row["status"], "ERROR") ||
        serialize(joinpath(jobs_dir, "$t.jls"), row)                 # saved immediately (not errors)
    return row
end

t0 = time()
ch = RemoteChannel(() -> Channel{Dict}(max(1, length(jobs))))
mon = @async for c in 1:length(jobs)
    r = take!(ch)
    println(@sprintf("  [%2d/%2d %6.0fs] %-11s ρ=%2.0f%%  primal=%10.3f  dual=%10.3f  %s",
            c, length(jobs), time() - t0, r["dataset"], 100r["q_ratio"],
            r["primal"], r["dual"], r["status"]))
end
λ_, tl_, jd_, ld_ = L2_REG, TLIM, JOBS, LOGS        # plain values, sent to the workers
pmap(jobs; on_error = identity) do j
    r = run_one(j..., λ_, tl_, jd_, ld_)
    put!(ch, r);  r
end
wait(mon)
collect_results()
