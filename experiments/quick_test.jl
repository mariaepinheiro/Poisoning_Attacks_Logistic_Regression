# ==============================================================================
# quick_test.jl — smoke test (≈ 5 min): fgl, ρ = 5 %, SCIP 120 s per method
# ==============================================================================
const ROOT = normpath(joinpath(@__DIR__, ".."))   # repository root
const SRC  = joinpath(ROOT, "src")
const DIR  = ROOT
include(joinpath(SRC, "models_global_regularization.jl"))
include(joinpath(SRC, "bounds.jl"))
include(joinpath(SRC, "models_bounded.jl"))
include(joinpath(SRC, "common.jl"))
include(joinpath(SRC, "datasets.jl"))
using Random, Printf

D = benchmarks["fgl"]
Xtr, ytr, _, _ = stratified_split(D.X, D.y, 0.30; rng = MersenneTwister(42))
Xs, _ = min_max_scale(Xtr, Xtr);  Xb = add_bias(Xs)
n, d = size(Xb);  q = poison_budget(n, 0.05);  yp = poison_labels(q);  λ = 1.0
println("fgl: n=$n, d=$d, q=$q   (expected from the first round: n=64, q=3)")

Bp = compute_bounds(Xb, ytr, yp, λ; mode = :prop3)
Bf = compute_bounds(Xb, ytr, yp, λ; mode = :full)
for (lab, B) in (("prop3", Bp), ("full", Bf))
    s = bounds_summary(B)
    @printf("  %-6s R=%7.3f  β width=%7.3f  θ width=%7.3f  z^p width=%7.3f  time=%5.1fs  clamped=%d\n",
            lab, s.R, s.beta_width, s.theta_width, s.zp_width, s.time, B.info["n_alpha_clamped"])
end

ref = PA_logistic_regression_model(Xb, ytr, yp; solver = :Ipopt, l2_reg = λ,
          rng = MersenneTwister(42001), sigmoid_form = :division, t_form = :softplus)
for (lab, B) in (("prop3", Bp), ("full", Bf))
    v = bounds_violation(B, Xb, ytr, ref.beta; Xp = ref.X_p)
    @printf("  Ipopt solution vs %-5s bounds: violation = %.1e  %s\n", lab, v, v < 1e-3 ? "OK" : "PROBLEM")
end
@printf("  Ipopt objective = %.4f   (first round: 33.37)\n", ref.objective)

r = solve_PAG(Xb, ytr, yp, λ, Bp; solver = :Ipopt, time_limit = 60.0, rng = MersenneTwister(42001))
@printf("  PA_L (Ipopt, new model) objective = %.4f  %s\n", r.objective, r.status)

for (lab, f) in (("PA_G",  () -> solve_PAG(Xb, ytr, yp, λ, Bp; time_limit = 120.0)),
                 ("CAR_0", () -> solve_CAR(Xb, ytr, yp, λ, Bp; time_limit = 120.0)),
                 ("CAR",   () -> solve_CAR(Xb, ytr, yp, λ, Bf; time_limit = 120.0)))
    r = f()
    @printf("  %-6s primal=%9.4f  dual=%10.4f  %s\n", lab, r.objective, r.objective_bound, r.status)
end
