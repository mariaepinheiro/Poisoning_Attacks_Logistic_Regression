# ==============================================================================
# tables_paper.jl — the two tables of the numerical section
#   tables/tab_attack_drops.tex   Phase 1: ΔAC, ΔMCC, times (mean ± std)
#   tables/tab_obj_gap.tex        Phase 2: primal, dual, Ū and gap of the 4 methods
#   Usage:  julia --project=. analysis/tables_paper.jl
#
#   Phase 2: gaps use min{dual, Ū} with the analytic bound of (36),
#            Ū = (n + q) log 2 − q φ(−R√d), where n, d, q are read from the
#            result files and R from bounds_summary.csv.  All values are taken
#            at full precision from the CSV files; rounding is for display only.
# ==============================================================================
using CSV, DataFrames, Printf, Statistics

const ROOT = normpath(joinpath(@__DIR__, ".."))   # repository root
const SRC  = joinpath(ROOT, "src")
const DIR  = ROOT
mkpath(joinpath(DIR, "tables"))

const ORDER   = ["biopsy", "Smarket", "fgl", "College", "OJ", "Banknote", "Ionosphere", "Spambase"]
const DISPLAY = Dict("biopsy" => "Biopsy")
const RHOS    = [0.05, 0.10, 0.15, 0.20]
show_name(ds) = get(DISPLAY, ds, ds)
pct(ρ) = round(Int, 100ρ)
num(x) = (ismissing(x) || (x isa Real && (isnan(x) || abs(x) > 1e90))) ? NaN : Float64(x)
pm(v)  = @sprintf("\$%.3f \\pm %.3f\$", mean(v), std(v))

# ── Phase 1 (unchanged) ──────────────────────────────────────────────────────
p1 = CSV.read(joinpath(DIR, "results", "phase1", "phase1_results.csv"), DataFrame)
p1.q_ratio = round.(p1.q_ratio, digits = 2)

open(joinpath(DIR, "tables", "tab_attack_drops.tex"), "w") do io
    println(io, raw"""\begin{table}[t]
\centering
\footnotesize
\caption{Attack effectiveness: drops $\Delta\AC$ and $\Delta\MCC$ induced by
  PA\_L, computed per partition, and solve times in seconds; PA\_L times include
  all 5~starts.  All values: mean~$\pm$~standard deviation over 5~partitions.}
\label{tab:attack_drops}
\begin{tabular}{@{}l r c rrr@{}}
\toprule
& LR & & \multicolumn{3}{c}{PA\_L} \\
\cmidrule(lr){2-2} \cmidrule(lr){4-6}
Dataset & Time (s) & $\rho$ & $\Delta\AC$ & $\Delta\MCC$ & Time (s) \\
\midrule""")
    for (k, ds) in enumerate(ORDER)
        S = p1[p1.dataset .== ds, :]
        lr = [mean(S[S.partition .== p, :time_lr]) for p in sort(unique(S.partition))]
        for (i, ρ) in enumerate(RHOS)
            T = S[S.q_ratio .== ρ, :]
            c1 = i == 1 ? "\\multirow{4}{*}{$(show_name(ds))}" : ""
            c2 = i == 1 ? "\\multirow{4}{*}{$(pm(lr))}" : ""
            println(io, "$c1 & $c2 & $(pct(ρ))\\% & $(pm(T.drop_ac)) & $(pm(T.drop_mcc)) & $(pm(T.time_pal)) \\\\")
        end
        println(io, k < length(ORDER) ? "\\addlinespace" : "\\bottomrule")
    end
    println(io, raw"""\end{tabular}
\end{table}""")
end

# ── Phase 2 ──────────────────────────────────────────────────────────────────
r1 = CSV.read(joinpath(DIR, "results", "phase2", "phase2_results.csv"), DataFrame)
r2 = CSV.read(joinpath(DIR, "results", "phase2_pagfull", "phase2_pagfull.csv"), DataFrame)
bnd = CSV.read(joinpath(DIR, "results", "phase2", "bounds_summary.csv"), DataFrame)
r1.method = replace(r1.method, "PA_G" => "PA_G_0")          # Prop. 3 only
r2.method = replace(r2.method, "PA_G_full" => "PA_G")       # all bounds
cols = [:dataset, :q_ratio, :method, :n, :d, :q, :primal, :dual]
p2 = vcat(select(r1, cols), select(r2, cols))
p2.q_ratio  = round.(p2.q_ratio, digits = 2)
bnd.q_ratio = round.(bnd.q_ratio, digits = 2)
p2.primal = num.(p2.primal);  p2.dual = num.(p2.dual)

const METHODS = ["PA_G_0", "CAR_0", "PA_G", "CAR"]
const MNAME   = Dict("PA_G_0" => "PA\\_G\$_0\$", "CAR_0" => "CAR\$_0\$",
                     "PA_G" => "PA\\_G", "CAR" => "CAR")
function getv(ds, ρ, m, c)
    v = p2[(p2.dataset .== ds) .& (p2.q_ratio .== ρ) .& (p2.method .== m), c]
    @assert length(v) == 1 "Esperada 1 linha para $ds, $ρ, $m"
    return v[1]
end

# analytic bound of (36): Ū = (n + q) log 2 − q φ(−R√d), with n, d, q of the run
φ(z) = z > 0 ? z + log1p(exp(-z)) : log1p(exp(z))
function Ubar(ds, ρ)
    R = bnd[(bnd.dataset .== ds) .& (bnd.q_ratio .== ρ), :R]
    @assert length(R) == 1 "Esperada 1 linha em bounds_summary para $ds, $ρ"
    n, d, q = getv(ds, ρ, "CAR", :n), getv(ds, ρ, "CAR", :d), getv(ds, ρ, "CAR", :q)
    return (n + q) * log(2) - q * φ(-R[1] * sqrt(d))
end

zbest(ds, ρ)   = maximum(filter(!isnan, [getv(ds, ρ, m, :primal) for m in METHODS]))
dualcap(ds, ρ, m) = min(getv(ds, ρ, m, :dual), Ubar(ds, ρ))
gapv(ds, ρ, m) = 100 * (dualcap(ds, ρ, m) - zbest(ds, ρ)) / abs(zbest(ds, ρ))   # full precision
disp(v) = round(v, digits = 1)                                                  # display only

open(joinpath(DIR, "tables", "tab_obj_gap.tex"), "w") do io
    println(io, raw"""\begin{table}[p]
\centering
\caption{Primal bounds $\underline{z}_f$, dual bounds $\overline{z}_f$
  (\texttt{SCIP}, time limit 3600\,s), analytic bound~$\bar U$, and
  gaps~\eqref{eq:gap} in \%. Smallest gap of each instance in bold;
  ``--'' = no feasible solution found.}
\label{tab:obj_gap}
\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{adjustbox}{max width=\textwidth}
\begin{tabular}{@{}ll rrrrrrrr@{}}
\toprule""")
    println(io, " & & ", join(show_name.(ORDER), " & "), " \\\\")
    for ρ in RHOS
        println(io, "\\midrule")
        println(io, "\\multicolumn{10}{@{}l}{\$\\rho = $(pct(ρ))\\%\$} \\\\")
        println(io, "\\cmidrule(r){1-2}")
        for (lab, kind) in (("\$\\underline{z}_f\$", :p), ("\$\\overline{z}_f\$", :d), ("gap (\\%)", :g))
            for (j, m) in enumerate(METHODS)
                cells = String[]
                for ds in ORDER
                    if kind == :p
                        v = getv(ds, ρ, m, :primal)
                        push!(cells, isnan(v) ? "--" : @sprintf("%.2f", v))
                    elseif kind == :d
                        push!(cells, @sprintf("%.1f", getv(ds, ρ, m, :dual)))
                    else
                        v = disp(gapv(ds, ρ, m))
                        best = minimum(disp(gapv(ds, ρ, mm)) for mm in METHODS)
                        push!(cells, v == best ? @sprintf("\\textbf{%.1f}", v) : @sprintf("%.1f", v))
                    end
                end
                c1 = j == 1 ? "\\multirow{4}{*}{$lab}" : ""
                println(io, "$c1 & $(MNAME[m]) & ", join(cells, " & "), " \\\\")
            end
            if kind == :d                                   # row with Ū after the duals
                println(io, "\\addlinespace[2pt]")
                println(io, " & \$\\bar U\$ & ",
                        join([@sprintf("%.1f", Ubar(ds, ρ)) for ds in ORDER], " & "), " \\\\")
            end
            kind != :g && println(io, "\\addlinespace[2pt]")
        end
    end
    println(io, raw"""\bottomrule
\end{tabular}
\end{adjustbox}
\end{table}""")
end
# ── Bound widths and bound computation times ─────────────────────────────────
bw = CSV.read(joinpath(DIR, "results", "phase2", "bounds_summary.csv"), DataFrame)

open(joinpath(DIR, "tables", "tab_bound_widths.tex"), "w") do io
    println(io, raw"""\begin{table}[t]
\centering
\footnotesize
\caption{Average bound widths with the bounds of
  Proposition~\ref{prop:beta-norm-bound} only (Prop.~3) and with all bounds
  of Section~\ref{subsec:valid-bounds} (All), and time to compute all
  bounds.  Mean over the four budgets~$\rho$.}
\label{tab:bound_widths}
\begin{tabular}{@{}l rr rr rr rr r@{}}
\toprule
 & \multicolumn{2}{c}{$\beta$} & \multicolumn{2}{c}{$\theta^{tr}$}
 & \multicolumn{2}{c}{$z^p$} & \multicolumn{2}{c}{$\alpha^p$} & Time \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9}
Dataset & Prop.~3 & All & Prop.~3 & All & Prop.~3 & All & Prop.~3 & All & (s) \\
\midrule""")
    for ds in ORDER
        S = bw[bw.dataset .== ds, :]
        m(c) = mean(S[!, c])
        @printf(io, "%s & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f & %.3f & %.3f & %.2f \\\\\n",
                show_name(ds), m(:beta_w_prop3), m(:beta_w_full),
                m(:theta_w_prop3), m(:theta_w_full), m(:zp_w_prop3), m(:zp_w_full),
                m(:alphap_w_prop3), m(:alphap_w_full), m(:bounds_time_full))
    end
    println(io, raw"""\bottomrule
\end{tabular}
\end{table}""")
end

println("Written: tables/tab_attack_drops.tex, tables/tab_obj_gap.tex, tables/tab_bound_widths.tex")