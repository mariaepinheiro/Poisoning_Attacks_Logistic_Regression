# ==============================================================================
# tables_paper.jl — the two tables of the numerical section
#   tables/tab_attack_drops.tex   Phase 1: ΔAC, ΔMCC, times (mean ± std)
#   tables/tab_obj_gap.tex        Phase 2: primal, dual, gap of the 4 methods
#   Usage:  julia --project=. analysis/tables_paper.jl
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
num(x) = (ismissing(x) || (x isa Real && isnan(x))) ? NaN : Float64(x)
pm(v)  = @sprintf("\$%.3f \\pm %.3f\$", mean(v), std(v))

# ── Phase 1 ──────────────────────────────────────────────────────────────────
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
        # LR time: mean of the runs of each partition, then mean ± std over partitions
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
r1.method = replace(r1.method, "PA_G" => "PA_G_0")          # Prop. 3 only
r2.method = replace(r2.method, "PA_G_full" => "PA_G")       # all bounds
cols = [:dataset, :q_ratio, :method, :primal, :dual]
p2 = vcat(select(r1, cols), select(r2, cols))
p2.q_ratio = round.(p2.q_ratio, digits = 2)
p2.primal = num.(p2.primal);  p2.dual = num.(p2.dual)

const METHODS = ["PA_G_0", "CAR_0", "PA_G", "CAR"]
const MNAME   = Dict("PA_G_0" => "PA\\_G\$_0\$", "CAR_0" => "CAR\$_0\$",
                     "PA_G" => "PA\\_G", "CAR" => "CAR")
getv(ds, ρ, m, c) = p2[(p2.dataset .== ds) .& (p2.q_ratio .== ρ) .& (p2.method .== m), c][1]
zbest(ds, ρ) = maximum(filter(!isnan, [getv(ds, ρ, m, :primal) for m in METHODS]))
gapv(ds, ρ, m) = round(100 * (getv(ds, ρ, m, :dual) - zbest(ds, ρ)) / abs(zbest(ds, ρ)), digits = 1)

open(joinpath(DIR, "tables", "tab_obj_gap.tex"), "w") do io
    println(io, raw"""\begin{table}[p]
\centering
\caption{Global optimization: primal bounds $\underline{z}_f$, dual bounds
  $\overline{z}_f$, and gaps~\eqref{eq:gap} (in \%) obtained by \texttt{SCIP}
  within the time limit of 3600\,s.  PA\_G$_0$ and CAR$_0$ use only the bounds
  of Proposition~\ref{prop:beta-norm-bound}; PA\_G and CAR use all bounds of
  Section~\ref{subsec:valid-bounds}.  Gaps are computed with respect to
  $z_{\mathrm{best}}$, the largest primal bound among the methods that found a
  feasible solution; the smallest gap of each instance is shown in bold.
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
                        v = gapv(ds, ρ, m);  best = minimum(gapv(ds, ρ, mm) for mm in METHODS)
                        push!(cells, v == best ? @sprintf("\\textbf{%.1f}", v) : @sprintf("%.1f", v))
                    end
                end
                c1 = j == 1 ? "\\multirow{4}{*}{$lab}" : ""
                println(io, "$c1 & $(MNAME[m]) & ", join(cells, " & "), " \\\\")
            end
            kind != :g && println(io, "\\addlinespace[2pt]")
        end
    end
    println(io, raw"""\bottomrule
\end{tabular}
\end{adjustbox}
\end{table}""")
end



# ── Bound widths ─────────────────────────────────────────────────────────────
bw = CSV.read(joinpath(DIR, "results", "phase2", "bounds_summary.csv"), DataFrame)

open(joinpath(DIR, "tables", "tab_bound_widths.tex"), "w") do io
    println(io, raw"""\begin{table}[t]
\centering
\footnotesize
\caption{Average width of the bounds obtained from
  Proposition~\ref{prop:beta-norm-bound} only (Prop.~3) and from all bounds of
  Section~\ref{subsec:valid-bounds} (All), for the components of~$\beta$, the
  margins~$\theta^{tr}$ of the training points, the inner products~$z^p$ of the
  poisoning points, and the variables~$\alpha^p$.  Mean over the four
  budgets~$\rho$.}
\label{tab:bound_widths}
\begin{tabular}{@{}l rr rr rr rr@{}}
\toprule
 & \multicolumn{2}{c}{$\beta$} & \multicolumn{2}{c}{$\theta^{tr}$}
 & \multicolumn{2}{c}{$z^p$} & \multicolumn{2}{c}{$\alpha^p$} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9}
Dataset & Prop.~3 & All & Prop.~3 & All & Prop.~3 & All & Prop.~3 & All \\
\midrule""")
    for ds in ORDER
        S = bw[bw.dataset .== ds, :]
        m(c) = mean(S[!, c])
        @printf(io, "%s & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f & %.3f & %.3f \\\\\n",
                show_name(ds), m(:beta_w_prop3), m(:beta_w_full),
                m(:theta_w_prop3), m(:theta_w_full), m(:zp_w_prop3), m(:zp_w_full),
                m(:alphap_w_prop3), m(:alphap_w_full))
    end
    println(io, raw"""\bottomrule
\end{tabular}
\end{table}""")
end

println("Written: tables/tab_attack_drops.tex, tables/tab_obj_gap.tex, tables/tab_bound_widths.tex")
