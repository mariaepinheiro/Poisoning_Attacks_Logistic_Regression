# ==============================================================================
# figures_phase2.jl — gap and dual-ratio figures for the global optimization
#   Reads  results/phase2/phase2_results.csv          (PA_G_0, CAR_0, CAR)
#          results/phase2_pagfull/phase2_pagfull.csv   (PA_G with all bounds)
#   Writes gap_grid.pdf and ratio_grid.pdf
#
#   Style: one color per dataset (same as Phase 1);
#          PA_G dashed + square, CAR solid + circle;
#          hollow marker = bounds of Prop. 3 only (_0), filled = all bounds.
# ==============================================================================
using CSV, DataFrames, Plots, Measures, LaTeXStrings
gr()

const ROOT = normpath(joinpath(@__DIR__, ".."))   # repository root
const SRC  = joinpath(ROOT, "src")
const DIR  = ROOT
mkpath(joinpath(DIR, "figures"))

# ==========================================
# DADOS
# ==========================================
r1 = CSV.read(joinpath(DIR, "results", "phase2", "phase2_results.csv"), DataFrame)
r2 = CSV.read(joinpath(DIR, "results", "phase2_pagfull", "phase2_pagfull.csv"), DataFrame)
r1.method = replace(r1.method, "PA_G" => "PA_G_0")           # Prop. 3 only
r2.method = replace(r2.method, "PA_G_full" => "PA_G")        # all bounds
cols = [:dataset, :q_ratio, :method, :primal, :dual]
df = vcat(select(r1, cols), select(r2, cols))

datasets    = ["biopsy", "Smarket", "fgl", "College", "OJ", "Banknote", "Ionosphere", "Spambase"]
names_show  = Dict("biopsy" => "Biopsy")
rhos        = [0.05, 0.10, 0.15, 0.20]
x_positions = 1:length(rhos)
x_labels    = ["5%", "10%", "15%", "20%"]
methods     = ["PA_G_0", "CAR_0", "PA_G", "CAR"]

num(x) = (ismissing(x) || (x isa Real && isnan(x))) ? NaN : Float64(x)

function getv(ds, rho, m, c)
    sub = df[(df.dataset .== ds) .& isapprox.(df.q_ratio, rho) .& (df.method .== m), :]
    @assert nrow(sub) == 1 "Esperada 1 linha para $ds, $rho, $m"
    return num(sub[1, c])
end

# gaps (%) of the four methods w.r.t. z_best, and the two dual ratios
function series(ds)
    gaps = Dict(m => Float64[] for m in methods)
    r0, r = Float64[], Float64[]
    for rho in rhos
        z_best = maximum(filter(!isnan, [getv(ds, rho, m, :primal) for m in methods]))
        for m in methods
            push!(gaps[m], 100 * (getv(ds, rho, m, :dual) - z_best) / abs(z_best))
        end
        push!(r0, getv(ds, rho, "PA_G_0", :dual) / getv(ds, rho, "CAR_0", :dual))
        push!(r,  getv(ds, rho, "PA_G",   :dual) / getv(ds, rho, "CAR",   :dual))
    end
    return gaps, r0, r
end

# ==========================================
# CONFIGURAÇÕES GLOBAIS (mesmas da Fase 1)
# ==========================================
default(
    linewidth = 2,
    markersize = 5,
    markerstrokewidth = 1.5,
    grid = true,
    gridalpha = 0.2,
    framestyle = :box,
    fontfamily = "Computer Modern",
    titlefontsize = 12,
    guidefontsize = 10,
    tickfontsize = 10,
    legend = false,
    margin = 4mm
)

base_colors = palette(:tab10)[1:length(datasets)]
color_map   = Dict(ds => base_colors[i] for (i, ds) in enumerate(datasets))

# line style and marker per method; `hollow` = bounds of Prop. 3 only
style = Dict(
    "PA_G_0" => (ls = :dash,  mk = :square, hollow = true),
    "PA_G"   => (ls = :dash,  mk = :square, hollow = false),
    "CAR_0"  => (ls = :solid, mk = :circle, hollow = true),
    "CAR"    => (ls = :solid, mk = :circle, hollow = false))

mcolor(c, hollow) = hollow ? :white : c

# ==========================================
# SUBGRÁFICOS
# ==========================================
gap_subplots, ratio_subplots = [], []

for ds in datasets
    c = color_map[ds]
    gaps, r0, r = series(ds)
    ttl = get(names_show, ds, ds)

    # --- Gap (log) ---
    p_gap = plot(title = ttl, xlabel = "ρ", ylabel = "gap (%, log scale)",
                 xticks = (x_positions, x_labels), yscale = :log10, ylims = (1, 1e4))
    for m in methods            # hollow drawn first, filled on top
        s = style[m]
        plot!(p_gap, x_positions, gaps[m]; color = c, linestyle = s.ls, marker = s.mk,
              markercolor = mcolor(c, s.hollow), markerstrokecolor = c)
    end
    push!(gap_subplots, p_gap)

    # --- Dual ratio (log) ---
    p_ratio = plot(title = ttl, xlabel = "ρ", ylabel = "dual ratio",
                   xticks = (x_positions, x_labels), yscale = :log10, ylims = (0.8, 100))
    hline!(p_ratio, [1.0]; color = :gray, linestyle = :dot, linewidth = 1, marker = :none)
    plot!(p_ratio, x_positions, r0; color = c, linestyle = :dash, marker = :diamond,
          markercolor = :white, markerstrokecolor = c)
    plot!(p_ratio, x_positions, r;  color = c, linestyle = :solid, marker = :diamond,
          markercolor = c, markerstrokecolor = c)
    push!(ratio_subplots, p_ratio)
end

# ==========================================
# PAINÉIS DE LEGENDA (desenhados à mão)
# ==========================================
function legend_panel(entries)
    p = plot(xlims = (0, 1), ylims = (0, 1), framestyle = :box, grid = false,
             ticks = false, title = " ")
    ys = range(0.80, 0.20; length = length(entries))
    for ((lab, ls, mk, hollow), y) in zip(entries, ys)
        plot!(p, [0.08, 0.38], [y, y]; color = :black, linestyle = ls, marker = :none)
        scatter!(p, [0.23], [y]; marker = mk, markersize = 6,
                 markercolor = hollow ? :white : :black, markerstrokecolor = :black)
        annotate!(p, 0.46, y, text(lab, 12, :left, "Computer Modern"))
    end
    return p
end

# If the LaTeX labels do not render, replace them by plain strings,
# e.g. "PA_G (Prop. 3)", "PA_G (all bounds)", "CAR (Prop. 3)", "CAR (all bounds)".
gap_legend = legend_panel([
    (L"\mathrm{PA\_G}_0", :dash,  :square, true),
    (L"\mathrm{PA\_G}",   :dash,  :square, false),
    (L"\mathrm{CAR}_0",   :solid, :circle, true),
    (L"\mathrm{CAR}",     :solid, :circle, false)])

ratio_legend = legend_panel([
    (L"r_0", :dash,  :diamond, true),
    (L"r",   :solid, :diamond, false)])

# ==========================================
# LAYOUTS (8 datasets + legenda = 3 x 3)
# ==========================================
plt_gap_grid = plot(gap_subplots..., gap_legend;
                    layout = (3, 3), size = (1000, 780),
                    plot_title = " ", plot_titlevspan = 0.03)

plt_ratio_grid = plot(ratio_subplots..., ratio_legend;
                      layout = (3, 3), size = (1000, 780),
                      plot_title = " ", plot_titlevspan = 0.03)

display(plt_gap_grid)
display(plt_ratio_grid)

# ==========================================
# SALVAR (PDF para LaTeX)
# ==========================================
savefig(plt_gap_grid, joinpath(DIR, "figures", "gap_grid.pdf"))
savefig(plt_gap_grid, joinpath(DIR, "figures", "gap_grid.png"))
savefig(plt_ratio_grid, joinpath(DIR, "figures", "ratio_grid.pdf"))
savefig(plt_ratio_grid, joinpath(DIR, "figures", "ratio_grid.png"))
