# ==============================================================================
# figures_phase2.jl — dual-ratio figure for the global optimization
#   Reads  results/phase2/phase2_results.csv          (PA_G_0, CAR_0, CAR)
#          results/phase2_pagfull/phase2_pagfull.csv   (PA_G with all bounds)
#          results/phase2/bounds_summary.csv           (R, for the bound Ū)
#   Writes ratio_grid.pdf (and .png)
#
#   Dual bounds are replaced by min{dual, Ū}, with the analytic bound
#   Ū = (n + q) log 2 − q φ(−R√d) of inequality (36).
#
#   Style: one color per dataset (same as Phase 1);
#          r_0 dashed, larger hollow diamond; r solid, smaller filled diamond
#          (the two curves nearly coincide, so both remain visible).
#   Vertical axis: linear, with its own scale in each panel.
# ==============================================================================
using CSV, DataFrames, Plots, Measures, LaTeXStrings
gr()

const ROOT = normpath(joinpath(@__DIR__, ".."))   # repository root
const DIR  = ROOT
mkpath(joinpath(DIR, "figures"))

# ==========================================
# DADOS
# ==========================================
r1  = CSV.read(joinpath(DIR, "results", "phase2", "phase2_results.csv"), DataFrame)
r2  = CSV.read(joinpath(DIR, "results", "phase2_pagfull", "phase2_pagfull.csv"), DataFrame)
bnd = CSV.read(joinpath(DIR, "results", "phase2", "bounds_summary.csv"), DataFrame)
r1.method = replace(r1.method, "PA_G" => "PA_G_0")           # Prop. 3 only
r2.method = replace(r2.method, "PA_G_full" => "PA_G")        # all bounds
cols = [:dataset, :q_ratio, :method, :n, :d, :q, :dual]
df = vcat(select(r1, cols), select(r2, cols))

datasets    = ["biopsy", "Smarket", "fgl", "College", "OJ", "Banknote", "Ionosphere", "Spambase"]
names_show  = Dict("biopsy" => "Biopsy")
rhos        = [0.05, 0.10, 0.15, 0.20]
x_positions = 1:length(rhos)
x_labels    = ["5%", "10%", "15%", "20%"]

num(x) = (ismissing(x) || (x isa Real && isnan(x))) ? NaN : Float64(x)

function getv(ds, rho, m, c)
    sub = df[(df.dataset .== ds) .& isapprox.(df.q_ratio, rho) .& (df.method .== m), :]
    @assert nrow(sub) == 1 "Esperada 1 linha para $ds, $rho, $m"
    return num(sub[1, c])
end

# limite analítico da desigualdade (36): Ū = (n + q) log 2 − q φ(−R√d)
φ(z) = z > 0 ? z + log1p(exp(-z)) : log1p(exp(z))
function Ubar(ds, rho)
    b = bnd[(bnd.dataset .== ds) .& isapprox.(bnd.q_ratio, rho), :]
    @assert nrow(b) == 1 "Esperada 1 linha em bounds_summary para $ds, $rho"
    n, d, q = getv(ds, rho, "CAR", :n), getv(ds, rho, "CAR", :d), getv(ds, rho, "CAR", :q)
    return (n + q) * log(2) - q * φ(-b[1, :R] * sqrt(d))
end

# the two dual ratios, computed with min{dual, Ū}
function ratios(ds)
    r0, r = Float64[], Float64[]
    for rho in rhos
        U = Ubar(ds, rho)
        dual(m) = min(getv(ds, rho, m, :dual), U)
        push!(r0, dual("PA_G_0") / dual("CAR_0"))
        push!(r,  dual("PA_G")   / dual("CAR"))
    end
    return r0, r
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

# ==========================================
# SUBGRÁFICOS
# ==========================================
ratio_subplots = []

for ds in datasets
    c = color_map[ds]
    r0, r = ratios(ds)
    ttl = get(names_show, ds, ds)

    # eixo linear, próprio de cada painel, começando logo abaixo de 1
    ymax = max(1.1, 1.05 * maximum(vcat(r0, r)))
    p_ratio = plot(title = ttl, xlabel = "ρ", ylabel = "dual ratio",
                   xticks = (x_positions, x_labels), ylims = (0.98, ymax))
    hline!(p_ratio, [1.0]; color = :gray, linestyle = :dot, linewidth = 1, marker = :none)
    plot!(p_ratio, x_positions, r0; color = c, linestyle = :dash, marker = :diamond,
          markersize = 8, markercolor = :white, markerstrokecolor = c)
    plot!(p_ratio, x_positions, r;  color = c, linestyle = :solid, marker = :diamond,
          markersize = 4, markercolor = c, markerstrokecolor = c)
    push!(ratio_subplots, p_ratio)
end

# ==========================================
# PAINEL DE LEGENDA (desenhado à mão)
# ==========================================
function legend_panel(entries)
    p = plot(xlims = (0, 1), ylims = (0, 1), framestyle = :box, grid = false,
             ticks = false, title = " ")
    ys = range(0.80, 0.20; length = length(entries))
    for ((lab, ls, mk, hollow, ms), y) in zip(entries, ys)
        plot!(p, [0.08, 0.38], [y, y]; color = :black, linestyle = ls, marker = :none)
        scatter!(p, [0.23], [y]; marker = mk, markersize = ms,
                 markercolor = hollow ? :white : :black, markerstrokecolor = :black)
        annotate!(p, 0.46, y, text(lab, 12, :left, "Computer Modern"))
    end
    return p
end

ratio_legend = legend_panel([
    (L"r_0", :dash,  :diamond, true,  8),
    (L"r",   :solid, :diamond, false, 4)])

# ==========================================
# LAYOUT (8 datasets + legenda = 3 x 3)
# ==========================================
plt_ratio_grid = plot(ratio_subplots..., ratio_legend;
                      layout = (3, 3), size = (1000, 780),
                      plot_title = " ", plot_titlevspan = 0.03)

display(plt_ratio_grid)

# ==========================================
# SALVAR (PDF para LaTeX)
# ==========================================
savefig(plt_ratio_grid, joinpath(DIR, "figures", "ratio_grid.pdf"))
