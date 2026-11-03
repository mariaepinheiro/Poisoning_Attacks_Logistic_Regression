# ==============================================================================
# figures_phase1.jl — AC and MCC under attack (Phase 1)
#   Reads  results/phase1/phase1_results.csv
#   Writes ac_grid.pdf and mcc_grid.pdf
#   Mean over the 5 partitions, shaded band = ± 1 standard deviation.
#   The first point ("LR") is the clean model.
#   Layout: 4 rows × 2 columns (one panel per dataset).
# ==============================================================================
using CSV, DataFrames, Plots, Measures, Statistics
gr()

const NROWS   = 4             # linhas
const NCOLS   = 2             # colunas
const FIGSIZE = (700, 1000)   # largura × altura

const ROOT = normpath(joinpath(@__DIR__, ".."))   # repository root
const SRC  = joinpath(ROOT, "src")
const DIR  = ROOT
mkpath(joinpath(DIR, "figures"))
df = CSV.read(joinpath(DIR, "results", "phase1", "phase1_results.csv"), DataFrame)
df.q_ratio = round.(df.q_ratio, digits = 2)

datasets    = ["biopsy", "Smarket", "fgl", "College", "OJ", "Banknote", "Ionosphere", "Spambase"]
names_show  = Dict("biopsy" => "Biopsy")
rhos        = [0.05, 0.10, 0.15, 0.20]
x_positions = 1:(length(rhos) + 1)
x_labels    = ["LR", "5%", "10%", "15%", "20%"]

# mean and std over partitions: first point = clean LR, then PA_L for each ρ
function curve(ds, col_lr, col_pa)
    sub = df[df.dataset .== ds, :]
    lr  = sub[sub.q_ratio .== rhos[1], col_lr]        # one value per partition
    m = [mean(lr)];  s = [std(lr)]
    for ρ in rhos
        v = sub[sub.q_ratio .== ρ, col_pa]
        push!(m, mean(v));  push!(s, std(v))
    end
    return m, s
end

# ==========================================
# CONFIGURAÇÕES GLOBAIS (mesmas das outras figuras)
# ==========================================
default(
    linewidth = 2,
    markersize = 4,
    marker = :circle,
    markerstrokewidth = 0,
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

function grid_plot(col_lr, col_pa, ylab, ylims)
    subplots = []
    for ds in datasets
        c = color_map[ds]
        m, s = curve(ds, col_lr, col_pa)
        p = plot(x_positions, m; ribbon = s, fillalpha = 0.2,
                 color = c, markercolor = c,
                 title = get(names_show, ds, ds), xlabel = "ρ", ylabel = ylab,
                 xticks = (x_positions, x_labels), ylims = ylims)
        push!(subplots, p)
    end
    return plot(subplots...; layout = (NROWS, NCOLS), size = FIGSIZE,
                plot_title = " ", plot_titlevspan = 0.03)
end

plt_ac  = grid_plot(:ac_lr,  :ac_pal,  "AC",  (0.2, 1.05))
plt_mcc = grid_plot(:mcc_lr, :mcc_pal, "MCC", (-0.3, 1.05))

println("Layout usado: $(NROWS) linhas × $(NCOLS) colunas")

display(plt_ac)
display(plt_mcc)

savefig(plt_ac, joinpath(DIR, "figures", "ac_grid.pdf"))
savefig(plt_mcc, joinpath(DIR, "figures", "mcc_grid.pdf"))