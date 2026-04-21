"""
MT2013 – Probability and Statistics | Task 2: Hypothesis Testing
Person 3 – Coding & Visualization
==========================================================
Section A : General Visualizations  (histogram, boxplot, bar chart, scatter)
Section B : Two-Sample t-Test       (Well vs. River contaminant_ppm)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ── colour palette (consistent across all plots) ──────────────────────────────
PALETTE = {
    "Well":   "#2196F3",
    "River":  "#4CAF50",
    "Lake":   "#FF9800",
    "Pond":   "#9C27B0",
    "Spring": "#F44336",
    "Tap":    "#00BCD4",
}
SOURCE_ORDER = ["Well", "River", "Lake", "Pond", "Spring", "Tap"]
COLORS       = [PALETTE[s] for s in SOURCE_ORDER]

# ── load data ──────────────────────────────────────────────────────────────────
df = pd.read_csv("Data_cleaned_for_analysis.csv")
print(f"Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns\n")

# =============================================================================
# SECTION A – GENERAL VISUALIZATIONS
# =============================================================================

# ── A1. Histogram – distribution of contaminant_ppm ──────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(df["contaminant_ppm"], bins=40, color="#2196F3", edgecolor="white",
        linewidth=0.6, alpha=0.88)
ax.axvline(df["contaminant_ppm"].mean(), color="#E53935", linewidth=2,
           linestyle="--", label=f"Mean = {df['contaminant_ppm'].mean():.2f} ppm")
ax.axvline(df["contaminant_ppm"].median(), color="#FFC107", linewidth=2,
           linestyle=":", label=f"Median = {df['contaminant_ppm'].median():.2f} ppm")
ax.set_title("Distribution of Contaminant Level (ppm)", fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Contaminant Level (ppm)", fontsize=11)
ax.set_ylabel("Frequency", fontsize=11)
ax.legend(fontsize=10)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("A1_histogram_contaminant.png", dpi=150, bbox_inches="tight")
plt.close()
print("✔ Saved: A1_histogram_contaminant.png")

# ── A2. Boxplot – contaminant_ppm by water_source_type (group comparison) ────
fig, ax = plt.subplots(figsize=(10, 6))
data_by_source = [df.loc[df["water_source_type"] == s, "contaminant_ppm"].values
                  for s in SOURCE_ORDER]
bp = ax.boxplot(data_by_source, patch_artist=True, notch=False,
                medianprops=dict(color="white", linewidth=2),
                whiskerprops=dict(linewidth=1.3),
                capprops=dict(linewidth=1.3),
                flierprops=dict(marker="o", markersize=3, alpha=0.4))
for patch, color in zip(bp["boxes"], COLORS):
    patch.set_facecolor(color)
    patch.set_alpha(0.82)
ax.set_xticklabels(SOURCE_ORDER, fontsize=11)
ax.set_title("Contaminant Level (ppm) by Water Source Type", fontsize=14,
             fontweight="bold", pad=12)
ax.set_xlabel("Water Source Type", fontsize=11)
ax.set_ylabel("Contaminant Level (ppm)", fontsize=11)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("A2_boxplot_by_source.png", dpi=150, bbox_inches="tight")
plt.close()
print("✔ Saved: A2_boxplot_by_source.png")

# ── A3. Bar chart – mean contaminant_ppm per water source ────────────────────
means = df.groupby("water_source_type")["contaminant_ppm"].mean().reindex(SOURCE_ORDER)
sems  = df.groupby("water_source_type")["contaminant_ppm"].sem().reindex(SOURCE_ORDER)

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(SOURCE_ORDER, means, color=COLORS, edgecolor="white",
              linewidth=0.8, alpha=0.88, width=0.6,
              yerr=sems, capsize=5, error_kw=dict(linewidth=1.4, color="#444"))
for bar, val in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
            f"{val:.2f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
ax.set_title("Mean Contaminant Level (ppm) by Water Source Type\n(error bars = ±1 SE)",
             fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Water Source Type", fontsize=11)
ax.set_ylabel("Mean Contaminant Level (ppm)", fontsize=11)
ax.set_ylim(0, means.max() * 1.25)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("A3_barchart_means.png", dpi=150, bbox_inches="tight")
plt.close()
print("✔ Saved: A3_barchart_means.png")

# ── A4. Scatter plot – contaminant_ppm vs. ph_level (coloured by source) ─────
fig, ax = plt.subplots(figsize=(9, 6))
for source in SOURCE_ORDER:
    sub = df[df["water_source_type"] == source]
    ax.scatter(sub["ph_level"], sub["contaminant_ppm"],
               color=PALETTE[source], label=source,
               alpha=0.35, s=18, linewidths=0)
# overall regression line
slope, intercept, r, p, _ = stats.linregress(df["ph_level"], df["contaminant_ppm"])
x_line = np.linspace(df["ph_level"].min(), df["ph_level"].max(), 200)
ax.plot(x_line, intercept + slope * x_line, color="#E53935",
        linewidth=2, linestyle="--",
        label=f"Linear fit  r = {r:.3f}")
ax.set_title("Contaminant Level vs. pH Level\n(coloured by water source type)",
             fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("pH Level", fontsize=11)
ax.set_ylabel("Contaminant Level (ppm)", fontsize=11)
ax.legend(fontsize=9, loc="upper right", framealpha=0.8)
ax.grid(linestyle="--", alpha=0.35)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("A4_scatter_ppm_vs_ph.png", dpi=150, bbox_inches="tight")
plt.close()
print("✔ Saved: A4_scatter_ppm_vs_ph.png\n")

# =============================================================================
# SECTION B – TWO-SAMPLE t-TEST  (Well vs. River, large-sample z approach)
# =============================================================================

print("=" * 65)
print("SECTION B – Two-Sample t-Test (Well vs. River)")
print("Research question: Is the mean contaminant level in well water")
print("significantly different from that in river water?")
print("=" * 65)

# ── extract groups ─────────────────────────────────────────────────────────────
well  = df.loc[df["water_source_type"] == "Well",  "contaminant_ppm"]
river = df.loc[df["water_source_type"] == "River", "contaminant_ppm"]

x_well  = well.mean();   s_well  = well.std(ddof=1);   n_well  = len(well)
x_river = river.mean();  s_river = river.std(ddof=1);  n_river = len(river)

print(f"\n{'Statistic':<28} {'Well':>12} {'River':>12}")
print("-" * 54)
print(f"{'Sample size (n)':<28} {n_well:>12} {n_river:>12}")
print(f"{'Sample mean (x̄)':<28} {x_well:>12.8f} {x_river:>12.8f}")
print(f"{'Sample std dev (s)':<28} {s_well:>12.8f} {s_river:>12.8f}")

# ── hypotheses ─────────────────────────────────────────────────────────────────
print("\nHypotheses:")
print("  H₀: μ_Well = μ_River   (i.e. μ_Well − μ_River = 0)")
print("  H₁: μ_Well ≠ μ_River   (two-tailed)")
print("  Significance level α = 0.05")

# ── test statistic (large-sample z) ───────────────────────────────────────────
#   z₀ = (x̄₁ − x̄₂ − (μ₁−μ₂)₀) / sqrt(s₁²/n₁ + s₂²/n₂)
diff       = x_well - x_river          # observed difference
se_diff    = np.sqrt(s_well**2 / n_well + s_river**2 / n_river)
z0         = diff / se_diff

# ── critical value & p-value ───────────────────────────────────────────────────
alpha      = 0.05
z_critical = stats.norm.ppf(1 - alpha / 2)
p_value    = 2 * (1 - stats.norm.cdf(abs(z0)))

print(f"\nCalculations:")
print(f"  Difference in means (x̄_Well − x̄_River) = {diff:+.8f}")
print(f"  Standard error of difference            = {se_diff:.8f}")
print(f"  Test statistic  z₀                      = {z0:+.8f}")
print(f"  Critical value  z_{{α/2}} = z_0.025        = ±{z_critical:.4f}")
print(f"  p-value (two-tailed)                    = {p_value:.6f}")

# ── decision ───────────────────────────────────────────────────────────────────
print(f"\nDecision rule:  Reject H₀ if |z₀| > {z_critical:.4f}")
print(f"                |z₀| = {abs(z0):.4f}  →  ", end="")
if abs(z0) > z_critical:
    print("REJECT H₀")
    conclusion = "reject"
else:
    print("FAIL TO REJECT H₀")
    conclusion = "fail to reject"

print(f"\nConclusion:")
print(f"  At the 5 % significance level we {conclusion} H₀.")
if conclusion == "fail to reject":
    print("  There is insufficient evidence to conclude that the mean contaminant")
    print("  level in well water differs from that in river water.")
else:
    print("  There is sufficient evidence that the mean contaminant level in")
    print("  well water significantly differs from that in river water.")

# ── B-plot: visualise the two distributions + z statistic ─────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left – overlapping histograms
ax = axes[0]
ax.hist(well,  bins=30, color=PALETTE["Well"],  alpha=0.65, edgecolor="white",
        label=f"Well  (n={n_well}, x̄={x_well:.4f})", density=True)
ax.hist(river, bins=30, color=PALETTE["River"], alpha=0.65, edgecolor="white",
        label=f"River (n={n_river}, x̄={x_river:.4f})", density=True)
ax.axvline(x_well,  color=PALETTE["Well"],  linewidth=2.2, linestyle="--")
ax.axvline(x_river, color=PALETTE["River"], linewidth=2.2, linestyle="--")
ax.set_title("Distribution: Well vs. River\n(contaminant ppm)", fontsize=12,
             fontweight="bold")
ax.set_xlabel("Contaminant Level (ppm)"); ax.set_ylabel("Density")
ax.legend(fontsize=9); ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.spines[["top", "right"]].set_visible(False)

# Right – standard normal with rejection regions
ax = axes[1]
x_n = np.linspace(-4, 4, 400)
ax.plot(x_n, stats.norm.pdf(x_n), color="#333", linewidth=2)

# shade rejection regions
x_rej_l = np.linspace(-4, -z_critical, 200)
x_rej_r = np.linspace(z_critical, 4, 200)
ax.fill_between(x_rej_l, stats.norm.pdf(x_rej_l), color="#E53935", alpha=0.35,
                label=f"Rejection region (α/2={alpha/2})")
ax.fill_between(x_rej_r, stats.norm.pdf(x_rej_r), color="#E53935", alpha=0.35)

# mark test statistic
ax.axvline(z0, color="#1565C0", linewidth=2.2, linestyle="--",
           label=f"z₀ = {z0:.4f}")
ax.axvline(-z_critical, color="#E53935", linewidth=1.5, linestyle=":")
ax.axvline( z_critical, color="#E53935", linewidth=1.5, linestyle=":",
            label=f"±z_{{α/2}} = ±{z_critical:.2f}")

ax.set_title("Standard Normal: Two-Sample t-Test\n(Well vs. River)", fontsize=12,
             fontweight="bold")
ax.set_xlabel("z"); ax.set_ylabel("Density")
ax.legend(fontsize=9); ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.spines[["top", "right"]].set_visible(False)
ax.annotate(f"z₀ = {z0:.4f}\np = {p_value:.4f}\n→ Fail to Reject H₀",
            xy=(z0, stats.norm.pdf(z0)),
            xytext=(z0 + 0.6, 0.25),
            arrowprops=dict(arrowstyle="->", color="#1565C0"),
            fontsize=9, color="#1565C0",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#1565C0", alpha=0.9))

plt.tight_layout()
plt.savefig("B1_two_sample_ttest.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n✔ Saved: B1_two_sample_ttest.png")
print("\nAll outputs complete.")