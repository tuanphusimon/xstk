"""
MT2013 – Probability and Statistics | Task 2: Hypothesis Testing
Person 3 – Coding & Visualization
==========================================================
Section A : General Visualizations  (histogram, boxplot, bar chart, scatter)
Section B : Two-Sample t-Test       (Well vs. River contaminant_ppm)
Section C : ANOVA                   (contaminant_ppm across all 6 water source types)
Section D : Chi-Square Test 1       (Well vs. River × High/Low)
Section E : Chi-Square Test 2       (All 6 sources  × High/Low)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
import warnings
import os  # Added to handle folder creation and paths
warnings.filterwarnings("ignore")

# ── setup output directory ───────────────────────────────────────────────────
OUTPUT_DIR = "graph images"
os.makedirs(OUTPUT_DIR, exist_ok=True)  # Creates the folder if it doesn't exist

# ── helper functions ─────────────────────────────────────────────────────────
def banner(title):
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)

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
# SECTION A – GENERAL VISUALIZATIONS & DATA SUMMARY
# =============================================================================
banner("SECTION A – GENERAL VISUALIZATIONS & DATA SUMMARY")

# ── A0. Data Summary (Descriptive Statistics) ────────────────────────────────
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print("Descriptive Statistics for Quantitative Variables:")
print("-" * 65)
print(df.describe())
print("-" * 65)

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
plt.savefig(os.path.join(OUTPUT_DIR, "A1_histogram_contaminant.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"\n✔ Saved: {OUTPUT_DIR}/A1_histogram_contaminant.png")

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
plt.savefig(os.path.join(OUTPUT_DIR, "A2_boxplot_by_source.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"✔ Saved: {OUTPUT_DIR}/A2_boxplot_by_source.png")

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
plt.savefig(os.path.join(OUTPUT_DIR, "A3_barchart_means.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"✔ Saved: {OUTPUT_DIR}/A3_barchart_means.png")

# ── A4. Scatter plot – contaminant_ppm vs. ph_level (coloured by source) ─────
fig, ax = plt.subplots(figsize=(9, 6))
for source in SOURCE_ORDER:
    sub = df[df["water_source_type"] == source]
    ax.scatter(sub["ph_level"], sub["contaminant_ppm"],
               color=PALETTE[source], label=source,
               alpha=0.35, s=18, linewidths=0)
# overall regression line
slope, intercept, r, p_reg, _ = stats.linregress(df["ph_level"], df["contaminant_ppm"])
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
plt.savefig(os.path.join(OUTPUT_DIR, "A4_scatter_ppm_vs_ph.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"✔ Saved: {OUTPUT_DIR}/A4_scatter_ppm_vs_ph.png\n")


# =============================================================================
# SECTION B – TWO-SAMPLE t-TEST  (Well vs. River, large-sample z approach)
# =============================================================================
banner("SECTION B – Two-Sample t-Test (Well vs. River)")
print("Research question: Is the mean contaminant level in well water")
print("significantly different from that in river water?")

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
ax.set_title("Distribution: Well vs. River\n(contaminant ppm)", fontsize=12, fontweight="bold")
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

ax.set_title("Standard Normal: Two-Sample t-Test\n(Well vs. River)", fontsize=12, fontweight="bold")
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
plt.savefig(os.path.join(OUTPUT_DIR, "B1_two_sample_ttest.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"\n✔ Saved: {OUTPUT_DIR}/B1_two_sample_ttest.png")


# =============================================================================
# SECTION C – ANOVA
# =============================================================================
banner("SECTION C – One-Way ANOVA (all 6 water source types)")
print("Research question: Does mean contaminant level differ")
print("significantly across Lake, Pond, River, Spring, Tap, Well?")

groups      = [df.loc[df["water_source_type"] == s, "contaminant_ppm"].values
               for s in SOURCE_ORDER]
group_means = np.array([g.mean() for g in groups])
group_ns    = np.array([len(g)   for g in groups])
group_stds  = np.array([g.std(ddof=1) for g in groups])

N  = sum(group_ns)          # total observations
a  = len(groups)            # number of groups

# Grand mean
grand_mean = sum(ni * xi for ni, xi in zip(group_ns, group_means)) / N

# Sum of Squares
SSTr = sum(ni * (xi - grand_mean)**2
           for ni, xi in zip(group_ns, group_means))          # between groups
SSE  = sum(((g - gm)**2).sum()
           for g, gm in zip(groups, group_means))              # within groups
SST  = SSTr + SSE                                              # total

# Degrees of freedom
df_Tr = a - 1
df_E  = N - a

# Mean squares
MSTr = SSTr / df_Tr
MSE  = SSE  / df_E

# F statistic
F0 = MSTr / MSE

# Critical value and p-value
F_crit = stats.f.ppf(1 - 0.05, df_Tr, df_E)
p_anova = 1 - stats.f.cdf(F0, df_Tr, df_E)

print(f"\n{'':4}{'Statistic':<35} {'Value':>14}")
print("  " + "-" * 50)
print(f"  {'Total observations (N)':<35} {N:>14}")
print(f"  {'Number of groups (a)':<35} {a:>14}")
print(f"  {'Grand mean (x̄..)':<35} {grand_mean:>14.8f}")
print()
print(f"  {'SSTr  (between groups)':<35} {SSTr:>14.6f}")
print(f"  {'SSE   (within  groups)':<35} {SSE:>14.6f}")
print(f"  {'SST   (total)':<35} {SST:>14.6f}")
print()
print(f"  {'df_Treatment (a−1)':<35} {df_Tr:>14}")
print(f"  {'df_Error     (N−a)':<35} {df_E:>14}")
print()
print(f"  {'MSTr = SSTr / df_Tr':<35} {MSTr:>14.8f}")
print(f"  {'MSE  = SSE  / df_E':<35} {MSE:>14.8f}")
print()
print(f"  {'F₀ = MSTr / MSE':<35} {F0:>14.8f}")
print(f"  {'F_crit = F(0.05; {df_Tr}; {df_E})':<35} {F_crit:>14.4f}")
print(f"  {'p-value':<35} {p_anova:>14.6f}")

print(f"\nDecision rule:  Reject H₀ if F₀ > F_crit = {F_crit:.4f}")
print(f"                F₀ = {F0:.4f}  →  ", end="")
if F0 > F_crit:
    print("REJECT H₀")
    anova_decision = "reject"
else:
    print("FAIL TO REJECT H₀")
    anova_decision = "fail to reject"

print(f"\nConclusion:")
print(f"  At α = 0.05 we {anova_decision} H₀.")
print("  There is no statistically significant evidence that the mean")
print("  contaminant level differs across the six water source types.")

# ── C-plot 1: group means with CI ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
cis = [stats.t.interval(0.95, df=n-1, loc=m, scale=s/np.sqrt(n))
       for m, s, n in zip(group_means, group_stds, group_ns)]
lower_err = group_means - np.array([ci[0] for ci in cis])
upper_err = np.array([ci[1] for ci in cis]) - group_means
ax.bar(SOURCE_ORDER, group_means, color=COLORS, alpha=0.82, edgecolor="white",
       width=0.6, yerr=[lower_err, upper_err], capsize=5,
       error_kw=dict(linewidth=1.4, color="#444"))
ax.axhline(grand_mean, color="#E53935", linewidth=1.8, linestyle="--",
           label=f"Grand mean = {grand_mean:.4f}")
for i, (s, m) in enumerate(zip(SOURCE_ORDER, group_means)):
    ax.text(i, m + upper_err[i] + 0.05, f"{m:.4f}",
            ha="center", va="bottom", fontsize=8.5, fontweight="bold")
ax.set_title("ANOVA: Group Means with 95% CI\n(contaminant ppm by water source)",
             fontsize=12, fontweight="bold")
ax.set_xlabel("Water Source Type"); ax.set_ylabel("Mean Contaminant (ppm)")
ax.legend(fontsize=9); ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.spines[["top","right"]].set_visible(False)

# ── C-plot 2: F distribution with F₀ ─────────────────────────────────────────
ax = axes[1]
x_f = np.linspace(0, 5, 500)
y_f = stats.f.pdf(x_f, df_Tr, df_E)
ax.plot(x_f, y_f, color="#333", linewidth=2, label=f"F({df_Tr}, {df_E})")
x_rej = np.linspace(F_crit, 5, 200)
ax.fill_between(x_rej, stats.f.pdf(x_rej, df_Tr, df_E),
                color="#E53935", alpha=0.35, label=f"Rejection region (α=0.05)")
ax.axvline(F0, color="#1565C0", linewidth=2.2, linestyle="--",
           label=f"F₀ = {F0:.4f}")
ax.axvline(F_crit, color="#E53935", linewidth=1.5, linestyle=":",
           label=f"F_crit = {F_crit:.2f}")
ax.annotate(f"F₀ = {F0:.4f}\np = {p_anova:.4f}\n→ Fail to Reject H₀",
            xy=(F0, stats.f.pdf(F0, df_Tr, df_E)),
            xytext=(F0 + 0.5, 0.4),
            arrowprops=dict(arrowstyle="->", color="#1565C0"),
            fontsize=9, color="#1565C0",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#1565C0", alpha=0.9))
ax.set_title(f"F-Distribution: ANOVA Test\n(df_Tr={df_Tr}, df_E={df_E})",
             fontsize=12, fontweight="bold")
ax.set_xlabel("F"); ax.set_ylabel("Density")
ax.legend(fontsize=9); ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.spines[["top","right"]].set_visible(False)
ax.set_xlim(0, 5)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "C1_anova.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"\n✔ Saved: {OUTPUT_DIR}/C1_anova.png")


# =============================================================================
# SECTION D – Chi-Square Test 1: Well vs. River × High/Low
# =============================================================================
banner("SECTION D – Chi-Square Test 1: Well vs. River (High/Low)")
print("Research question: Is there an association between water source")
print("(Well vs. River) and contamination category (High ≥5 / Low <5)?")

df["contam_cat"] = df["contaminant_ppm"].apply(lambda x: "High" if x >= 5 else "Low")

sub2  = df[df["water_source_type"].isin(["Well", "River"])]
ct2   = pd.crosstab(sub2["water_source_type"], sub2["contam_cat"])
ct2   = ct2.reindex(index=["Well","River"], columns=["High","Low"])

print("\nObserved Frequency Table:")
print(ct2.to_string())

# row/col totals
row_totals = ct2.sum(axis=1).values    # [Well_total, River_total]
col_totals = ct2.sum(axis=0).values    # [High_total,  Low_total]
N2         = ct2.values.sum()

# Expected frequencies
E2 = np.outer(row_totals, col_totals) / N2
E2_df = pd.DataFrame(E2, index=["Well","River"], columns=["High","Low"])

print("\nExpected Frequency Table:")
print(E2_df.round(4).to_string())

# Chi-square statistic
O2  = ct2.values.astype(float)
chi2_2 = ((O2 - E2)**2 / E2).sum()
df2_chi = (ct2.shape[0] - 1) * (ct2.shape[1] - 1)
chi2_crit_2 = stats.chi2.ppf(0.95, df2_chi)
p_chi2_2    = 1 - stats.chi2.cdf(chi2_2, df2_chi)

print(f"\n  χ² statistic             = {chi2_2:.4f}")
print(f"  Degrees of freedom       = {df2_chi}")
print(f"  χ²_crit (α=0.05, df={df2_chi}) = {chi2_crit_2:.4f}")
print(f"  p-value                  = {p_chi2_2:.4f}")

print(f"\nDecision rule:  Reject H₀ if χ² > χ²_crit = {chi2_crit_2:.4f}")
print(f"                χ² = {chi2_2:.4f}  →  ", end="")
if chi2_2 > chi2_crit_2:
    print("REJECT H₀")
    d_decision = "reject"
else:
    print("FAIL TO REJECT H₀")
    d_decision = "fail to reject"

print(f"\nConclusion:")
print(f"  At α = 0.05 we {d_decision} H₀.")
print("  The probability of High/Low contamination is roughly the same")
print("  for both Well and River water sources.")

# ── D-plot 1: observed vs expected grouped bar ────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
x_pos  = np.arange(2)
width  = 0.2
labels = ["Well-High", "Well-Low", "River-High", "River-Low"]
obs_vals = [O2[0,0], O2[0,1], O2[1,0], O2[1,1]]
exp_vals = [E2[0,0], E2[0,1], E2[1,0], E2[1,1]]
obs_colors = ["#2196F3","#90CAF9","#4CAF50","#A5D6A7"]

# grouped: observed (solid) vs expected (hatched)
for i, (lbl, ov, ev, oc) in enumerate(zip(labels, obs_vals, exp_vals, obs_colors)):
    xp = (i % 2) * 0.22 + (i // 2) * 1.0 - 0.11
    ax.bar(xp,        ov, width=0.18, color=oc,    alpha=0.88, label=f"Obs {lbl}" if i<2 else "_")
    ax.bar(xp + 0.20, ev, width=0.18, color=oc, alpha=0.40,
           hatch="///", edgecolor="grey", label=f"Exp {lbl}" if i<2 else "_")
    ax.text(xp,        ov + 1, f"{ov:.0f}", ha="center", fontsize=8)
    ax.text(xp + 0.20, ev + 1, f"{ev:.1f}", ha="center", fontsize=8, color="grey")

ax.set_xticks([0.10, 1.10])
ax.set_xticklabels(["Well", "River"], fontsize=11)
ax.set_title("Chi-Square Test 1: Well vs. River\nObserved vs. Expected Frequencies",
             fontsize=12, fontweight="bold")
ax.set_ylabel("Count"); ax.set_xlabel("Water Source")
obs_patch = mpatches.Patch(color="#888", label="Observed (solid)")
exp_patch = mpatches.Patch(color="#aaa", hatch="///", label="Expected (hatched)")
ax.legend(handles=[obs_patch, exp_patch], fontsize=9)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.spines[["top","right"]].set_visible(False)

# ── D-plot 2: chi-square distribution ────────────────────────────────────────
ax = axes[1]
x_c = np.linspace(0, 8, 400)
y_c = stats.chi2.pdf(x_c, df2_chi)
ax.plot(x_c, y_c, color="#333", linewidth=2, label=f"χ²(df={df2_chi})")
x_rej = np.linspace(chi2_crit_2, 8, 200)
ax.fill_between(x_rej, stats.chi2.pdf(x_rej, df2_chi),
                color="#E53935", alpha=0.35, label=f"Rejection region (α=0.05)")
ax.axvline(chi2_2, color="#1565C0", linewidth=2.2, linestyle="--",
           label=f"χ² = {chi2_2:.4f}")
ax.axvline(chi2_crit_2, color="#E53935", linewidth=1.5, linestyle=":",
           label=f"χ²_crit = {chi2_crit_2:.2f}")
ax.annotate(f"χ² = {chi2_2:.4f}\np = {p_chi2_2:.4f}\n→ Fail to Reject H₀",
            xy=(chi2_2, stats.chi2.pdf(chi2_2, df2_chi)),
            xytext=(chi2_2 + 0.8, 0.25),
            arrowprops=dict(arrowstyle="->", color="#1565C0"),
            fontsize=9, color="#1565C0",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#1565C0", alpha=0.9))
ax.set_title(f"χ²-Distribution: Test 1 (Well vs. River)\ndf = {df2_chi}",
             fontsize=12, fontweight="bold")
ax.set_xlabel("χ²"); ax.set_ylabel("Density")
ax.legend(fontsize=9); ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.spines[["top","right"]].set_visible(False)
ax.set_xlim(0, 8)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "D1_chisq_well_vs_river.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"\n✔ Saved: {OUTPUT_DIR}/D1_chisq_well_vs_river.png")


# =============================================================================
# SECTION E – Chi-Square Test 2: All 6 sources × High/Low
# =============================================================================
banner("SECTION E – Chi-Square Test 2: All 6 Sources (High/Low)")
print("Research question: Is contamination category (High/Low) independent")
print("of water source type across all six sources?")

ct_all  = pd.crosstab(df["water_source_type"], df["contam_cat"])
ct_all  = ct_all.reindex(index=SOURCE_ORDER, columns=["High","Low"])

print("\nObserved Frequency Table:")
print(ct_all.to_string())

row_tot_all = ct_all.sum(axis=1).values
col_tot_all = ct_all.sum(axis=0).values
N_all       = ct_all.values.sum()
p_high      = col_tot_all[0] / N_all     # overall P(High)
p_low       = col_tot_all[1] / N_all     # overall P(Low)

print(f"\n  Overall P(High) = {col_tot_all[0]}/{N_all} = {p_high:.4f}")
print(f"  Overall P(Low)  = {col_tot_all[1]}/{N_all} = {p_low:.4f}")

E_all = np.outer(row_tot_all, col_tot_all) / N_all
E_all_df = pd.DataFrame(E_all, index=SOURCE_ORDER, columns=["High","Low"])

print("\nExpected Frequency Table:")
print(E_all_df.round(4).to_string())

# Chi-square statistic
O_all    = ct_all.values.astype(float)
contrib  = (O_all - E_all)**2 / E_all         # cell contributions
chi2_all = contrib.sum()
df_all   = (ct_all.shape[0]-1) * (ct_all.shape[1]-1)
chi2_crit_all = stats.chi2.ppf(0.95, df_all)
p_chi2_all    = 1 - stats.chi2.cdf(chi2_all, df_all)

print(f"\nCell-by-cell χ² contributions:")
contrib_df = pd.DataFrame(contrib.round(4), index=SOURCE_ORDER, columns=["High","Low"])
print(contrib_df.to_string())

print(f"\n  Total χ² statistic          = {chi2_all:.4f}")
print(f"  Degrees of freedom          = {df_all}")
print(f"  χ²_crit (α=0.05, df={df_all}) = {chi2_crit_all:.4f}")
print(f"  p-value                     = {p_chi2_all:.4f}")

print(f"\nDecision rule:  Reject H₀ if χ² > χ²_crit = {chi2_crit_all:.4f}")
print(f"                χ² = {chi2_all:.4f}  →  ", end="")
if chi2_all > chi2_crit_all:
    print("REJECT H₀")
    e_decision = "reject"
else:
    print("FAIL TO REJECT H₀")
    e_decision = "fail to reject"

print(f"\nConclusion:")
print(f"  At α = 0.05 we {e_decision} H₀.")
print("  The likelihood of High/Low contamination is independent of")
print("  whether the source is a Lake, Pond, River, Spring, Tap, or Well.")

# ── E-plot 1: heatmap of observed & contributions ────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Observed heatmap
ax = axes[0]
obs_vals_plot = O_all
im = ax.imshow(obs_vals_plot, cmap="Blues", aspect="auto")
ax.set_xticks([0, 1]); ax.set_xticklabels(["High (≥5)", "Low (<5)"], fontsize=11)
ax.set_yticks(range(6)); ax.set_yticklabels(SOURCE_ORDER, fontsize=11)
for i in range(6):
    for j in range(2):
        ax.text(j, i, f"O={obs_vals_plot[i,j]:.0f}\nE={E_all[i,j]:.1f}",
                ha="center", va="center", fontsize=9,
                color="white" if obs_vals_plot[i,j] > 260 else "#222")
ax.set_title("Chi-Square Test 2: All 6 Sources\nObserved (O) & Expected (E) Frequencies",
             fontsize=12, fontweight="bold")
plt.colorbar(im, ax=ax, shrink=0.8)

# Chi-square distribution for test 2
ax = axes[1]
x_c2 = np.linspace(0, 18, 500)
y_c2 = stats.chi2.pdf(x_c2, df_all)
ax.plot(x_c2, y_c2, color="#333", linewidth=2, label=f"χ²(df={df_all})")
x_rej2 = np.linspace(chi2_crit_all, 18, 200)
ax.fill_between(x_rej2, stats.chi2.pdf(x_rej2, df_all),
                color="#E53935", alpha=0.35, label=f"Rejection region (α=0.05)")
ax.axvline(chi2_all, color="#1565C0", linewidth=2.2, linestyle="--",
           label=f"χ² = {chi2_all:.4f}")
ax.axvline(chi2_crit_all, color="#E53935", linewidth=1.5, linestyle=":",
           label=f"χ²_crit = {chi2_crit_all:.2f}")
ax.annotate(f"χ² = {chi2_all:.4f}\np = {p_chi2_all:.4f}\n→ Fail to Reject H₀",
            xy=(chi2_all, stats.chi2.pdf(chi2_all, df_all)),
            xytext=(chi2_all + 2, 0.05),
            arrowprops=dict(arrowstyle="->", color="#1565C0"),
            fontsize=9, color="#1565C0",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#1565C0", alpha=0.9))
ax.set_title(f"χ²-Distribution: Test 2 (All 6 Sources)\ndf = {df_all}",
             fontsize=12, fontweight="bold")
ax.set_xlabel("χ²"); ax.set_ylabel("Density")
ax.legend(fontsize=9); ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.spines[["top","right"]].set_visible(False)
ax.set_xlim(0, 18)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "E1_chisq_all_sources.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"\n✔ Saved: {OUTPUT_DIR}/E1_chisq_all_sources.png")

# =============================================================================
# FINAL SUMMARY TABLE
# =============================================================================
banner("FINAL SUMMARY – All Hypothesis Tests (Task 2)")
print(f"  {'Test':<35} {'Statistic':>12} {'Critical':>10} {'p-value':>10} {'Decision'}")
print("  " + "-" * 80)
print(f"  {'Two-Sample t-Test (Well vs River)':<35} {f'z₀={z0:.4f}':>12} {'±1.96':>10} {f'{p_value:.4f}':>10}  Fail to Reject H₀")
print(f"  {'ANOVA (6 sources)':<35} {f'F₀={F0:.4f}':>12} {f'{F_crit:.2f}':>10} {f'{p_anova:.4f}':>10}  Fail to Reject H₀")
print(f"  {'Chi-Square 1 (Well vs River)':<35} {f'χ²={chi2_2:.4f}':>12} {'3.84':>10} {f'{p_chi2_2:.4f}':>10}  Fail to Reject H₀")
print(f"  {'Chi-Square 2 (All 6 sources)':<35} {f'χ²={chi2_all:.4f}':>12} {'11.07':>10} {f'{p_chi2_all:.4f}':>10}  Fail to Reject H₀")
print()
print("  Overall: No test found a statistically significant difference in")
print("  contamination levels across any grouping at α = 0.05.")
print()
print("✔ All sections complete.")
