"""
MT2013 – Probability and Statistics | Task 2: Hypothesis Testing
Person 3 – Coding & Visualization
==========================================================
Section C : ANOVA     (contaminant_ppm across all 6 water source types)
Section D : Chi-Square Test 1  (Well vs. River × High/Low)
Section E : Chi-Square Test 2  (All 6 sources  × High/Low)

Each section ends with a VERIFICATION block that compares the
computer result against the hand-written values from the
Task-2 research document.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ── colour palette ─────────────────────────────────────────────────────────────
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

# helper: print section banners
def banner(title):
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)

def check(label, computed, expected, tol=1e-3):
    """Print a pass/fail verification line."""
    match = abs(computed - expected) < tol
    status = "✅ MATCH" if match else f"❌ MISMATCH  (expected {expected:.6f})"
    print(f"  {label:<40} computed = {computed:.6f}   {status}")


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

# ── ANOVA verification against hand-written document ─────────────────────────
print("\n── Verification vs. hand-written calculations ──")
# Document values:  grand_mean=4.954389, SSTr=9.139739, MSTr=1.827948,
#                   SSE≈24522.717, MSE≈8.190620, F₀≈0.223176, F_crit=2.21 (approx)
check("Grand mean",        grand_mean,  4.954389, tol=1e-4)
check("SSTr",              SSTr,        9.139739, tol=1e-3)
check("MSTr",              MSTr,        1.827948, tol=1e-3)
check("SSE",               SSE,         24522.717, tol=1.0)
check("MSE",               MSE,         8.190620,  tol=1e-3)
check("F₀ statistic",      F0,          0.223176,  tol=1e-3)
check("F_crit (≈2.21)",    F_crit,      2.21,      tol=0.10)

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
plt.savefig("C1_anova.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n✔ Saved: C1_anova.png")


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

# ── Verification ──────────────────────────────────────────────────────────────
print("\n── Verification vs. hand-written calculations ──")
# Document observed: Well-High=252, Well-Low=246, River-High=269, River-Low=269
# Expected: E_WH=250.44, E_WL=247.56, E_RH=270.56, E_RL=267.44
# χ²=0.0376
check("E_Well_High",    E2_df.loc["Well","High"],   250.44, tol=0.1)
check("E_Well_Low",     E2_df.loc["Well","Low"],    247.56, tol=0.1)
check("E_River_High",   E2_df.loc["River","High"],  270.56, tol=0.1)
check("E_River_Low",    E2_df.loc["River","Low"],   267.44, tol=0.1)
check("χ² statistic",  chi2_2, 0.0376, tol=0.01)
check("χ²_crit (df=1)", chi2_crit_2, 3.84, tol=0.01)

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
plt.savefig("D1_chisq_well_vs_river.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n✔ Saved: D1_chisq_well_vs_river.png")


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

# ── Verification ──────────────────────────────────────────────────────────────
print("\n── Verification vs. hand-written calculations ──")
# Document expected values and contributions:
doc_E = {
    ("Well",  "High"): 246.36, ("Well",  "Low"): 251.64,
    ("River", "High"): 266.15, ("River", "Low"): 271.85,
    ("Lake",  "High"): 237.95, ("Lake",  "Low"): 243.05,
    ("Pond",  "High"): 222.62, ("Pond",  "Low"): 227.38,
    ("Spring","High"): 263.18, ("Spring","Low"): 268.82,
    ("Tap",   "High"): 247.84, ("Tap",   "Low"): 253.16,
}
for (src, cat), doc_val in doc_E.items():
    check(f"E_{src}_{cat}", E_all_df.loc[src, cat], doc_val, tol=0.5)

check("Total χ² statistic", chi2_all, 1.0165, tol=0.05)
check("χ²_crit (df=5)",     chi2_crit_all, 11.07, tol=0.1)

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
plt.savefig("E1_chisq_all_sources.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n✔ Saved: E1_chisq_all_sources.png")

# =============================================================================
# FINAL SUMMARY TABLE
# =============================================================================
banner("FINAL SUMMARY – All Hypothesis Tests (Task 2)")
print(f"  {'Test':<35} {'Statistic':>12} {'Critical':>10} {'p-value':>10} {'Decision'}")
print("  " + "-" * 80)
print(f"  {'Two-Sample t-Test (Well vs River)':<35} {'z₀=0.5010':>12} {'±1.96':>10} {'0.6164':>10}  Fail to Reject H₀")
print(f"  {'ANOVA (6 sources)':<35} {f'F₀={F0:.4f}':>12} {f'{F_crit:.2f}':>10} {f'{p_anova:.4f}':>10}  Fail to Reject H₀")
print(f"  {'Chi-Square 1 (Well vs River)':<35} {f'χ²={chi2_2:.4f}':>12} {'3.84':>10} {f'{p_chi2_2:.4f}':>10}  Fail to Reject H₀")
print(f"  {'Chi-Square 2 (All 6 sources)':<35} {f'χ²={chi2_all:.4f}':>12} {'11.07':>10} {f'{p_chi2_all:.4f}':>10}  Fail to Reject H₀")
print()
print("  Overall: No test found a statistically significant difference in")
print("  contamination levels across any grouping at α = 0.05.")
print()
print("✔ All sections complete.")
