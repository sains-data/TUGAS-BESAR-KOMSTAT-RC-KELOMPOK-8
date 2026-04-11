# Tugas Besar Komstat - Python/NumPy Implementation
# Simulasi Statistik Penumpang Angkutan Udara
# Kelompok 8 RC - Institut Teknologi Sumatera (ITERA) 2025

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.stats import norm, cauchy, kstest

# ─────────────────────────────────────────────────────────────
# 1. Load & Import Data
# ─────────────────────────────────────────────────────────────

url = (
    "https://raw.githubusercontent.com/sains-data/"
    "TUGAS-BESAR-KOMSTAT-RC-KELOMPOK-8/refs/heads/main/"
    "PenumpangAngkutanUdara.csv"
)

df = pd.read_csv(url, sep=";")
print("=== Head of Data ===")
print(df.head())
print("\nShape:", df.shape)

# ─────────────────────────────────────────────────────────────
# 2. Cleaning Data
# ─────────────────────────────────────────────────────────────

print("\nMissing values:", df.isnull().sum().sum())

df["Total_Kedatangan"] = pd.to_numeric(df["Total_Kedatangan"], errors="coerce").astype("Int64")
# Note: column name "Total_Keberangktan" preserves the original typo from the CSV source
df["Total_Keberangktan"] = pd.to_numeric(df["Total_Keberangktan"], errors="coerce").astype("Int64")
df["Total_Penumpang"] = df["Total_Kedatangan"] + df["Total_Keberangktan"]

print("\n=== Descriptive Statistics ===")
print(df.describe())

# ─────────────────────────────────────────────────────────────
# 3. Plot Data
# ─────────────────────────────────────────────────────────────

cols = ["Total_Kedatangan", "Total_Keberangktan", "Total_Penumpang"]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col in zip(axes, cols):
    ax.hist(df[col].dropna(), bins=15, color="steelblue", edgecolor="black")
    ax.set_title(f"Histogram {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Frequency")
plt.tight_layout()
plt.savefig("histogram_features.png", dpi=100)
plt.close()

# Line chart
df["Tanggal"] = pd.to_datetime(
    df["Tahun"].astype(str) + "-" + df["Bulan"].astype(str) + "-01"
)

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(df["Tanggal"], df["Total_Keberangktan"], color="steelblue", lw=2, label="Keberangkatan")
ax.plot(df["Tanggal"], df["Total_Kedatangan"], color="firebrick", lw=2, label="Kedatangan")
ax.set_xlabel("Tanggal")
ax.set_ylabel("Jumlah Penumpang")
ax.set_title("Pergerakan Penumpang Angkutan Udara")
ax.legend()
plt.tight_layout()
plt.savefig("line_chart_keberangkatan_kedatangan.png", dpi=100)
plt.close()

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(df["Tanggal"], df["Total_Penumpang"], color="#128eb7", lw=2)
ax.set_xlabel("Tanggal")
ax.set_ylabel("Jumlah Penumpang")
ax.set_title("Pergerakan Total Penumpang Angkutan Udara")
plt.tight_layout()
plt.savefig("line_chart_total.png", dpi=100)
plt.close()

# Boxplot
fig, ax = plt.subplots(figsize=(5, 7))
ax.boxplot(df["Total_Penumpang"].dropna(), patch_artist=True,
           boxprops=dict(facecolor="steelblue", color="black"))
ax.set_title("Boxplot Total Penumpang")
ax.set_ylabel("Jumlah Penumpang")
plt.tight_layout()
plt.savefig("boxplot_total.png", dpi=100)
plt.close()

# ─────────────────────────────────────────────────────────────
# 4. Simulasi
# ─────────────────────────────────────────────────────────────

rng = np.random.default_rng(42)

# Histogram data asli
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(df["Total_Penumpang"].dropna(), bins=30, color="steelblue", edgecolor="black")
ax.set_title("Total Penumpang (data asli)")
ax.set_xlabel("Total Penumpang")
plt.tight_layout()
plt.savefig("hist_data_asli.png", dpi=100)
plt.close()

# ─────────────────────────────────────────────────────────────
# 5. Uji Asumsi Normalitas
# ─────────────────────────────────────────────────────────────

data = df["Total_Penumpang"].dropna().to_numpy(dtype=float)

stat, p_value = stats.shapiro(data)
print(f"\n=== Shapiro-Wilk Test ===")
print(f"Statistic: {stat:.4f}, p-value: {p_value:.4f}")
print("Gagal tolak H0 (data berdistribusi normal)" if p_value > 0.05 else "Tolak H0")

fig, ax = plt.subplots(figsize=(6, 5))
(osm, osr), (slope, intercept, r) = stats.probplot(data, dist="norm")
ax.scatter(osm, osr, color="steelblue", s=15)
ax.plot(osm, slope * np.array(osm) + intercept, color="red", lw=1.5)
ax.set_title("QQ-Plot Total Penumpang")
ax.set_xlabel("Theoretical Quantiles")
ax.set_ylabel("Sample Quantiles")
plt.tight_layout()
plt.savefig("qqplot.png", dpi=100)
plt.close()

# ─────────────────────────────────────────────────────────────
# 6. Estimasi Parameter (MLE Normal)
# ─────────────────────────────────────────────────────────────

mu_hat, sd_hat = norm.fit(data)
print(f"\n=== MLE Parameter Estimates ===")
print(f"mu_hat  = {mu_hat:.4f}")
print(f"sd_hat  = {sd_hat:.4f}")

# ─────────────────────────────────────────────────────────────
# 7. Simulasi – Inverse-Transform Method
# ─────────────────────────────────────────────────────────────

rng_sim = np.random.default_rng(12345)
sample_sizes = [100, 1000, 10000]
simulated_inv = {}

for n in sample_sizes:
    u = rng_sim.uniform(size=n)
    simulated_inv[n] = norm.ppf(u, loc=mu_hat, scale=sd_hat)

rows_inv = []
for n in sample_sizes:
    s = simulated_inv[n]
    rows_inv.append({
        "Sample_Size": n,
        "Simulated_Mean": s.mean(),
        "Simulated_SD": s.std(ddof=1),
        "Mean_Bias": abs(s.mean() - mu_hat),
        "SD_Bias": abs(s.std(ddof=1) - sd_hat),
    })
results_inv = pd.DataFrame(rows_inv)
print("\n=== Inverse-Transform Results ===")
print(results_inv)

# Bias plot – Inverse-Transform
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(results_inv["Sample_Size"], results_inv["Mean_Bias"], marker="o", color="blue", label="Mean Bias")
ax.plot(results_inv["Sample_Size"], results_inv["SD_Bias"], marker="o", color="red", label="SD Bias")
ax.set_title("Mean and SD Bias vs. Sample Size (Inverse-Transform)")
ax.set_xlabel("Sample Size (n)")
ax.set_ylabel("Absolute Bias")
ax.legend()
plt.tight_layout()
plt.savefig("bias_inverse_transform.png", dpi=100)
plt.close()

# Side-by-side histograms
x_inv = simulated_inv[10000]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.hist(data, bins=10, color="#c97c00", edgecolor="black")
ax1.set_title("Data Asli")
ax2.hist(x_inv, bins=10, color="#128eb7", edgecolor="black")
ax2.set_title("Inverse-Transform")
for ax in (ax1, ax2):
    ax.set_xlabel("Nilai")
plt.tight_layout()
plt.savefig("sidebyside_inverse_transform.png", dpi=100)
plt.close()

# Overlay histogram
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(data, bins=20, density=True, color=(0, 0, 1, 0.4), label="Data Asli")
ax.hist(x_inv, bins=20, density=True, color=(1, 0, 0, 0.4), label="Inverse-Transform")
xs = np.linspace(min(data.min(), x_inv.min()), max(data.max(), x_inv.max()), 300)
ax.plot(xs, stats.gaussian_kde(data)(xs), color="blue", lw=2)
ax.plot(xs, stats.gaussian_kde(x_inv)(xs), color="red", lw=2)
ax.set_title("Overlay Histogram Data Asli & Inverse-Transform")
ax.set_xlabel("Nilai")
ax.legend()
plt.tight_layout()
plt.savefig("overlay_inverse_transform.png", dpi=100)
plt.close()

ks_inv = kstest(x_inv, lambda x: norm.cdf(x, loc=data.mean(), scale=data.std(ddof=1)))
print(f"\nKS-Test Inverse-Transform: statistic={ks_inv.statistic:.4f}, p-value={ks_inv.pvalue:.4f}")

# ─────────────────────────────────────────────────────────────
# 8. Simulasi – Acceptance-Rejection Method (Cauchy proposal)
# ─────────────────────────────────────────────────────────────

def normal_ar(n, rng, c=1.5):
    """Generate n standard-normal samples via Acceptance-Rejection with Cauchy proposal.

    Returns raw N(0,1) samples; the caller is responsible for scaling/shifting
    to the desired mean and standard deviation.
    """
    samples = np.empty(n)
    count = 0
    while count < n:
        x_prop = cauchy.rvs(random_state=rng)
        u = rng.uniform()
        f_x = norm.pdf(x_prop)
        g_x = cauchy.pdf(x_prop)
        if u <= f_x / (c * g_x):
            samples[count] = x_prop
            count += 1
    return samples


rng_ar = np.random.default_rng(12345)
simulated_ar = {}
for n in sample_sizes:
    simulated_ar[n] = mu_hat + sd_hat * normal_ar(n, rng_ar)

rows_ar = []
for n in sample_sizes:
    s = simulated_ar[n]
    rows_ar.append({
        "Sample_Size": n,
        "Simulated_Mean": s.mean(),
        "Simulated_SD": s.std(ddof=1),
        "Mean_Bias": abs(s.mean() - mu_hat),
        "SD_Bias": abs(s.std(ddof=1) - sd_hat),
    })
results_ar = pd.DataFrame(rows_ar)
print("\n=== Acceptance-Rejection Results ===")
print(results_ar)

# Bias plot – Acceptance-Rejection
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(results_ar["Sample_Size"], results_ar["Mean_Bias"], marker="o", color="blue", label="Mean Bias")
ax.plot(results_ar["Sample_Size"], results_ar["SD_Bias"], marker="o", color="red", label="SD Bias")
ax.set_title("Mean and SD Bias vs. Sample Size (Acceptance-Rejection)")
ax.set_xlabel("Sample Size (n)")
ax.set_ylabel("Absolute Bias")
ax.legend()
plt.tight_layout()
plt.savefig("bias_acceptance_rejection.png", dpi=100)
plt.close()

x_ar = simulated_ar[10000]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.hist(data, bins=20, color="#c97c00", edgecolor="black")
ax1.set_title("Data Asli")
ax2.hist(x_ar, bins=20, color="#128eb7", edgecolor="black")
ax2.set_title("Acceptance-Rejection")
for ax in (ax1, ax2):
    ax.set_xlabel("Nilai")
plt.tight_layout()
plt.savefig("sidebyside_acceptance_rejection.png", dpi=100)
plt.close()

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(data, bins=20, density=True, color=(0, 0, 1, 0.4), label="Data Asli")
ax.hist(x_ar, bins=20, density=True, color=(1, 0, 0, 0.4), label="Acceptance-Rejection")
ax.plot(xs, stats.gaussian_kde(data)(xs), color="blue", lw=2)
ax.plot(xs, stats.gaussian_kde(x_ar)(xs), color="red", lw=2)
ax.set_title("Overlay Histogram Data Asli & Acceptance-Rejection")
ax.set_xlabel("Nilai")
ax.legend()
plt.tight_layout()
plt.savefig("overlay_acceptance_rejection.png", dpi=100)
plt.close()

ks_ar = kstest(x_ar, lambda x: norm.cdf(x, loc=data.mean(), scale=data.std(ddof=1)))
print(f"KS-Test Acceptance-Rejection: statistic={ks_ar.statistic:.4f}, p-value={ks_ar.pvalue:.4f}")

# ─────────────────────────────────────────────────────────────
# 9. Simulasi – Box-Muller Transformation Method
# ─────────────────────────────────────────────────────────────

def normal_box_muller(n, mean=0.0, sd=1.0, rng=None):
    """Generate n samples from N(mean, sd) using Box-Muller transform."""
    if rng is None:
        rng = np.random.default_rng()
    m = int(np.ceil(n / 2))
    U1 = rng.uniform(size=m)
    U2 = rng.uniform(size=m)
    R = np.sqrt(-2.0 * np.log(U1))
    theta = 2.0 * np.pi * U2
    Z1 = R * np.cos(theta)
    Z2 = R * np.sin(theta)
    Z = np.concatenate([Z1, Z2])[:n]
    return mean + sd * Z


rng_bm = np.random.default_rng(12345)
simulated_bm = {}
for n in sample_sizes:
    simulated_bm[n] = normal_box_muller(n, mean=mu_hat, sd=sd_hat, rng=rng_bm)

rows_bm = []
for n in sample_sizes:
    s = simulated_bm[n]
    rows_bm.append({
        "Sample_Size": n,
        "Simulated_Mean": s.mean(),
        "Simulated_SD": s.std(ddof=1),
        "Mean_Bias": abs(s.mean() - mu_hat),
        "SD_Bias": abs(s.std(ddof=1) - sd_hat),
    })
results_bm = pd.DataFrame(rows_bm)
print("\n=== Box-Muller Results ===")
print(results_bm)

# Bias plot – Box-Muller
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(results_bm["Sample_Size"], results_bm["Mean_Bias"], marker="o", color="blue", label="Mean Bias")
ax.plot(results_bm["Sample_Size"], results_bm["SD_Bias"], marker="o", color="red", label="SD Bias")
ax.set_title("Mean and SD Bias vs. Sample Size (Box-Muller)")
ax.set_xlabel("Sample Size (n)")
ax.set_ylabel("Absolute Bias")
ax.legend()
plt.tight_layout()
plt.savefig("bias_box_muller.png", dpi=100)
plt.close()

x_box = simulated_bm[10000]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.hist(data, bins=20, color="#c97c00", edgecolor="black")
ax1.set_title("Data Asli")
ax2.hist(x_box, bins=20, color="#128eb7", edgecolor="black")
ax2.set_title("Transformation Method (Box-Muller)")
for ax in (ax1, ax2):
    ax.set_xlabel("Nilai")
plt.tight_layout()
plt.savefig("sidebyside_box_muller.png", dpi=100)
plt.close()

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(data, bins=20, density=True, color=(0, 0, 1, 0.4), label="Data Asli")
ax.hist(x_box, bins=20, density=True, color=(1, 0, 0, 0.4), label="Transformation Method")
ax.plot(xs, stats.gaussian_kde(data)(xs), color="blue", lw=2)
ax.plot(xs, stats.gaussian_kde(x_box)(xs), color="red", lw=2)
ax.set_title("Overlay Histogram Data Asli & Transformation Method")
ax.set_xlabel("Nilai")
ax.legend()
plt.tight_layout()
plt.savefig("overlay_box_muller.png", dpi=100)
plt.close()

ks_bm = kstest(x_box, lambda x: norm.cdf(x, loc=data.mean(), scale=data.std(ddof=1)))
print(f"KS-Test Box-Muller: statistic={ks_bm.statistic:.4f}, p-value={ks_bm.pvalue:.4f}")

# ─────────────────────────────────────────────────────────────
# 10. Perbandingan Ketiga Metode (n=10,000)
# ─────────────────────────────────────────────────────────────

comparison = pd.DataFrame([
    {
        "Method": "Inverse-Transform",
        "Mean_Bias": results_inv.loc[results_inv["Sample_Size"] == 10000, "Mean_Bias"].values[0],
        "SD_Bias": results_inv.loc[results_inv["Sample_Size"] == 10000, "SD_Bias"].values[0],
        "KS_p_value": ks_inv.pvalue,
    },
    {
        "Method": "Acceptance-Rejection",
        "Mean_Bias": results_ar.loc[results_ar["Sample_Size"] == 10000, "Mean_Bias"].values[0],
        "SD_Bias": results_ar.loc[results_ar["Sample_Size"] == 10000, "SD_Bias"].values[0],
        "KS_p_value": ks_ar.pvalue,
    },
    {
        "Method": "Box-Muller",
        "Mean_Bias": results_bm.loc[results_bm["Sample_Size"] == 10000, "Mean_Bias"].values[0],
        "SD_Bias": results_bm.loc[results_bm["Sample_Size"] == 10000, "SD_Bias"].values[0],
        "KS_p_value": ks_bm.pvalue,
    },
])

print("\n=== Comparison of All Methods (n=10000) ===")
print(comparison.to_string(index=False))

# Grouped bar chart
x = np.arange(len(comparison["Method"]))
width = 0.35
fig, ax = plt.subplots(figsize=(10, 5))
bars1 = ax.bar(x - width / 2, comparison["Mean_Bias"], width, color="#c97c00", label="Mean Bias")
bars2 = ax.bar(x + width / 2, comparison["SD_Bias"], width, color="#128eb7", label="SD Bias")
ax.set_xticks(x)
ax.set_xticklabels(comparison["Method"])
ax.set_title("Comparison of Mean and SD Bias by Simulation Method (n=10000)")
ax.set_xlabel("Simulation Method")
ax.set_ylabel("Absolute Bias Value")
ax.legend()
plt.tight_layout()
plt.savefig("comparison_bias_all_methods.png", dpi=100)
plt.close()

print("\nAll plots saved. Script finished.")
