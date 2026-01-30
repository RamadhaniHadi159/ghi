import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# ============================
# 1. Load dataset
# ============================
data = pd.read_csv("/data/mahasiswa/ramadhani/ecmwf/datafix/aodspwvo3/data/datapendukungfix.txt", sep="\t")

# ============================
# 2. Pilih parameter untuk korelasi
# ============================
params = ["t2m", "d2m", "sp", "aod550", "α", "β", "tcno2", "tcwv"]
df = data[params]

# ============================
# 3. Hitung correlation matrix
# ============================
correlation_matrix = df.corr()

# ============================
# 4. Plot heatmap
# ============================
plt.figure(figsize=(10, 8))
sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap='coolwarm',
    fmt=".2f",
    linewidths=0.5,
    square=True
)

plt.title("Correlation Heatmap")
plt.show()

plt.savefig("/data/mahasiswa/ramadhani/github/hasil/grafik_korelasi_parameter_pendukung.pdf",
            bbox_inches="tight"
)
plt.show()

print("Grafik korelasi parameter pendukung selesai dan tersimpan.")