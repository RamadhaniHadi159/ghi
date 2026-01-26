import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from scipy.interpolate import griddata
from matplotlib.colors import BoundaryNorm
import os

# ===============================
# KONFIGURASI
# ===============================
plt.rcParams["savefig.dpi"] = 1000
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

USE_LOG = False
N_LEVELS = 15
RES_PLOT = 0.01

# ===============================
# SHAPEFILE
# ===============================
shapefile = "/data/mahasiswa/ramadhani/ecmwf/datawilayah/gadm41_IDN_2.shp"
gdf = gpd.read_file(shapefile).to_crs(epsg=4326)

jatim = gdf[gdf["NAME_1"] == "Jawa Timur"]
wilayah_bg = gdf[gdf["NAME_1"].isin(["Jawa Timur", "Jawa Tengah", "Bali"])]

# ===============================
# KABUPATEN JAWA TIMUR (TANPA KOTA)
# ===============================
kabkot_jatim = jatim[~jatim["NAME_2"].str.startswith("Kota ")].copy()

kabkot_jatim["LABEL_NAME"] = (
    kabkot_jatim["NAME_2"]
    .str.replace("Kabupaten", "", regex=False)
    .str.replace("Kota", "", regex=False)
    .str.title()
)

# ===============================
# TITIK LABEL AMAN
# ===============================
def safe_label_point(geom):
    c = geom.centroid
    return c if geom.contains(c) else geom.representative_point()

kabkot_jatim["label_point"] = kabkot_jatim.geometry.apply(safe_label_point)

# ===============================
# OFFSET MANUAL LABEL KABUPATEN
# ===============================
kab_label_override = {
    "Bangkalan":    {"dx": 0.00, "dy": 0.00},
    "Banyuwangi":   {"dx": 0.00, "dy": 0.00},
    "Blitar":       {"dx": 0.00, "dy": 0.00},
    "Bojonegoro":   {"dx": 0.00, "dy": 0.00},
    "Bondowoso":    {"dx": 0.00, "dy": 0.00},
    "Gresik":       {"dx": 0.02, "dy": -0.06},
    "Jember":       {"dx": 0.00, "dy": 0.00},
    "Jombang":      {"dx": 0.03, "dy": -0.05},
    "Kediri":       {"dx": -0.04, "dy": 0.02},
    "Lamongan":     {"dx": 0.00, "dy": 0.03},
    "Lumajang":     {"dx": 0.00, "dy": 0.00},
    "Madiun":       {"dx": 0.00, "dy": -0.01},
    "Magetan":      {"dx": 0.00, "dy": -0.02},
    "Malang":       {"dx": 0.06, "dy": -0.04},
    "Mojokerto":    {"dx": -0.04, "dy": 0.06},
    "Nganjuk":      {"dx": 0.00, "dy": 0.05},
    "Ngawi":        {"dx": -0.03, "dy": 0.00},
    "Pacitan":      {"dx": 0.00, "dy": 0.00},
    "Pamekasan":    {"dx": 0.00, "dy": 0.00},
    "Pasuruan":     {"dx": 0.05, "dy": 0.02},
    "Ponorogo":     {"dx": 0.00, "dy": 0.00},
    "Probolinggo":  {"dx": 0.04, "dy": -0.02},
    "Sampang":      {"dx": 0.00, "dy": -0.08},
    "Sidoarjo":     {"dx": -0.05, "dy": -0.05},
    "Situbondo":    {"dx": 0.00, "dy": 0.00},
    "Sumenep":      {"dx": 0.00, "dy": 0.00},
    "Trenggalek":   {"dx": 0.00, "dy": 0.00},
    "Tuban":        {"dx": 0.00, "dy": 0.00},
    "Tulungagung":  {"dx": 0.00, "dy": 0.05},
}

# ===============================
# DATA GHI
# ===============================
df = pd.read_csv(
    "/data/mahasiswa/ramadhani/ecmwf/datafix/ghi/data/baru/dataghifix.txt",
    sep=r"\s+"
)

df["date"] = pd.to_datetime(df["date"])
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month

def periode_3bulan(bulan):
    if bulan in [12, 1, 2]:
        return "Des-Feb"
    elif bulan in [3, 4, 5]:
        return "Mar-May"
    elif bulan in [6, 7, 8]:
        return "Jun-Aug"
    else:
        return "Sep-Nov"

df["periode"] = df["month"].apply(periode_3bulan)

# ===============================
# SKALA WARNA
# ===============================
if USE_LOG:
    valid = df["ssrd_wm2"][df["ssrd_wm2"] > 0]
    GLOBAL_VMIN = np.log10(valid.min())
    GLOBAL_VMAX = np.log10(valid.max())
    cbar_label = "log10(GHI) [log(W m⁻²)]"
else:
    GLOBAL_VMIN = df["ssrd_wm2"].min()
    GLOBAL_VMAX = df["ssrd_wm2"].max()
    cbar_label = "GHI (W m⁻²)"

levels = np.linspace(GLOBAL_VMIN, GLOBAL_VMAX, N_LEVELS + 1)
norm = BoundaryNorm(levels, ncolors=256)

# ===============================
# OUTPUT
# ===============================
output_dir = "/data/mahasiswa/ramadhani/github/hasil/peta_jatim1"
os.makedirs(output_dir, exist_ok=True)

# ===============================
# LABEL PROVINSI
# ===============================
jateng_bali = wilayah_bg[wilayah_bg["NAME_1"].isin(["Jawa Tengah", "Bali"])]
jateng_bali_prov = jateng_bali.dissolve(by="NAME_1").reset_index()
jateng_bali_prov["label_point"] = jateng_bali_prov.geometry.representative_point()

label_prov_override = {
    "Jawa Tengah": {"dx": 0.75, "dy": 0.05},
    "Bali": {"dx": -0.15, "dy": 0.10},
}

# ===============================
# LOOP PLOT
# ===============================
for (tahun, periode), grup in df.groupby(["year", "periode"]):

    rata = grup.groupby(["latitude", "longitude"])["ssrd_wm2"].mean().reset_index()
    if rata.empty:
        continue

    nilai = np.log10(rata["ssrd_wm2"]) if USE_LOG else rata["ssrd_wm2"]

    minx, miny, maxx, maxy = jatim.total_bounds

    xi = np.arange(minx, maxx + RES_PLOT, RES_PLOT)
    yi = np.arange(miny, maxy + RES_PLOT, RES_PLOT)
    xi, yi = np.meshgrid(xi, yi)

    zi = griddata(
        (rata["longitude"], rata["latitude"]),
        nilai,
        (xi, yi),
        method="linear"
    )

    grid_points = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(xi.ravel(), yi.ravel()),
        crs="EPSG:4326"
    )

    mask = grid_points.within(jatim.geometry.union_all()).values
    zi[~mask.reshape(zi.shape)] = np.nan

    # ===============================
    # PLOT
    # ===============================
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_facecolor("#cfe8f3")

    jateng_bali_prov.plot(ax=ax, color="white", edgecolor="black", linewidth=0.6, zorder=1)

    cs = ax.contourf(
        xi, yi, zi,
        levels=levels,
        cmap="magma",
        norm=norm,
        zorder=2
    )

    kabkot_jatim.boundary.plot(ax=ax, color="black", linewidth=0.3, zorder=3)
    jatim.boundary.plot(ax=ax, color="black", linewidth=0.5, zorder=4)

    # ===============================
    # LABEL KABUPATEN
    # ===============================
    for _, row in kabkot_jatim.iterrows():
        x = row["label_point"].x
        y = row["label_point"].y
        name = row["LABEL_NAME"]

        if name in kab_label_override:
            x += kab_label_override[name]["dx"]
            y += kab_label_override[name]["dy"]

        ax.text(
            x, y, name,
            fontsize=5.5,
            ha="center",
            va="center",
            color="black",
            path_effects=[pe.withStroke(linewidth=1.2, foreground="white")],
            zorder=5
        )

    # ===============================
    # LABEL PROVINSI
    # ===============================
    for _, row in jateng_bali_prov.iterrows():
        x = row["label_point"].x
        y = row["label_point"].y

        if row["NAME_1"] in label_prov_override:
            x += label_prov_override[row["NAME_1"]]["dx"]
            y += label_prov_override[row["NAME_1"]]["dy"]

        ax.text(
            x, y, row["NAME_1"],
            fontsize=9,
            fontweight="bold",
            ha="center",
            va="center",
            color="black",
            path_effects=[pe.withStroke(linewidth=1.5, foreground="white")],
            zorder=6
        )

    # ===============================
    # FOKUS PETA (FINAL & NORMAL)
    # ===============================
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)

    ax.set_xticks(np.arange(110, 117, 1))
    ax.set_yticks(np.arange(-9, -4, 1))
    ax.grid(True, linestyle="--", linewidth=0.2, alpha=0.6)

    # ===============================
    # COLORBAR
    # ===============================
    cbar = fig.colorbar(cs, ax=ax, shrink=0.85, ticks=levels)
    cbar.set_label(cbar_label, fontsize=10)
    cbar.ax.tick_params(labelsize=8)
    cbar.ax.ticklabel_format(style="plain", useOffset=False)

    ax.set_title(f"Peta GHI Jawa Timur - {periode} {tahun}", fontsize=12)
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")

    plt.savefig(
        f"{output_dir}/peta_ghi_jatim_{tahun}_{periode}.pdf",
        bbox_inches="tight"
    )
    plt.close()

print("✅ Peta GHI Jawa Timur berhasil dibuat (FINAL & NORMAL)")
