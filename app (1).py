"""
Sistem Pengkelasan Guna Tanah (LULC) - Multi-model ML
Jalankan:  streamlit run app.py --server.maxUploadSize 2000
"""
import glob
import io
import os
import tempfile
import time
import zipfile

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
import rasterio
import streamlit as st
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from rasterio.features import rasterize
from rasterio.mask import mask as rio_mask
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import (
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    accuracy_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SEED = 42
SVM_MAX_SAMPLES = 15000  # SVM perlahan kalau data terlalu banyak


# ------------------------------------------------------------------ helpers
def save_tif(uploaded, folder):
    path = os.path.join(folder, uploaded.name)
    with open(path, "wb") as f:
        f.write(uploaded.getbuffer())
    return path


def read_zipped_shp(uploaded, folder):
    """Upload shapefile sebagai .zip (mengandungi .shp .shx .dbf .prj)."""
    zpath = os.path.join(folder, uploaded.name)
    with open(zpath, "wb") as f:
        f.write(uploaded.getbuffer())
    sub = os.path.join(folder, os.path.splitext(uploaded.name)[0])
    with zipfile.ZipFile(zpath) as z:
        z.extractall(sub)
    shps = glob.glob(os.path.join(sub, "**", "*.shp"), recursive=True)
    if not shps:
        raise ValueError(f"Tiada fail .shp dalam {uploaded.name}")
    return gpd.read_file(shps[0])


def get_models():
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=300, n_jobs=-1, random_state=SEED
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=300, n_jobs=-1, random_state=SEED
        ),
        "Gradient Boosting": HistGradientBoostingClassifier(random_state=SEED),
        "SVM (RBF)": make_pipeline(
            StandardScaler(), SVC(kernel="rbf", C=10, gamma="scale")
        ),
        "KNN": make_pipeline(
            StandardScaler(), KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
        ),
        "Neural Network (MLP)": make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=SEED),
        ),
    }


def rasterize_labels(gdf, col, class_map, shape, transform, crs):
    g = gdf.to_crs(crs)
    shapes = [
        (geom, class_map[v])
        for geom, v in zip(g.geometry, g[col])
        if geom is not None and not geom.is_empty and v in class_map
    ]
    if not shapes:
        return np.zeros(shape, dtype="uint8")
    return rasterize(shapes, out_shape=shape, transform=transform, fill=0, dtype="uint8")


# ----------------------------------------------------------------- pipeline
def run_pipeline(tif_path, boundary, train_gdf, test_gdf, col, outdir, progress=None):
    os.makedirs(outdir, exist_ok=True)
    log = progress or (lambda *_: None)

    # 1) Baca & potong imej mengikut sempadan
    log("Membaca & memotong imej satelit...")
    with rasterio.open(tif_path) as src:
        crs, nod, meta = src.crs, src.nodata, src.meta.copy()
        b = boundary.to_crs(crs)
        arr, transform = rio_mask(src, b.geometry, crop=True, filled=False)
    data = np.ma.getdata(arr).astype("float32")
    invalid = np.ma.getmaskarray(arr).any(axis=0) | ~np.isfinite(data).all(axis=0)
    if nod is not None:
        invalid |= (data == nod).all(axis=0)
    bands, h, w = data.shape

    # 2) Label training / testing
    names = sorted(train_gdf[col].dropna().unique(), key=str)
    class_map = {n: i + 1 for i, n in enumerate(names)}
    n_cls = len(names)
    lab_tr = rasterize_labels(train_gdf, col, class_map, (h, w), transform, crs)
    lab_te = rasterize_labels(test_gdf, col, class_map, (h, w), transform, crs)
    sel_tr = (lab_tr > 0) & ~invalid
    sel_te = (lab_te > 0) & ~invalid & ~sel_tr  # elak pixel bertindih
    X_tr, y_tr = data[:, sel_tr].T, lab_tr[sel_tr]
    X_te, y_te = data[:, sel_te].T, lab_te[sel_te]
    if len(y_tr) == 0 or len(y_te) == 0:
        raise ValueError(
            "Tiada pixel training/testing dalam kawasan imej. "
            "Semak CRS, sempadan dan polygon."
        )

    # 3) Latih & bandingkan model
    rows, fitted = [], {}
    rng = np.random.default_rng(SEED)
    for name, model in get_models().items():
        log(f"Melatih model: {name}...")
        Xf, yf = X_tr, y_tr
        if name.startswith("SVM") and len(y_tr) > SVM_MAX_SAMPLES:
            idx = rng.choice(len(y_tr), SVM_MAX_SAMPLES, replace=False)
            Xf, yf = X_tr[idx], y_tr[idx]
        t0 = time.time()
        model.fit(Xf, yf)
        t_train = time.time() - t0
        pred = model.predict(X_te)
        rows.append(
            {
                "Model": name,
                "Overall Accuracy": accuracy_score(y_te, pred),
                "Kappa": cohen_kappa_score(y_te, pred),
                "F1 (macro)": f1_score(y_te, pred, average="macro", zero_division=0),
                "Masa latih (s)": round(t_train, 1),
            }
        )
        fitted[name] = model
    comp = pd.DataFrame(rows).sort_values("Kappa", ascending=False).reset_index(drop=True)
    best_name = comp.loc[0, "Model"]
    best = fitted[best_name]
    log(f"Model terbaik: {best_name}. Menghasilkan peta LULC...")

    # 4) Ramal keseluruhan imej (ikut chunk)
    flat = data.reshape(bands, -1)
    valid_idx = np.flatnonzero(~invalid.ravel())
    pred_all = np.zeros(h * w, dtype="uint8")
    for i in range(0, len(valid_idx), 200_000):
        idx = valid_idx[i : i + 200_000]
        pred_all[idx] = best.predict(flat[:, idx].T).astype("uint8")
    lulc = pred_all.reshape(h, w)

    tif_out = os.path.join(outdir, "LULC_classified.tif")
    meta.update(
        driver="GTiff", count=1, dtype="uint8", nodata=0,
        height=h, width=w, transform=transform, compress="lzw",
    )
    with rasterio.open(tif_out, "w", **meta) as dst:
        dst.write(lulc, 1)

    # 5) Validation assessment (model terbaik)
    labels = list(range(1, n_cls + 1))
    y_pred = best.predict(X_te)
    cm = confusion_matrix(y_te, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"Ref_{n}" for n in names],
                         columns=[f"Pred_{n}" for n in names])
    rep = pd.DataFrame(
        classification_report(y_te, y_pred, labels=labels, target_names=[str(n) for n in names],
                              output_dict=True, zero_division=0)
    ).T
    with np.errstate(divide="ignore", invalid="ignore"):
        prod = np.diag(cm) / cm.sum(axis=1)  # producer's accuracy
        user = np.diag(cm) / cm.sum(axis=0)  # user's accuracy
    acc_df = pd.DataFrame(
        {"Kelas": names, "Producer_Accuracy": prod, "User_Accuracy": user,
         "Pixel_Test": cm.sum(axis=1)}
    )
    summary = pd.DataFrame(
        {"Metrik": ["Model terbaik", "Overall Accuracy", "Kappa", "F1 (macro)",
                    "Pixel training", "Pixel testing"],
         "Nilai": [best_name, accuracy_score(y_te, y_pred),
                   cohen_kappa_score(y_te, y_pred),
                   f1_score(y_te, y_pred, average="macro", zero_division=0),
                   len(y_tr), len(y_te)]}
    )

    # 6) Statistik luas
    counts = np.bincount(lulc.ravel(), minlength=n_cls + 1)[1:]
    area = pd.DataFrame({"Kelas": names, "Kod": labels, "Pixel": counts})
    if crs is not None and crs.is_projected:
        area["Luas_ha"] = counts * abs(transform.a * transform.e) / 10000
        area["Peratus"] = area["Luas_ha"] / area["Luas_ha"].sum() * 100
    else:
        area["Peratus"] = counts / counts.sum() * 100  # CRS geografik: tiada luas tepat

    # 7) CSV
    comp.to_csv(os.path.join(outdir, "01_model_comparison.csv"), index=False)
    summary.to_csv(os.path.join(outdir, "02_validation_summary.csv"), index=False)
    cm_df.to_csv(os.path.join(outdir, "03_confusion_matrix.csv"))
    acc_df.to_csv(os.path.join(outdir, "04_per_class_accuracy.csv"), index=False)
    rep.to_csv(os.path.join(outdir, "05_classification_report.csv"))
    area.to_csv(os.path.join(outdir, "06_area_per_class.csv"), index=False)
    pd.DataFrame({"Kod": labels, "Kelas": names}).to_csv(
        os.path.join(outdir, "07_class_mapping.csv"), index=False)
    inv = {v: k for k, v in class_map.items()}
    pd.DataFrame({"Rujukan": [inv[i] for i in y_te],
                  "Ramalan": [inv[i] for i in y_pred]}).to_csv(
        os.path.join(outdir, "08_test_predictions.csv"), index=False)

    # 8) Graf
    figs = {}
    colors = [plt.cm.tab20(i % 20) for i in range(n_cls)]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(comp)); wd = 0.27
    ax.bar(x - wd, comp["Overall Accuracy"], wd, label="Overall Accuracy")
    ax.bar(x, comp["Kappa"], wd, label="Kappa")
    ax.bar(x + wd, comp["F1 (macro)"], wd, label="F1 macro")
    ax.set_xticks(x); ax.set_xticklabels(comp["Model"], rotation=25, ha="right")
    ax.set_ylim(0, 1.05); ax.set_title("Perbandingan Model"); ax.legend()
    fig.tight_layout(); figs["model_comparison"] = fig

    fig, ax = plt.subplots(figsize=(max(6, n_cls), max(5, n_cls * 0.8)))
    cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n_cls)); ax.set_yticks(range(n_cls))
    ax.set_xticklabels(names, rotation=45, ha="right"); ax.set_yticklabels(names)
    for i in range(n_cls):
        for j in range(n_cls):
            ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.5 else "black", fontsize=8)
    ax.set_xlabel("Ramalan"); ax.set_ylabel("Rujukan")
    ax.set_title(f"Confusion Matrix (ternormal) - {best_name}")
    fig.colorbar(im, ax=ax, fraction=0.046); fig.tight_layout()
    figs["confusion_matrix"] = fig

    fig, ax = plt.subplots(figsize=(8, 5))
    ycol = "Luas_ha" if "Luas_ha" in area else "Pixel"
    ax.bar([str(n) for n in names], area[ycol], color=colors)
    ax.set_ylabel(ycol); ax.set_title("Luas setiap kelas LULC")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right"); fig.tight_layout()
    figs["area_per_class"] = fig

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(np.ma.masked_equal(lulc, 0), cmap=ListedColormap(colors),
              vmin=0.5, vmax=n_cls + 0.5, interpolation="nearest")
    ax.legend(handles=[Patch(color=colors[i], label=str(n)) for i, n in enumerate(names)],
              loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set_axis_off(); ax.set_title(f"Peta LULC - {best_name}")
    fig.tight_layout(); figs["lulc_map"] = fig

    for k, f in figs.items():
        f.savefig(os.path.join(outdir, f"graph_{k}.png"), dpi=200, bbox_inches="tight")

    # 9) Zip semua output
    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in glob.glob(os.path.join(outdir, "*")):
            z.write(p, os.path.basename(p))

    return {"best": best_name, "comp": comp, "summary": summary, "cm": cm_df,
            "acc": acc_df, "area": area, "figs": figs, "tif": tif_out,
            "zip": zbuf.getvalue()}


# ---------------------------------------------------------------------- UI
st.set_page_config(page_title="Sistem LULC", layout="wide")
st.title("Sistem Pengkelasan Guna Tanah (LULC)")
st.caption("Latih beberapa model ML, bandingkan, dan guna model terbaik untuk peta akhir.")

c1, c2 = st.columns(2)
with c1:
    f_tif = st.file_uploader("1) Imej satelit (GeoTIFF, multi-band)", type=["tif", "tiff"])
    f_bnd = st.file_uploader("2) Sempadan (shapefile .zip)", type=["zip"])
with c2:
    f_trn = st.file_uploader("3) Polygon training (shapefile .zip)", type=["zip"])
    f_tst = st.file_uploader("4) Polygon testing (shapefile .zip)", type=["zip"])

if f_trn is not None:
    tmp_cols = tempfile.mkdtemp()
    cols = [c for c in read_zipped_shp(f_trn, tmp_cols).columns if c != "geometry"]
    class_col = st.selectbox("Kolum kelas guna tanah (dalam training & testing)", cols)
else:
    class_col = None

if st.button("Jalankan klasifikasi", type="primary",
             disabled=not all([f_tif, f_bnd, f_trn, f_tst, class_col])):
    work = tempfile.mkdtemp()
    try:
        status = st.empty()
        tif_path = save_tif(f_tif, work)
        bnd, trn, tst = (read_zipped_shp(f, work) for f in (f_bnd, f_trn, f_tst))
        if class_col not in tst.columns:
            st.error(f"Kolum '{class_col}' tiada dalam shapefile testing.")
        else:
            st.session_state["res"] = run_pipeline(
                tif_path, bnd, trn, tst, class_col, os.path.join(work, "out"),
                progress=lambda m: status.info(m),
            )
            status.success("Siap!")
    except Exception as e:  # noqa: BLE001
        st.error(f"Ralat: {e}")

res = st.session_state.get("res")
if res:
    st.subheader(f"Model terbaik: {res['best']}")
    a, b = st.columns(2)
    a.markdown("**Perbandingan model** (disusun ikut Kappa)")
    a.dataframe(res["comp"], width="stretch")
    b.markdown("**Ringkasan validation**")
    b.dataframe(res["summary"].astype(str), width="stretch")
    st.markdown("**Producer's / User's accuracy setiap kelas**")
    st.dataframe(res["acc"], width="stretch")
    st.markdown("**Luas setiap kelas**")
    st.dataframe(res["area"], width="stretch")
    g1, g2 = st.columns(2)
    g1.pyplot(res["figs"]["lulc_map"]); g2.pyplot(res["figs"]["model_comparison"])
    g1.pyplot(res["figs"]["confusion_matrix"]); g2.pyplot(res["figs"]["area_per_class"])
    d1, d2 = st.columns(2)
    with open(res["tif"], "rb") as fh:
        d1.download_button("Muat turun raster LULC (.tif)", fh.read(), "LULC_classified.tif")
    d2.download_button("Muat turun SEMUA (zip: tif, csv, graf)", res["zip"], "LULC_output.zip")
