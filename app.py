import os
import json
import time
from datetime import datetime, timezone

import numpy as np
import streamlit as st
import mne
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.collections
from scipy.signal import decimate
from pymongo import MongoClient
from pymongo.server_api import ServerApi

# ====================
# PAGE CONFIGURATION
# ====================
st.set_page_config(page_title="EEG Dashboard", layout="wide")
st.markdown("<style>section.main{overflow-anchor: none;}</style>", unsafe_allow_html=True)

BANDS = {"Delta": (0.5, 4), "Theta": (4, 8), "Alpha": (8, 13), "Beta": (13, 30), "Gamma": (30, 45)}
BAND_ORDER = ["Delta", "Theta", "Alpha", "Beta", "Gamma", "Broadband"]
CANVAS_POINT_BUDGET = 3000  # points per channel sent to the browser, regardless of recording length

BAND_DESCRIPTIONS = {
    "Delta": "slow, high-amplitude activity generally associated with deep rest or drowsiness",
    "Theta": "activity often linked to a light, meditative, or drowsy state",
    "Alpha": "activity generally associated with a calm, relaxed, eyes-closed-type state",
    "Beta": "activity often associated with active thinking, alertness, or mild tension",
    "Gamma": "activity sometimes linked to high-level cognitive processing or focus",
}

# ====================
# MONGODB CONNECTION
# ====================
def _get_secret_or_env(key, default):
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.environ.get(key, default)

MONGO_URI = _get_secret_or_env(
    "MONGO_URI",
    "mongodb+srv://test:REPLACE_ME_WITH_LOCAL_PASSWORD@cluster0.1m2rwpo.mongodb.net/?appName=Cluster0"
)
MONGO_DB_NAME = _get_secret_or_env("MONGO_DB", "eeg_dataset_server")
MONGO_COLLECTION = "datasets"

# Guessed document field names
CHANNEL_KEYS = ["channels", "channel_names", "ch_names"]
SFREQ_KEYS = ["sampling_rate", "sfreq", "fs", "sample_rate"]
DATA_KEYS = ["data", "eeg_data", "samples", "values"]
ORDER_KEYS = ["chunk_index", "sequence", "sequence_index", "part", "part_number", "order"]
PATIENT_KEYS = ["patient_id", "patient", "subject_id", "session_id"]


@st.cache_resource(show_spinner=False)
def get_mongo_client():
    try:
        client = MongoClient(MONGO_URI, server_api=ServerApi("1"), serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
        return client
    except Exception:
        return None


@st.cache_data(ttl=30, show_spinner="Fetching EEG data from MongoDB...")
def fetch_dataset_documents():
    client = get_mongo_client()
    if client is None:
        return None
    try:
        coll = client[MONGO_DB_NAME][MONGO_COLLECTION]
        docs = list(coll.find())
        return docs if docs else None
    except Exception:
        return None


def _first_present(doc, candidate_keys):
    for k in candidate_keys:
        if k in doc and doc[k] is not None:
            return doc[k]
    return None


def build_raw_from_documents(docs):
    """
    Stitches every document into one continuous (channels, samples) array,
    in sequence — e.g. cluster0's chunk 1, then chunk 2, etc. Multiple
    files are treated as one patient's split recording, not separate
    datasets to pick between.
    """
    if not docs:
        return None, None, None, 0

    def sort_key(d):
        for k in ORDER_KEYS:
            if k in d:
                return d[k]
        return d.get("_id")  # falls back to Mongo's chronological insert order

    docs = sorted(docs, key=sort_key)

    ch_names, sfreq, chunks = None, None, []
    for d in docs:
        names = _first_present(d, CHANNEL_KEYS)
        fs = _first_present(d, SFREQ_KEYS)
        raw_vals = _first_present(d, DATA_KEYS)
        if raw_vals is None:
            continue
        arr = np.array(raw_vals, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if names and arr.shape[0] != len(names) and arr.ndim == 2 and arr.shape[1] == len(names):
            arr = arr.T  # normalize to (channels, samples)
        chunks.append(arr)
        ch_names = ch_names or names
        sfreq = sfreq or fs

    if not chunks or ch_names is None or sfreq is None:
        return None, None, None, 0

    try:
        full = np.concatenate(chunks, axis=1)
    except ValueError:
        # channel counts didn't match across files — use the smallest common count
        min_ch = min(c.shape[0] for c in chunks)
        full = np.concatenate([c[:min_ch] for c in chunks], axis=1)
        ch_names = ch_names[:min_ch]

    return full, list(ch_names), float(sfreq), len(chunks)


# =================================
# CHANNEL NAMING / MONTAGE HELPERS
# =================================
@st.cache_resource
def get_fallback_pool():
    return mne.channels.make_standard_montage("standard_1005").ch_names


def pick_channel_names(n_channels):
    exact = {16: "biosemi16", 32: "biosemi32", 64: "biosemi64"}
    if n_channels in exact:
        return mne.channels.make_standard_montage(exact[n_channels]).ch_names
    pool = get_fallback_pool()
    if n_channels <= len(pool):
        return pool[:n_channels]
    return pool + [f"CH{i}" for i in range(len(pool), n_channels)]


def classify_region(ch):
    c = ch.upper()
    if c.startswith("FP") or c.startswith("AF"):
        return "Frontal"
    if c.startswith("FC"):
        return "Central"
    if c.startswith("FT"):
        return "Temporal"
    if c.startswith("F"):
        return "Frontal"
    if c.startswith("CP"):
        return "Parietal"
    if c.startswith("TP"):
        return "Temporal"
    if c.startswith("C"):
        return "Central"
    if c.startswith("PO"):
        return "Occipital"
    if c.startswith("P"):
        return "Parietal"
    if c.startswith("T"):
        return "Temporal"
    if c.startswith("O") or c == "IZ":
        return "Occipital"
    return "Other"


@st.cache_resource
def get_montage():
    return mne.channels.make_standard_montage("standard_1005")


@st.cache_resource
def build_info(ch_names, sfreq):
    info = mne.create_info(ch_names=list(ch_names), sfreq=sfreq, ch_types="eeg")
    info.set_montage(get_montage(), on_missing="ignore")
    return info


@st.cache_resource
def load_mock_data(n_channels, sfreq, duration_sec=60):
    ch_names = pick_channel_names(n_channels)
    n_samples = duration_sec * sfreq
    t = np.arange(n_samples) / sfreq
    rng = np.random.default_rng(42)
    data_uv = np.zeros((n_channels, n_samples))
    for i in range(n_channels):
        sig = np.zeros(n_samples)
        for freq, base_amp in [(2, 8), (6, 5), (10, 6), (20, 3), (40, 2)]:
            amp = base_amp * rng.uniform(0.1, 1.5)
            sig += amp * np.sin(2 * np.pi * freq * t + rng.uniform(0, 2 * np.pi))
        pink = np.cumsum(rng.normal(0, 1, n_samples))
        pink = (pink - pink.mean()) / pink.std() * 4
        data_uv[i] = sig + pink
    return data_uv, ch_names, float(sfreq)


def downsample_for_display(data_uv, target_points=CANVAS_POINT_BUDGET):
    """Anti-aliased downsample of a (channels, samples) µV array so the
    browser payload stays a bounded size no matter how long the stitched
    recording is."""
    out = data_uv
    while out.shape[1] > target_points * 2:
        out = decimate(out, 2, axis=1, zero_phase=True)
    if out.shape[1] > target_points:
        idx = np.linspace(0, out.shape[1] - 1, target_points).astype(int)
        out = out[:, idx]
    return out


# ========
# HEADER
# ========
st.title("EEG Monitoring Dashboard")

if "session_start" not in st.session_state:
    st.session_state.session_start = time.time()
if "refresh_count" not in st.session_state:
    st.session_state.refresh_count = 0
if "regions_seen" not in st.session_state:
    st.session_state.regions_seen = set()
st.session_state.refresh_count += 1

# ======================
# SIDEBAR — DATA SOURCE
# ======================
st.sidebar.header("Data Source")
docs = fetch_dataset_documents()
mongo_data, mongo_ch_names, mongo_sfreq, n_files = build_raw_from_documents(docs)
mongo_available = mongo_data is not None

source_choice = st.sidebar.radio(
    "Source", ["Auto (MongoDB, fallback to mock)", "MongoDB only", "Mock data only"], index=0
)

use_mongo = mongo_available and source_choice != "Mock data only"
if source_choice == "MongoDB only" and not mongo_available:
    st.sidebar.error("MongoDB has no usable documents right now — check the debug panel below.")

with st.sidebar.expander("🔍 Debug: raw document fields"):
    if docs:
        preview = {k: v for k, v in docs[0].items() if k != "_id"}
        st.json(preview, expanded=False)
        st.caption(
            "If channel/rate/data field names above don't match "
            "CHANNEL_KEYS/SFREQ_KEYS/DATA_KEYS near the top of app.py, add them there."
        )
    elif get_mongo_client() is None:
        st.caption("Couldn't connect to MongoDB (network/credentials). Using mock data.")
    else:
        st.caption("Connected, but the 'datasets' collection is empty.")

if use_mongo:
    RAW_DATA = mongo_data * 1e-6 if np.nanmax(np.abs(mongo_data)) > 1 else mongo_data  # normalize to volts
    CH_NAMES = mongo_ch_names
    sfreq = mongo_sfreq
    st.sidebar.success(f"MongoDB: {n_files} file(s) stitched · {len(CH_NAMES)} channels · {sfreq:.0f} Hz")
else:
    st.sidebar.header("Mock Data Settings")
    n_channels = st.sidebar.number_input("Number of channels", min_value=4, max_value=256, value=64, step=1)
    sfreq = st.sidebar.number_input("Sampling rate (Hz)", min_value=32, max_value=1024, value=160, step=8)
    mock_data_uv, CH_NAMES, sfreq = load_mock_data(n_channels, sfreq)
    RAW_DATA = mock_data_uv * 1e-6

TOTAL_SAMPLES = RAW_DATA.shape[1]
FALLBACK_POOL = get_fallback_pool()
RECOGNIZED = [ch for ch in CH_NAMES if ch in FALLBACK_POOL]
CUSTOM_CHANNELS = [ch for ch in CH_NAMES if ch not in FALLBACK_POOL]
INFO = build_info(tuple(CH_NAMES), sfreq)

st.sidebar.header("Display Controls")
window_sec = st.sidebar.slider("Sweep / analysis window (seconds)", 2, 15, 6)
topo_refresh = st.sidebar.slider("Topomap refresh interval (s)", 0.5, 3.0, 1.0, 0.5)

region_map = {}
for ch in CH_NAMES:
    region_map.setdefault(classify_region(ch), []).append(ch)
region_options = [r for r in ["Frontal", "Central", "Temporal", "Parietal", "Occipital", "Other"] if r in region_map]
region_choice = st.sidebar.multiselect("Brain regions to display", options=region_options, default=region_options)
st.session_state.regions_seen |= set(region_choice)

selected_channels = [ch for ch in CH_NAMES if classify_region(ch) in region_choice] or CH_NAMES
st.divider()

col_chart, col_topo = st.columns([3, 2])

# ------------------------------------------------------------
# LEFT: sweep chart, HTML5 canvas + requestAnimationFrame, now driven
# by real (downsampled) sample values instead of a JS-side formula.
# ------------------------------------------------------------
def render_sweep_monitor(channels, data_uv, sfreq_effective, window_sec, height_per_channel=34):
    total_height = max(200, len(channels) * height_per_channel)
    payload = json.dumps({
        "channels": channels,
        "windowSec": window_sec,
        "dt": 1.0 / sfreq_effective,
        "data": data_uv.tolist(),
        "totalPoints": data_uv.shape[1],
    })

    html = f"""
    <div style="background:#ffffff;border:1px solid #e0e0e0;border-radius:8px;padding:6px;">
      <canvas id="eegCanvas" style="width:100%;display:block;"></canvas>
    </div>
    <script>
    (function() {{
        const cfg = {payload};
        const channels = cfg.channels;
        const windowSec = cfg.windowSec;
        const canvas = document.getElementById("eegCanvas");
        const ctx = canvas.getContext("2d");
        const dpr = window.devicePixelRatio || 1;

        function resize() {{
            const w = canvas.clientWidth || canvas.parentElement.clientWidth;
            const h = {total_height};
            canvas.style.height = h + "px";
            canvas.width = w * dpr;
            canvas.height = h * dpr;
            ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        }}
        window.addEventListener("resize", resize);
        resize();

        const totalDur = cfg.dt * (cfg.totalPoints - 1);

        // Real-data lookup with linear interpolation between samples —
        // loops the actual stitched recording instead of drawing a formula.
        function valueAt(chIndex, tSec) {{
            let tt = tSec % totalDur;
            if (tt < 0) tt += totalDur;
            const idxF = tt / cfg.dt;
            const i0 = Math.floor(idxF);
            const i1 = Math.min(i0 + 1, cfg.totalPoints - 1);
            const frac = idxF - i0;
            const arr = cfg.data[chIndex];
            return arr[i0] + (arr[i1] - arr[i0]) * frac;
        }}

        const POINTS = 300;
        const startT = performance.now() / 1000;

        function draw(tsMs) {{
            const nowSec = tsMs / 1000 - startT;
            const w = canvas.clientWidth || canvas.parentElement.clientWidth;
            const h = {total_height};
            const laneH = h / channels.length;

            ctx.fillStyle = "#ffffff";
            ctx.fillRect(0, 0, w, h);

            channels.forEach((name, i) => {{
                const laneY = i * laneH;
                const midY = laneY + laneH / 2;
                const amp = laneH * 0.42;

                ctx.strokeStyle = "rgba(0,0,0,0.08)";
                ctx.beginPath(); ctx.moveTo(0, laneY); ctx.lineTo(w, laneY); ctx.stroke();

                ctx.strokeStyle = "#000000";
                ctx.lineWidth = 1.2;
                ctx.beginPath();
                for (let p = 0; p <= POINTS; p++) {{
                    const frac = p / POINTS;
                    const x = frac * w;
                    const tAtPoint = nowSec - windowSec * (1 - frac);
                    const val = valueAt(i, tAtPoint) * 1e6; // volts -> µV
                    const y = midY - (val / 25) * amp;
                    if (p === 0) {{ ctx.moveTo(x, y); }} else {{ ctx.lineTo(x, y); }}
                }}
                ctx.stroke();

                ctx.fillStyle = "#333333";
                ctx.font = "11px monospace";
                ctx.fillText(name, 6, laneY + 13);
            }});

            requestAnimationFrame(draw);
        }}
        requestAnimationFrame(draw);
    }})();
    </script>
    """
    st.iframe(html, height=total_height + 20)


with col_chart:
    st.subheader("Live Multi-Channel EEG Signal")
    st.caption(
        f"{len(selected_channels)} channels · {sfreq:.0f} Hz · {window_sec}s window · "
        f"{' + '.join(region_choice) if region_choice else 'All regions'} · "
        f"{'MongoDB (live)' if use_mongo else 'Mock data'}"
    )
    sel_idx = [CH_NAMES.index(ch) for ch in selected_channels]
    display_data = downsample_for_display(RAW_DATA[sel_idx])
    display_sfreq = sfreq * display_data.shape[1] / RAW_DATA.shape[1]
    render_sweep_monitor(selected_channels, display_data, display_sfreq, window_sec)

# ----------
# TOPOMAPS
# ----------
if "topo_pos" not in st.session_state:
    st.session_state.topo_pos = 0


def band_power(chan_data_uv, sfreq, lo, hi):
    min_sec = max(2.0, 3.0 / lo)
    if chan_data_uv.shape[1] < min_sec * sfreq:
        return None
    filtered = mne.filter.filter_data(chan_data_uv, sfreq, lo, hi, verbose=False)
    return np.mean(filtered ** 2, axis=1)


with col_topo:
    @st.fragment(run_every=topo_refresh)
    def live_topomaps():
        st.session_state.topo_pos = (st.session_state.topo_pos + int(sfreq)) % TOTAL_SAMPLES
        pos = st.session_state.topo_pos
        window_samples = int(window_sec * sfreq)
        idx = np.arange(pos - window_samples, pos) % TOTAL_SAMPLES

        st.subheader("Topographic Maps — All Bands")
        st.caption("Hover a topomap and click the ⤢ icon in its top-right corner to enlarge it.")
        if not RECOGNIZED:
            st.warning("None of the current channels match a known electrode montage, so no topomap can be drawn. See the Session Report below for a signal summary instead.")
            return

        recognized_idx = [CH_NAMES.index(ch) for ch in RECOGNIZED]
        chan_data_uv = RAW_DATA[recognized_idx][:, idx] * 1e6
        recognized_info = build_info(tuple(RECOGNIZED), sfreq)

        # One figure PER band, stacked vertically. Each st.pyplot() call
        # gets its own native Streamlit "fullscreen" button (hover the
        # top-right corner of any topomap) — that's the built-in
        # enlarge/zoom, no extra code needed for it.
        for band_name in BAND_ORDER:
            if band_name == "Broadband":
                power = np.mean(np.abs(chan_data_uv), axis=1)
                cbar_label = "µV"
                insufficient = False
            else:
                lo, hi = BANDS[band_name]
                power = band_power(chan_data_uv, sfreq, lo, hi)
                cbar_label = "µV²"
                insufficient = power is None

            if insufficient:
                st.caption(f"**{band_name}** — collecting data (needs ~{max(2.0, 3.0/lo):.0f}s+ window)")
                continue

            fig, ax = plt.subplots(figsize=(4.2, 4.2))
            im, _ = mne.viz.plot_topomap(
                power, recognized_info, axes=ax, show=False, cmap="jet",
                sensors=True, contours=0, extrapolate="head",
            )
            ax.set_title(band_name, fontsize=11)
            cbar = fig.colorbar(im, ax=ax, shrink=0.75)
            cbar.ax.tick_params(labelsize=7)
            cbar.set_label(cbar_label, fontsize=8)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        if CUSTOM_CHANNELS:
            st.caption(f"{len(CUSTOM_CHANNELS)} channel(s) don't match a known montage and aren't shown here: {', '.join(CUSTOM_CHANNELS[:6])}{'...' if len(CUSTOM_CHANNELS) > 6 else ''}. See Session Report.")

    live_topomaps()

st.divider()

# ===============
# SESSION REPORT
# ===============
st.subheader("Session Report")

analysis_window = min(TOTAL_SAMPLES, int(min(60, TOTAL_SAMPLES / sfreq) * sfreq))
report_data_uv = RAW_DATA[:, -analysis_window:] * 1e6 if analysis_window > 0 else RAW_DATA * 1e6

band_powers = {}
for band_name, (lo, hi) in BANDS.items():
    p = band_power(report_data_uv, sfreq, lo, hi)
    band_powers[band_name] = float(np.mean(p)) if p is not None else 0.0

total_power = sum(band_powers.values()) or 1.0
band_pct = {k: 100 * v / total_power for k, v in band_powers.items()}
dominant_band = max(band_pct, key=band_pct.get) if any(band_pct.values()) else None

session_minutes = (time.time() - st.session_state.session_start) / 60
recording_seconds = TOTAL_SAMPLES / sfreq

col_a, col_b = st.columns([1, 1])
with col_a:
    st.markdown("**App usage**")
    st.markdown(
        f"- Session length: {session_minutes:.1f} min\n"
        f"- Data source: {'MongoDB (' + str(n_files) + ' file(s) stitched)' if use_mongo else 'Mock data'}\n"
        f"- Recording represented: {recording_seconds:.0f}s across {len(CH_NAMES)} channels\n"
        f"- Regions viewed this session: {', '.join(sorted(st.session_state.regions_seen)) or '—'}\n"
        f"- Dashboard refreshes: {st.session_state.refresh_count}"
    )
with col_b:
    st.markdown("**Brainwave band mix** *(last {:.0f}s, selected channels)*".format(analysis_window / sfreq))
    st.bar_chart(band_pct)

if dominant_band:
    st.markdown(
        f"Over this window, **{dominant_band}** power is dominant ({band_pct[dominant_band]:.0f}% of band power) — "
        f"{BAND_DESCRIPTIONS[dominant_band]}. This is a general, research-based association, "
        f"**not a clinical or diagnostic assessment.**"
    )
if CUSTOM_CHANNELS:
    st.caption(f"Includes {len(CUSTOM_CHANNELS)} non-standard-montage channel(s) that can't appear on the topomap: {', '.join(CUSTOM_CHANNELS[:10])}")

report_text = (
    f"EEG Session Report\n"
    f"Generated: {datetime.now(timezone.utc).isoformat()}\n\n"
    f"App usage:\n"
    f"- Session length: {session_minutes:.1f} min\n"
    f"- Data source: {'MongoDB (' + str(n_files) + ' file(s) stitched)' if use_mongo else 'Mock data'}\n"
    f"- Recording represented: {recording_seconds:.0f}s across {len(CH_NAMES)} channels\n"
    f"- Regions viewed: {', '.join(sorted(st.session_state.regions_seen)) or 'none'}\n\n"
    f"Band power mix (last {analysis_window/sfreq:.0f}s): "
    + ", ".join(f"{k} {v:.1f}%" for k, v in band_pct.items()) + "\n"
    + (f"Dominant band: {dominant_band} — {BAND_DESCRIPTIONS.get(dominant_band, '')}\n" if dominant_band else "")
    + "This is a general, research-based association, not a clinical or diagnostic assessment.\n"
)
st.download_button("Download report (.txt)", report_text, file_name="eeg_session_report.txt")
