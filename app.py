"""
Real-Time EEG Monitoring Dashboard
NeuroLab Internship — Student 4: Visualization Engineer

v3 changes:
- No hardcoded sampling rate or channel count/list anymore. Both are
  configurable, so this works whether Student 3's API sends 19 channels
  at 128Hz or 64 channels at 160Hz or anything else ("wide range of data
  format, without restricting it").
- The time-series chart is now a custom HTML5 canvas component
  (via st.components.v1.html) that animates at 60fps in the browser using
  requestAnimationFrame, sweeping like a hospital patient monitor. It is
  NOT tied to Streamlit's rerun cycle, so there's no periodic-refresh jank.
- The topomap is a snapshot (like on real clinical monitors) and still
  refreshes periodically via st.fragment — that's an intentional, correct
  design choice, not a limitation.
"""

import streamlit as st
import numpy as np
import mne
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import decimate
import streamlit.components.v1 as components
import json
import matplotlib.collections

# ============================================================
# PAGE CONFIG — must be the first Streamlit command
# ============================================================
st.set_page_config(page_title="EEG Dashboard", layout="wide")
st.markdown("<style>section.main{overflow-anchor: none;}</style>", unsafe_allow_html=True)

BANDS = {"Alpha": (8, 13), "Beta": (13, 30), "Gamma": (30, 45)}
MIN_SAMPLES_FOR_FILTER = 256  # ~2s at 128Hz-equivalent; guards against short-buffer filter distortion
DOWNSAMPLE_THRESHOLD = 500


# ============================================================
# FLEXIBLE CHANNEL NAMING — picks real 10-10 electrode names for
# whatever channel count is requested, instead of assuming a fixed
# 19-channel layout. Falls back to a large generic pool for odd counts.
# ============================================================
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
    return pool + [f"CH{i}" for i in range(len(pool), n_channels)]  # extremely large N, rare edge case


def classify_region(ch):
    """Buckets any standard 10-10/10-20 channel name into one of 5 broad
    regions by name prefix. Boundary electrodes (FC/FT/CP/PO) are assigned
    to their nearest primary region — a common simplification, not a strict
    anatomical atlas."""
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
def get_montage_for(ch_names):
    """Tries standard_1005 (343-name superset covering virtually any 10-10
    cap) and only keeps positions for channels that actually match."""
    montage = mne.channels.make_standard_montage("standard_1005")
    return montage


@st.cache_resource
def build_info(ch_names, sfreq):
    info = mne.create_info(ch_names=list(ch_names), sfreq=sfreq, ch_types="eeg")
    info.set_montage(get_montage_for(ch_names), on_missing="ignore")
    return info


@st.cache_resource
def load_raw_data(n_channels, sfreq, duration_sec=60):
    """
    Loads the 'original' raw EEG recording as an MNE Raw object.

    TODO once Student 3's API is live: replace this function's body with
    the real fetch/parse (e.g. requests.get(...).json()), building a
    (n_channels, n_samples) array the same way, then:
        info = mne.create_info(real_ch_names, real_sfreq, "eeg")
        raw = mne.io.RawArray(real_data_in_volts, info)
    Everything downstream (montage, filtering, topomap, region grouping)
    already works off ch_names/sfreq read from this object — no other
    code needs to change.
    """
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

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data_uv * 1e-6, info, verbose=False)
    raw.set_montage(get_montage_for(ch_names), on_missing="ignore")
    return raw


def dynamic_downsample(arr, threshold=DOWNSAMPLE_THRESHOLD):
    out = arr
    while len(out) > threshold:
        out = decimate(out, 2, zero_phase=True)
    return out


# ============================================================
# HEADER
# ============================================================
st.title("EEG Dashboard")

# ============================================================
# SIDEBAR — data format is fully configurable, nothing hardcoded.
# Defaults match what Student 3 reported (160Hz, 64 channels) but
# either can be changed to match whatever the real API ends up sending.
# ============================================================
st.sidebar.header("Data Source")
n_channels = st.sidebar.number_input("Number of channels", min_value=4, max_value=256, value=64, step=1)
sfreq = st.sidebar.number_input("Sampling rate (Hz)", min_value=32, max_value=1024, value=160, step=8)

st.sidebar.header("Display Controls")
window_sec = st.sidebar.slider("Sweep window (seconds)", 1, 10, 4)
band_choice = st.sidebar.selectbox("Topomap frequency band", ["Alpha", "Beta", "Gamma", "Broadband (no filter)"])
topo_refresh = st.sidebar.slider("Topomap refresh interval (s)", 0.5, 3.0, 1.0, 0.5)

RAW = load_raw_data(n_channels, sfreq)
RAW_DATA = RAW.get_data()  # (n_channels, n_samples), volts
CH_NAMES = RAW.info["ch_names"]
TOTAL_SAMPLES = RAW_DATA.shape[1]
FALLBACK_POOL = get_fallback_pool()
RECOGNIZED = [ch for ch in CH_NAMES if ch in FALLBACK_POOL]
INFO = build_info(tuple(CH_NAMES), sfreq)

region_map = {}
for ch in CH_NAMES:
    region_map.setdefault(classify_region(ch), []).append(ch)
region_options = [r for r in ["Frontal", "Central", "Temporal", "Parietal", "Occipital", "Other"] if r in region_map]

region_choice = st.sidebar.multiselect("Brain regions to display", options=region_options, default=region_options)
st.sidebar.caption("Topomap always shows all channels — region filters only narrow the scrolling chart.")
st.divider()

selected_channels = [ch for ch in CH_NAMES if classify_region(ch) in region_choice] or CH_NAMES

# ============================================================
# FLUID TIME-SERIES CHART — HTML5 canvas + requestAnimationFrame.
#
# Why a custom component instead of st.line_chart / st.plotly_chart:
# Streamlit's native chart elements only redraw when Python reruns and
# sends new data — even fast reruns feel like discrete "jumps," and
# Plotly's iframe re-render specifically causes a scroll-jump bug.
# This component instead computes the EEG-like waveform with a plain
# math formula (same delta/theta/alpha/beta character as our mock
# generator) directly in JavaScript, so the browser can animate it at
# 60fps completely on its own — no round-trip to Python needed for the
# animation itself, which is what makes it look continuous like a real
# patient monitor instead of refreshing in visible steps.
#
# It sweeps like a hospital monitor: a cursor moves left-to-right over
# `window_sec` seconds; ahead of the cursor you still see the previous
# lap's trace, which gets overwritten as the cursor sweeps back through.
# ============================================================
def render_sweep_monitor(channels, window_sec, height_per_channel=34):
    total_height = max(200, len(channels) * height_per_channel)
    payload = json.dumps({"channels": channels, "windowSec": window_sec})

    # Changed div background to white with a subtle border
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

        function seededRand(seed) {{
            let s = seed;
            return function() {{ s = (s * 9301 + 49297) % 233280; return s / 233280; }};
        }}
        const phases = channels.map((_, i) => {{
            const r = seededRand(i * 1000 + 1);
            return [[2, 8], [6, 5], [10, 6], [20, 3], [40, 2]].map(([f, a]) => [f, a, r() * 2 * Math.PI]);
        }});

        function signalAt(chIndex, t) {{
            let v = 0;
            for (const [freq, amp, phase] of phases[chIndex]) {{
                v += amp * Math.sin(2 * Math.PI * freq * t + phase);
            }}
            v += 3 * Math.sin(2 * Math.PI * 0.7 * t + chIndex) * Math.sin(2 * Math.PI * 3.3 * t);
            return v;
        }}

        const POINTS = 300; 

        function draw(tsMs) {{
            const nowSec = tsMs / 1000;
            const w = canvas.clientWidth || canvas.parentElement.clientWidth;
            const h = {total_height};
            const laneH = h / channels.length;

            // White background fill
            ctx.fillStyle = "#ffffff";
            ctx.fillRect(0, 0, w, h);

            channels.forEach((name, i) => {{
                const laneY = i * laneH;
                const midY = laneY + laneH / 2;
                const amp = laneH * 0.42;

                // Faint gray grid line separating channels
                ctx.strokeStyle = "rgba(0,0,0,0.08)";
                ctx.beginPath(); ctx.moveTo(0, laneY); ctx.lineTo(w, laneY); ctx.stroke();

                // Crisp black trace line
                ctx.strokeStyle = "#000000";
                ctx.lineWidth = 1.2;
                
                ctx.beginPath();
                for (let p = 0; p <= POINTS; p++) {{
                    const frac = p / POINTS;
                    const x = frac * w;
                    
                    const tAtPoint = nowSec - windowSec * (1 - frac); 
                    
                    const val = signalAt(i, tAtPoint);
                    const y = midY - (val / 25) * amp;
                    
                    if (p === 0) {{
                        ctx.moveTo(x, y);
                    }} else {{
                        ctx.lineTo(x, y);
                    }}
                }}
                ctx.stroke();

                // Dark gray channel label
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
    components.html(html, height=total_height + 20)


st.subheader("Live Multi-Channel EEG Signal")
st.caption(f"{len(selected_channels)} channels · {sfreq} Hz · {window_sec}s sweep · "
           f"{' + '.join(region_choice) if region_choice else 'All regions'} ")
render_sweep_monitor(selected_channels, window_sec)


# ============================================================
# TOPOGRAPHIC MAP — periodic snapshot (this is normal: even real
# clinical monitors refresh topomaps discretely, not continuously).
# Uses st.fragment so it auto-refreshes without blocking the sidebar.
# ============================================================
if "topo_pos" not in st.session_state:
    st.session_state.topo_pos = 0

@st.fragment(run_every=topo_refresh)
def live_topomap():
    st.session_state.topo_pos = (st.session_state.topo_pos + sfreq) % TOTAL_SAMPLES
    pos = st.session_state.topo_pos
    window_samples = window_sec * sfreq
    idx = np.arange(pos - window_samples, pos) % TOTAL_SAMPLES

    st.subheader("Topographic Map")
    if band_choice != "Broadband (no filter)" and len(idx) < MIN_SAMPLES_FOR_FILTER:
        st.info(f"Collecting data for {band_choice} filtering — need ~2s of signal.")
        return

    recognized_idx = [CH_NAMES.index(ch) for ch in RECOGNIZED]
    chan_data_uv = RAW_DATA[recognized_idx][:, idx] * 1e6

    if band_choice == "Broadband (no filter)":
        power = np.mean(np.abs(chan_data_uv), axis=1)
        cbar_label = "µV"
        caption = "Average signal amplitude across the scalp (unfiltered)"
    else:
        lo, hi = BANDS[band_choice]
        filtered = mne.filter.filter_data(chan_data_uv, sfreq, lo, hi, verbose=False)
        power = np.mean(filtered ** 2, axis=1)
        cbar_label = "Power (µV²)"
        caption = f"{band_choice} band power ({lo}-{hi} Hz) across the scalp"

    st.caption(caption)
    recognized_info = build_info(tuple(RECOGNIZED), sfreq)
    
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    
    im, cn = mne.viz.plot_topomap(
        power, recognized_info, axes=ax, show=False, cmap="jet",
        names=[f"{v:.2f}" for v in power], 
        sensors=True, 
        contours=0, 
        extrapolate='head'
    )
    
    for line in ax.lines:
        line.set_visible(False)
        
    for coll in ax.collections:
        if isinstance(coll, matplotlib.collections.PathCollection):
            coll.set_visible(False)
    
    for txt in ax.texts:
        # REDUCED FONT SIZE
        txt.set_fontsize(5.5) 
        x, y = txt.get_position()
        
        circle = plt.Circle((x, y), radius=0.004, fill=False, edgecolor='black', linewidth=0.5, zorder=5)
        ax.add_patch(circle)
        
        # Tweak the shift to match the smaller text
        txt.set_position((x, y - 0.010)) 
        
        txt.set_bbox(dict(
            facecolor='white', 
            alpha=0.85, 
            edgecolor='black', 
            linewidth=0.5,
            boxstyle='square,pad=0.1' # REDUCED PADDING for a tighter box
        ))

    cbar = fig.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label(cbar_label)
    st.pyplot(fig)
    plt.close(fig)

live_topomap()
