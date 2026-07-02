import streamlit as st
import plotly.graph_objects as go
import numpy as np
import requests

API_URL = st.sidebar.text_input("Backend API URL", value="http://localhost:5000/api/eeg/latest")
#test
if st.sidebar.button("Fetch Data"):
    try:
        response = requests.get(API_URL, timeout=5)
        response.raise_for_status()
        data = response.json()
        st.success("Data received ✅")
        st.json(data)
    except requests.exceptions.RequestException as e:
        st.error(f"Could not reach API: {e}")

# Page configuration
st.set_page_config(
    page_title="EEG Dashboard",
    layout="wide"
)

# ---- Header ----
st.title("Real-Time EEG Dashboard")

st.divider()

# ---- Sidebar ----
st.sidebar.header("Dashboard Controls")
st.sidebar.write("This panel will later control live data refresh, filters, and report downloads.")
patient_id = st.sidebar.text_input("Patient / Session ID", value="EEG-001")
st.sidebar.button("Start Monitoring (coming soon)")

# ---- Main layout ----
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="Alpha Power", value="-- μV²")

with col2:
    st.metric(label="Beta Power", value="-- μV²")

with col3:
    st.metric(label="Signal Quality", value="-- %")

st.divider()

st.subheader("Live EEG Signal")
t = np.linspace(0, 10, 200)
signal = np.sin(t * 2) + np.sin(t * 5) * 0.3

fig = go.Figure()
fig.add_trace(go.Scatter(x=t, y=signal, mode="lines", name="EEG Amplitude"))
fig.update_layout(xaxis_title="Time (s)", yaxis_title="Amplitude (μV)")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Frequency Band Breakdown")
st.info("Bar chart for Delta / Theta / Alpha / Beta bands coming in Week 6.")
