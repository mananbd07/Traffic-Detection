import streamlit as st
import pandas as pd
import plotly.express as px
import json
import os
import cv2
import numpy as np
import time  # <--- THIS WAS MISSING
from ultralytics import YOLO

# --- 1. ABSOLUTE PATH RESOLUTION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_FOLDER = os.path.join(ROOT_DIR, "data")
MODEL_PATH = os.path.join(ROOT_DIR, "perception", "yolov8n.pt")
LIVE_JSON = os.path.join(DATA_FOLDER, "live_status.json")
RESULTS_CSV = os.path.join(DATA_FOLDER, "results.csv")

os.makedirs(DATA_FOLDER, exist_ok=True)

# --- 2. CONFIG & THEME ---
st.set_page_config(page_title="Jagatpura Digital Twin", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    div[data-testid="stMetricValue"] { color: #00d4ff; font-family: 'Courier New', Courier, monospace; }
    .stMetric { background-color: #161b22; border: 1px solid #30363d; padding: 15px; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. VISION RESOURCE ---
@st.cache_resource
def load_yolo():
    return YOLO(MODEL_PATH)

# --- 4. DATA UTILITIES ---
def get_live_data():
    try:
        with open(LIVE_JSON, "r") as f: return json.load(f)
    except: return {"step": 0, "active_cars": 0, "current_congestion": 0}

def get_csv_data():
    if not os.path.exists(RESULTS_CSV): return None
    try:
        df = pd.read_csv(RESULTS_CSV, sep=';')
        if 'tripinfo_id' not in df.columns: df = pd.read_csv(RESULTS_CSV, sep=',')
        return df
    except: return None

# --- 5. UI HEADER ---
st.title("🏙️ Jagatpura Traffic Command Center")
tab1, tab2 = st.tabs(["📊 Analytics Dashboard", "👁️ Live Perception Feed"])

# --- TAB 1: ANALYTICS ---
with tab1:
    live = get_live_data()
    df = get_csv_data()
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Sim Step", live['step'])
    m2.metric("Vehicles", live['active_cars'])
    m3.metric("Congestion", live['current_congestion'], delta_color="inverse")
    
    eff = 0
    if df is not None:
        eff = max(0, 100 - (df['tripinfo_waitingTime'].mean() / 10))
    m4.metric("AI Efficiency", f"{eff:.1f}%")

    st.markdown("---")
    if df is not None:
        fig = px.line(df, x='tripinfo_arrival', y='tripinfo_timeLoss', 
                     template='plotly_dark', title="Quantum Pressure Analysis")
        fig.update_traces(line_color='#00d4ff')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Start simulation in 01_Simulation to populate analytics.")

# --- TAB 2: LIVE PERCEPTION (CCTV CONSOLE) ---
with tab2:
    st.subheader("🛰️ AI Computer Vision - Intersection CCTV")
    col_cam, col_data = st.columns([2, 1])
    run_vision = st.toggle("Activate AI Perception (Live Feed)")
    
    if run_vision:
        model = load_yolo()
        cap = cv2.VideoCapture(0)
        st_frame = col_cam.empty()
        st_sidebar = col_data.empty()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            results = model(frame, verbose=False, classes=[2, 3, 5, 7])
            annotated_frame = results[0].plot()
            img_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            st_frame.image(img_rgb, channels="RGB", use_container_width=True)
            
            vehicle_count = len(results[0].boxes)
            st_sidebar.markdown(f"""
                <div style="background-color: #161b22; padding: 20px; border-radius: 10px; border: 1px solid #00d4ff;">
                    <h3 style="color: #00d4ff;">🛰️ Perception Stats</h3>
                    <hr>
                    <p style="font-size: 20px;">Detected Vehicles: <b>{vehicle_count}</b></p>
                    <p style="color: #888;">Mode: Real-Time Inference</p>
                    <p style="color: #888;">Model: YOLOv8 Nano</p>
                </div>
                """, unsafe_allow_html=True)
            
            if not run_vision: break
        cap.release()
    else:
        st.info("Toggle the switch above to engage the YOLOv8 perception layer.")

# Only refresh if camera is OFF to prevent UI flickering
if not run_vision:
    time.sleep(2)
    st.rerun()