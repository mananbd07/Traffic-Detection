import streamlit as st
import pandas as pd
import plotly.express as px
import json
import os
import cv2
import numpy as np
import time
import tempfile
from ultralytics import YOLO

# --- 1. ABSOLUTE PATH RESOLUTION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_FOLDER = os.path.join(ROOT_DIR, "data")
DEMO_DATA_FOLDER = os.path.join(ROOT_DIR, "demo_data")
MODEL_PATH = os.path.join(ROOT_DIR, "perception", "yolov8n.pt")

# --- 2. CONFIG & THEME ---
st.set_page_config(page_title="Jagatpura Digital Twin", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    div[data-testid="stMetricValue"] { color: #00d4ff; font-family: 'Courier New', Courier, monospace; }
    .stMetric { background-color: #161b22; border: 1px solid #30363d; padding: 15px; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. SIDEBAR: MODE SELECTION ---
st.sidebar.title("Configuration")
app_mode = st.sidebar.radio("Operation Mode", ["Demo (Public Web)", "Local Research (SUMO+Webcam)"])

if app_mode == "Demo (Public Web)":
    LIVE_JSON = os.path.join(DEMO_DATA_FOLDER, "demo_live_status.json")
    RESULTS_CSV = os.path.join(DEMO_DATA_FOLDER, "demo_results.csv")
    st.sidebar.info("Using precomputed simulation data.")
else:
    LIVE_JSON = os.path.join(DATA_FOLDER, "live_status.json")
    RESULTS_CSV = os.path.join(DATA_FOLDER, "results.csv")
    st.sidebar.warning("Requires local SUMO & Webcam.")
    os.makedirs(DATA_FOLDER, exist_ok=True)

# --- 4. VISION RESOURCE ---
@st.cache_resource
def load_yolo():
    return YOLO(MODEL_PATH)

# --- 5. DATA UTILITIES ---
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

# --- 6. UI HEADER ---
st.title("🏙️ Jagatpura Traffic Command Center")
tab1, tab2, tab3 = st.tabs(["📊 Analytics Dashboard", "👁️ Live Perception Feed", "ℹ️ About / Architecture"])

# --- TAB 1: ANALYTICS ---
with tab1:
    live = get_live_data()
    df = get_csv_data()
    
    if app_mode == "Demo (Public Web)":
        st.markdown("*Note: Displaying precomputed SUMO demonstration data. Not a live simulation.*")
        
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Sim Step", live.get('step', 0))
    m2.metric("Vehicles", live.get('active_cars', 0))
    m3.metric("Congestion", live.get('current_congestion', 0), delta_color="inverse")
    
    eff = 0
    if df is not None:
        if 'tripinfo_waitingTime' in df.columns:
            eff = max(0, 100 - (df['tripinfo_waitingTime'].mean() / 10))
    m4.metric("AI Efficiency", f"{eff:.1f}%")

    st.markdown("---")
    if df is not None and 'tripinfo_arrival' in df.columns and 'tripinfo_timeLoss' in df.columns:
        fig = px.line(df, x='tripinfo_arrival', y='tripinfo_timeLoss', 
                     template='plotly_dark', title="Quantum Pressure Analysis (Experimental)")
        fig.update_traces(line_color='#00d4ff')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Start simulation in 01_Simulation to populate analytics, or switch to Demo Mode.")

# --- TAB 2: LIVE PERCEPTION (CCTV CONSOLE) ---
with tab2:
    st.subheader("🛰️ AI Computer Vision - Intersection CCTV")
    col_cam, col_data = st.columns([2, 1])
    
    if app_mode == "Demo (Public Web)":
        upload_type = st.radio("Select Input Type:", ["Image", "Video"], horizontal=True)
        
        if upload_type == "Image":
            st.info("Upload an image of a traffic intersection to test the YOLOv8 Nano model.")
            uploaded_file = st.file_uploader("Upload Traffic Image", type=['jpg', 'jpeg', 'png'])
            
            if uploaded_file is not None:
                model = load_yolo()
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                frame = cv2.imdecode(file_bytes, 1)
                
                # Predict
                results = model(frame, verbose=False, classes=[2, 3, 5, 7])
                annotated_frame = results[0].plot()
                img_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                
                # Display
                col_cam.image(img_rgb, channels="RGB", use_container_width=True)
                
                # Counting
                vehicle_count = len(results[0].boxes)
                class_names = model.names
                counts = {class_names[c]: 0 for c in [2, 3, 5, 7]}
                for box in results[0].boxes:
                    cls_id = int(box.cls[0].item())
                    if cls_id in [2, 3, 5, 7]:
                        counts[class_names[cls_id]] += 1
                
                # Density
                density = "Low"
                if vehicle_count > 5: density = "Medium"
                if vehicle_count > 15: density = "High"
                
                col_data.markdown(f"""
                    <div style="background-color: #161b22; padding: 20px; border-radius: 10px; border: 1px solid #00d4ff;">
                        <h3 style="color: #00d4ff;">🛰️ Perception Stats</h3>
                        <hr>
                        <p style="font-size: 20px;">Total Detected: <b>{vehicle_count}</b></p>
                        <p>Estimated Density: <b>{density}</b></p>
                        <p style="color: #888;">Mode: Static Image Inference</p>
                        <p style="color: #888;">Model: YOLOv8 Nano</p>
                        <hr>
                        <h4>Category Breakdown:</h4>
                        <p>Cars: {counts.get('car', 0)}</p>
                        <p>Motorcycles: {counts.get('motorcycle', 0)}</p>
                        <p>Buses: {counts.get('bus', 0)}</p>
                        <p>Trucks: {counts.get('truck', 0)}</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("Upload a short traffic video to run sampled YOLOv8 inference. (Analyzes up to 10 sampled frames).")
            uploaded_video = st.file_uploader("Upload Traffic Video", type=['mp4', 'avi', 'mov'])
            
            if uploaded_video is not None:
                if st.button("Analyze Video"):
                    model = load_yolo()
                    
                    file_ext = os.path.splitext(uploaded_video.name)[1]
                    if not file_ext: file_ext = '.mp4'
                    
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext)
                    tfile.write(uploaded_video.read())
                    tfile.close()
                    
                    try:
                        cap = cv2.VideoCapture(tfile.name)
                        if not cap.isOpened():
                            st.error("Error: Could not open video file.")
                        else:
                            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                            fps = cap.get(cv2.CAP_PROP_FPS)
                            duration = total_frames / fps if fps > 0 else 0
                            
                            st.write(f"**Video Metadata:** {total_frames} frames | {fps:.1f} FPS | {duration:.1f}s")
                            
                            MAX_FRAMES = 10
                            sample_interval = max(1, total_frames // MAX_FRAMES)
                            
                            progress_bar = st.progress(0)
                            st_frame = col_cam.empty()
                            st_sidebar = col_data.empty()
                            
                            analyzed_frames = 0
                            cumulative_counts = {'car': 0, 'motorcycle': 0, 'bus': 0, 'truck': 0}
                            total_detections_all_frames = 0
                            peak_vehicles = 0
                            
                            for f in range(total_frames):
                                ret, frame = cap.read()
                                if not ret: break
                                
                                if f % sample_interval == 0 and analyzed_frames < MAX_FRAMES:
                                    results = model(frame, verbose=False, classes=[2, 3, 5, 7])
                                    annotated_frame = results[0].plot()
                                    img_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                                    
                                    st_frame.image(img_rgb, channels="RGB", use_container_width=True)
                                    
                                    vehicle_count = len(results[0].boxes)
                                    peak_vehicles = max(peak_vehicles, vehicle_count)
                                    total_detections_all_frames += vehicle_count
                                    
                                    class_names = model.names
                                    for box in results[0].boxes:
                                        cls_id = int(box.cls[0].item())
                                        if cls_id in [2, 3, 5, 7]:
                                            cumulative_counts[class_names[cls_id]] += 1
                                    
                                    analyzed_frames += 1
                                    progress_bar.progress(analyzed_frames / min(MAX_FRAMES, max(1, total_frames // sample_interval + 1)))
                            
                            cap.release()
                            avg_vehicles = total_detections_all_frames / analyzed_frames if analyzed_frames > 0 else 0
                            
                            st_sidebar.markdown(f"""
                                <div style="background-color: #161b22; padding: 20px; border-radius: 10px; border: 1px solid #00d4ff;">
                                    <h3 style="color: #00d4ff;">🛰️ Video Analysis Summary</h3>
                                    <hr>
                                    <p>Analyzed Frames: <b>{analyzed_frames} (Max {MAX_FRAMES})</b></p>
                                    <p>Peak Vehicles/Frame: <b>{peak_vehicles}</b></p>
                                    <p>Avg Vehicles/Frame: <b>{avg_vehicles:.1f}</b></p>
                                    <p style="font-size: 12px; color: #888;">* Detection counts represent YOLO detections across analyzed frames, not unique vehicles.</p>
                                    <hr>
                                    <h4>Cumulative Detections:</h4>
                                    <p>Cars: {cumulative_counts.get('car', 0)}</p>
                                    <p>Motorcycles: {cumulative_counts.get('motorcycle', 0)}</p>
                                    <p>Buses: {cumulative_counts.get('bus', 0)}</p>
                                    <p>Trucks: {cumulative_counts.get('truck', 0)}</p>
                                </div>
                                """, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Error processing video: {e}")
                    finally:
                        if os.path.exists(tfile.name):
                            os.remove(tfile.name)
            
    else:
        # LOCAL RESEARCH MODE (WEBCAM)
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
            st.info("Toggle the switch above to engage the YOLOv8 perception layer via Webcam.")
        
        # Only refresh if camera is OFF to prevent UI flickering
        if not run_vision:
            time.sleep(2)
            st.rerun()

# --- TAB 3: ABOUT / ARCHITECTURE ---
with tab3:
    st.markdown("""
    ## Traffic Detection & Optimization System
    This dashboard provides a unified interface for monitoring real-time vehicle perception and traffic optimization.
    
    **Features:**
    - **YOLOv8 Perception:** Real-time object detection mapping intersection queues.
    - **SUMO / TraCI:** Microscopic simulation engine (Running in Research Mode).
    - **Experimental Optimization:** Classical/Quantum heuristic modeling to mitigate traffic pressure.
    
    *Use the sidebar to switch between Public Demo mode (using precomputed simulation outputs and image uploads) and Local Research mode (requiring a full local setup and webcam).*
    """)