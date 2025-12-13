import streamlit as st
import tempfile
import cv2
import os
from yolo_car_counter import load_config, process_video

st.set_page_config(page_title='AI Vehicle Counter', layout='wide')

st.sidebar.title("Configuration")
model_choice = st.sidebar.selectbox('Model', ['yolov8n.pt', 'yolov8m.pt'])
conf_thresh = st.sidebar.slider('Confidence', 0.1, 0.9, 0.5)
target_classes = st.sidebar.multiselect(
    "Classes to Count", 
    ['car', 'truck', 'bus', 'motorbike', 'bicycle', 'person'],
    default=['car', 'truck', 'bus', 'motorbike']
)
st.title('Vehicle Counter (YOLOv8 + SORT)')

col1, col2 = st.columns([2, 1])

with col1:
    source_radio = st.radio("Select Video Source", ["Upload Video", "Use Default Config Video"])
    
    video_path_override = None
    if source_radio == "Upload Video":
        uploaded_file = st.file_uploader('Upload MP4', type=['mp4', 'mov', 'avi'])
        if uploaded_file:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            video_path_override = tfile.name
    
    start_btn = st.button('Start Counting', type='primary')

with col2:
    st.markdown("### Live Stats")
    counter_metric = st.empty()
    frame_metric = st.empty()
    curr_frame_display = st.empty()

if start_btn:
    overrides = {
        'model': model_choice,
        'conf_thresh': conf_thresh,
        'classes': target_classes,
        'display': False 
    }
    
    if video_path_override:
        overrides['video_path'] = video_path_override
        
    cfg = load_config('config.yaml', overrides=overrides)

    if not os.path.exists(cfg['video_path']):
        st.error(f"Video file not found: {cfg['video_path']}")
        st.stop()

    progress_bar = st.progress(0)
    
    def streamlit_callback(frame_idx, img, total, total_frames):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        curr_frame_display.image(img_rgb, channels='RGB', width="stretch")
        counter_metric.metric("Vehicles Counted", total)
        frame_metric.text(f"Frame: {frame_idx} / {total_frames}")
        
        if total_frames > 0:
            val = min(1.0, frame_idx / total_frames)
            progress_bar.progress(val)

    with st.spinner(f"Loading {model_choice}..."):
        try:
            stats = process_video(cfg, progress_callback=streamlit_callback)
            progress_bar.progress(100)
            st.success(f"Processing Complete! Total: {stats['total']}")
        except Exception as e:
            st.error(f"Error during processing: {e}")