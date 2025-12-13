import argparse
import os
import cv2
import numpy as np
import yaml
from ultralytics import YOLO
from sort import Sort

# Default configs
DEFAULT_CONFIG_PATH = 'config.yaml'

def load_config(path=None, overrides=None):
    cfg = {
        'video_path': 'video/sample_video.mp4',
        'model': 'yolov8n.pt',
        'device': 'cpu',
        'conf_thresh': 0.3,
        'classes': ['car', 'truck', 'bus', 'motorbike'],
        'tracker': {'max_age': 15, 'min_hits': 3, 'iou_threshold': 0.01},
        'display': True,
        'output_video': None,
    }
    
    if path and os.path.exists(path):
        with open(path, 'r') as f:
            file_cfg = yaml.safe_load(f)
            if file_cfg:
                cfg.update(file_cfg)
    
    if overrides:
        cfg.update(overrides)
    return cfg

def draw_overlay(img, total_count, tracker_results, list_id):
    """Draws total count and bounding boxes."""
    cv2.rectangle(img, (20, 10), (280, 60), (0, 0, 0), -1)
    cv2.putText(img, f"Vehicles: {total_count}", (40, 45), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    for result in tracker_results:
        x1, y1, x2, y2, obj_id = map(int, result)
      
        color = (0, 255, 0) 

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"ID: {obj_id}", (max(0, x1), max(35, y1)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

def process_video(cfg, progress_callback=None, st_frame_placeholder=None):
    video_path = cfg['video_path']
    conf_thresh = float(cfg.get('conf_thresh', 0.5))
    target_classes = set(cfg.get('classes', []))
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    model = YOLO(cfg.get('model', 'yolov8n.pt'))
    tracker = Sort(**cfg.get('tracker', {}))
    
    out_writer = None
    if cfg.get('output_video'):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(cfg['output_video'], fourcc, 20, (w, h))

    total_count = 0
    counted_ids = []
    frame_idx = 0

    while True:
        success, img = cap.read()
        if not success:
            break
        
        frame_idx += 1
        
        results = model(img, device=cfg.get('device', 'cpu'), stream=True, verbose=False)
        
        detections = np.empty((0, 5))
        
        for r in results:
            for box in r.boxes:
                class_name = model.names[int(box.cls[0])]
                conf = float(box.conf[0])
                
                if class_name in target_classes and conf >= conf_thresh:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    detections = np.vstack((detections, [x1, y1, x2, y2, conf]))

        tracker_results = tracker.update(detections)

        for res in tracker_results:
            x1, y1, x2, y2, obj_id = res
            
            if obj_id not in counted_ids:
                total_count += 1
                counted_ids.append(obj_id)

        draw_overlay(img, total_count, tracker_results, counted_ids)

        if out_writer: 
            out_writer.write(img)

        if progress_callback:
            progress_callback(frame_idx, img, total_count, total_frames)
            
        if cfg.get('display'):
            cv2.imshow('Vehicle Counter', img)
            
            # Check for 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Check if the 'X' button was clicked
            if cv2.getWindowProperty('Vehicle Counter', cv2.WND_PROP_VISIBLE) < 1:
                break

    cap.release()
    if out_writer:
        out_writer.release()
    cv2.destroyAllWindows()
    
    return {'total': total_count, 'frames': frame_idx, 'expected_frames': total_frames}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--video', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--no-display', dest='display', action='store_false')
    args = parser.parse_args()

    overrides = {k: v for k, v in vars(args).items() if v is not None and k != 'config'}
    
    cfg = load_config(args.config, overrides)
    print(f"Running on {cfg['video_path']}...")
    
    stats = process_video(cfg)
    print(f"Finished. Total vehicles: {stats['total']}")