# Car Counter

Vehicle counter using YOLOv8 + SORT.

## Features

* Lightweight YOLOv8-based detector with SORT tracking
* Modular `yolo_car_counter.py` with YAML config and CLI overrides
* Streamlit UI for quick demos and visualization

## Quick Start

### 1. Setup Environment
Create a virtual environment and install dependencies:

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# Mac/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Setup Video
You follow the steps below to setup testing video:

1. Create a folder named `video` in the project root.

2. Download a sample traffic video. We recommend using traffic videos from Pixabay.

3. Rename the downloaded file to `sample_video.mp4`.

4. Move the file into the `video/` folder so the path is `video/sample_video.mp4`.

(Note: You can use any video, but update the filename in `config.yaml` if it differs.)

### 3. Run the App

Option A: Run demo on Streamlit web interface:

```bash
streamlit run streamlit_app.py
```

Option B: Run from command line:

```bash
python yolo_car_counter.py --config config.yaml
```

## Config
You can override default settings from `config.yaml` via CLI arguments:

```bash
python yolo_car_counter.py --video video/my_new_video.mp4 --model yolov8m.pt
```

## Notes
Model: Default is `yolov8n.pt` for speed. To use a larger, more accurate model, change the model setting in `config.yaml` or pass `--model yolov8m.pt`.

Performance: Streamlit runs inference frame-by-frame in the server process — it is provided as a convenient UI for demos and small videos. For long videos, the CLI script is recommended.

