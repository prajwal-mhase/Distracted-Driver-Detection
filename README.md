# Distracted Driver Detection

This project detects driver activity from images and video using a deep learning model.

## Classes

- Safe driving
- Texting right
- Talking on phone right
- Texting left
- Talking on phone left
- Operating radio
- Drinking
- Reaching behind
- Hair and makeup
- Talking to passenger

## Setup

Activate the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Required Files

To run predictions, these files must exist in the project:

- `model/self_trained/distracted-11-0.99.hdf5`
- `model/self_trained/distracted-23-1.00.hdf5`
- `pickle_files/labels_list.pkl`

## Run Image Demo

```powershell
cd demo_on_image
streamlit run streamlit_demo.py
```

## Run Video Demo

```powershell
cd demo_on_video
python predict_distracted.py
```

## Project Folders

- `demo_on_image` - image prediction demo
- `demo_on_video` - video prediction demo
- `Training Notebooks` - model training notebooks
- `images` - sample images
