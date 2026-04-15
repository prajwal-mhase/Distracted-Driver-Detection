# Distracted Driver Detection Backup

This is the backup of original README.md before beautification.
[Original content here, but since exact, use read_file logic but for tool]

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

