from pathlib import Path
import importlib.util
import json
import sys


ROOT = Path(__file__).resolve().parent
IMAGE_MODEL = ROOT / "model" / "self_trained" / "distracted-11-0.99.hdf5"
VIDEO_MODEL = ROOT / "model" / "self_trained" / "distracted-23-1.00.hdf5"
LABELS_FILE = ROOT / "pickle_files" / "labels_list.pkl"
INPUT_VIDEO = ROOT / "demo_on_video" / "input_video.mp4"


def has_module(module_name):
    return importlib.util.find_spec(module_name) is not None


def main():
    status = {
        "python_entrypoint": "run_project.py",
        "assets": {
            "image_model": IMAGE_MODEL.exists(),
            "video_model": VIDEO_MODEL.exists(),
            "labels_file": LABELS_FILE.exists(),
            "input_video": INPUT_VIDEO.exists(),
        },
        "dependencies": {
            "tensorflow": has_module("tensorflow"),
            "cv2": has_module("cv2"),
            "streamlit": has_module("streamlit"),
        },
    }

    print("Project status")
    print(json.dumps(status, indent=2))

    missing_assets = [name for name, present in status["assets"].items() if not present]
    if missing_assets:
        print("\nProject code is runnable, but prediction demos are blocked by missing assets.")
        print("Add the missing model files and labels file to run inference.")
        return 0

    if not status["dependencies"]["cv2"]:
        print("\nVideo demo is ready except for OpenCV. Install 'opencv-python' to run it.")
        return 0

    print("\nStarting video demo...")
    sys.path.insert(0, str(ROOT / "demo_on_video"))
    from predict_distracted import main as run_video_demo

    return run_video_demo()


if __name__ == "__main__":
    raise SystemExit(main())
