import os
import sys

import numpy as np

from driver_prediction import get_missing_assets, predict_result


INPUT_VIDEO_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "input_video.mp4")
OUTPUT_VIDEO_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_video.mp4")


def main():
    missing_assets = get_missing_assets()
    if missing_assets:
        print("Cannot start video demo. Missing model assets:")
        for path in missing_assets:
            print(f"- {path}")
        return 1

    try:
        import cv2
    except ModuleNotFoundError:
        print("Cannot start video demo. Install 'opencv-python' in the virtual environment.")
        return 1

    if not os.path.exists(INPUT_VIDEO_FILE):
        print(f"Input video not found: {INPUT_VIDEO_FILE}")
        return 1

    vs = cv2.VideoCapture(INPUT_VIDEO_FILE)
    writer = None
    width = None
    height = None

    while True:
        grabbed, frame = vs.read()
        if not grabbed:
            break

        if width is None or height is None:
            height, width = frame.shape[:2]

        output = frame.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (128, 128))
        frame = np.expand_dims(frame, axis=0).astype("float32") / 255 - 0.5

        label = predict_result(frame)
        text = f"activity: {label}"
        cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 255, 0), 5)

        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(OUTPUT_VIDEO_FILE, fourcc, 30, (width, height), True)

        writer.write(output)
        cv2.imshow("Output", output)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    print("[INFO] cleaning up...")
    if writer is not None:
        writer.release()
    vs.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
