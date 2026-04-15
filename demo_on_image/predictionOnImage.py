import json
import os
import pickle

import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from PIL import ImageFile


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_MODEL_PATH = os.path.join(BASE_DIR, "model")
PICKLE_DIR = os.path.join(BASE_DIR, "pickle_files")
BEST_MODEL = os.path.join(BASE_MODEL_PATH, "self_trained", "distracted-11-0.99.hdf5")
LABELS_FILE = os.path.join(PICKLE_DIR, "labels_list.pkl")

CLASS_NAME_MAP = {
    "c0": "SAFE_DRIVING",
    "c1": "TEXTING_RIGHT",
    "c2": "TALKING_PHONE_RIGHT",
    "c3": "TEXTING_LEFT",
    "c4": "TALKING_PHONE_LEFT",
    "c5": "OPERATING_RADIO",
    "c6": "DRINKING",
    "c7": "REACHING_BEHIND",
    "c8": "HAIR_AND_MAKEUP",
    "c9": "TALKING_TO_PASSENGER",
}

_MODEL = None
_LABELS = None


def get_missing_assets():
    missing = []
    if not os.path.exists(BEST_MODEL):
        missing.append(BEST_MODEL)
    if not os.path.exists(LABELS_FILE):
        missing.append(LABELS_FILE)
    return missing


def _load_model_and_labels():
    global _MODEL, _LABELS
    if _MODEL is not None and _LABELS is not None:
        return _MODEL, _LABELS

    missing = get_missing_assets()
    if missing:
        raise FileNotFoundError(
            "Required model assets are missing:\n" + "\n".join(missing)
        )

    _MODEL = load_model(BEST_MODEL)
    with open(LABELS_FILE, "rb") as handle:
        _LABELS = pickle.load(handle)
    return _MODEL, _LABELS


def path_to_tensor(pil_image):
    resized = pil_image.convert("RGB").resize((128, 128))
    x = image.img_to_array(resized)
    return np.expand_dims(x, axis=0)


def return_prediction(pil_image):
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    model, labels_id = _load_model_and_labels()
    test_tensors = path_to_tensor(pil_image).astype("float32") / 255 - 0.5

    ypred_test = model.predict(test_tensors, verbose=0)
    ypred_class = int(np.argmax(ypred_test, axis=1))

    id_labels = {idx: class_name for class_name, idx in labels_id.items()}
    predicted_key = id_labels[ypred_class]
    return CLASS_NAME_MAP[predicted_key]


if __name__ == "__main__":
    print(json.dumps({"missing_assets": get_missing_assets()}, indent=2))
