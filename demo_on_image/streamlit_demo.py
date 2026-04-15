import time

import streamlit as st
from PIL import Image

from predictionOnImage import get_missing_assets, return_prediction


st.title("Distracted Driver Detection")


def main():
    missing_assets = get_missing_assets()
    if missing_assets:
        st.warning("Model files are missing. Add the required assets before running predictions.")
        for path in missing_assets:
            st.code(path)
        return

    file_uploaded = st.file_uploader("Choose File", type=["png", "jpg", "jpeg"])
    classify = st.button("Classify")

    if file_uploaded is not None:
        image = Image.open(file_uploaded)
        st.image(image, caption="Uploaded Image", use_column_width=True)
    else:
        image = None

    if classify:
        if image is None:
            st.write("Please upload an image first.")
        else:
            with st.spinner("Model working..."):
                prediction = return_prediction(image)
                time.sleep(1)
            st.success("Classified")
            st.write(prediction)


if __name__ == "__main__":
    main()
