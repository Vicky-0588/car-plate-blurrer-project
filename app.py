import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

st.set_page_config(page_title="License Plate Blurrer (OpenCV)", layout="centered")

st.title("🔒 Car Number-Plate Blurrer (No Tesseract)")
st.write("Upload a car image, and this app will blur the number plate(s) using OpenCV Haar Cascade.")

# Load OpenCV pre-trained Haar Cascade for license plates
# (comes with opencv-python, usually located in cv2.data.haarcascades)
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_russian_plate_number.xml"
plate_cascade = cv2.CascadeClassifier(CASCADE_PATH)

uploaded = st.file_uploader("Upload image (jpg, png)", type=["jpg", "jpeg", "png"])
blur_strength = st.slider("Blur strength (kernel size)", 5, 101, 31, step=2)
show_boxes = st.checkbox("Show detected plate boxes", value=False)

def load_image(file) -> np.ndarray:
    image = Image.open(file).convert("RGB")
    return np.array(image)[:, :, ::-1]  # PIL RGB -> OpenCV BGR

def blur_regions(img_bgr, regions, ksize):
    out = img_bgr.copy()
    for (x, y, w, h) in regions:
        roi = out[y:y+h, x:x+w]
        # Ensure kernel size fits ROI
        k = min(ksize, max(3, (min(roi.shape[0], roi.shape[1]) // 2) * 2 + 1))
        blurred = cv2.GaussianBlur(roi, (k, k), 0)
        out[y:y+h, x:x+w] = blurred
    return out

if uploaded is not None:
    file_bytes = uploaded.read()
    file_stream = io.BytesIO(file_bytes)
    img_bgr = load_image(file_stream)

    st.subheader("Original Image")
    st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_column_width=True)

    # Detect plates
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 20))

    if len(plates) == 0:
        st.warning("No license plate detected. Try another image or adjust detection parameters.")
    else:
        st.success(f"Detected {len(plates)} plate(s).")

    # Optionally show detection boxes
    if show_boxes and len(plates) > 0:
        debug = img_bgr.copy()
        for (x, y, w, h) in plates:
            cv2.rectangle(debug, (x, y), (x+w, y+h), (0, 255, 0), 2)
        st.subheader("Detected Plates (debug)")
        st.image(cv2.cvtColor(debug, cv2.COLOR_BGR2RGB), use_column_width=True)

    # Blur detected plates
    out_img = blur_regions(img_bgr, plates, blur_strength)

    st.subheader("Result (Plates Blurred)")
    st.image(cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB), use_column_width=True)

    # Download button
    out_pil = Image.fromarray(cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB))
    buf = io.BytesIO()
    out_pil.save(buf, format="PNG")
    st.download_button("Download Blurred Image", data=buf.getvalue(),
                       file_name="blurred_plate.png", mime="image/png")
else:
    st.info("Upload an image to start.")
