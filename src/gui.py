import streamlit as st
from streamlit_image_comparison import image_comparison

import cv2
import numpy as np
from PIL import Image
import requests

st.write("Streamlit is also great for more traditional ML use cases like computer vision or NLP. Here's an example of edge detection using OpenCV. 👁️") 

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file)
else:
    image = Image.open(requests.get("https://picsum.photos/200/120", stream=True).raw)

edges = cv2.Canny(np.array(image), 100, 200)
tab1, tab2 = st.tabs(["Detected edges", "Original"])
tab1.image(edges, use_column_width=True)
tab2.image(image, use_column_width=True)

image_comparison(
    img1=image,
    img2=edges,
    label1="Before",
    label2="After",
    width=700,
    starting_position=50,  # % where the slider starts
    show_labels=True,
    make_responsive=True,
    in_memory=True,
)
# tab2.image(image_comparison, use_column_width=True)
