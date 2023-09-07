import streamlit as st
from utils import *
from keras.models import load_model

st.title("Pneumonia prediction")
st.header("Upload an X-chest ray image")
file = st.file_uploader("Upload your image here", type=["jpeg","png","jpg"])

# loading the pretrained model
model = load_model("keras_model.h5", compile=False)

# defining the class names
with open("labels.txt") as f:
    class_names = [a[:-1].split(" ")[1] for a in f.readlines()]

if file:
    predicted_class, conf_score = classify(file, model, class_names)
    st.write("## {}".format(predicted_class))
    st.write("##### {}".format(conf_score))