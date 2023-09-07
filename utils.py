import numpy as np
from PIL import Image
import streamlit as st

def classify(file, model, class_names):
    image = Image.open(file).convert("RGB")
    st.image(image, use_column_width=True)
    image = image.resize((224,224))
    image = np.array(image)
    image = (image / 127.5) - 1
    image = image.reshape((1,224,224,3))

    y_pred = model.predict(image)
    index = np.argmax(y_pred)
    predicted_class = class_names[index]
    conf_score = y_pred[0][index]

    return predicted_class, conf_score