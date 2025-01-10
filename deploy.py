import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
# Streamlit Deployment
st.title('Klasifikasi Daun Sawi - Ada Hama atau Tidak')

uploaded_file = st.file_uploader("Upload gambar daun sawi", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).resize((150, 150))
    st.image(image, caption='Gambar yang diupload.', use_container_width=True)
    image = np.array(image.convert('RGB'))/255.0
    image = np.expand_dims(image, axis=0)

    # Load model and predict
    model = tf.keras.models.load_model('sawi_model.h5')
    prediction = model.predict(image)

    if np.argmax(prediction) == 0:
        st.write("### Daun Sawi Ada Hama")
    else:
        st.write("### Daun Sawi Tidak Ada Hama")