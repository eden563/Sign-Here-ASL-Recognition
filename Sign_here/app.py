
import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

st.set_page_config(
    page_title="Sign Here - ASL Recognition",
    page_icon="🤟",
    layout="centered"
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "sign_here_cnn.keras",
        compile=False
    )

@st.cache_data
def load_mapping():
    with open("class_mapping.json", "r") as file:
        return json.load(file)

model = load_model()
class_mapping = load_mapping()

st.title("🤟 Sign Here")
st.subheader("ASL Alphabet Recognition")

st.write(
    "Upload an image of a static ASL alphabet handshape. "
    "The model recognizes 24 letters: A–I and K–Y."
)

uploaded_file = st.file_uploader(
    "Upload an ASL hand image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")

    st.image(
        image,
        caption="Uploaded image",
        width=250
    )

    if st.button("Recognize Sign"):
        resized_image = image.resize((28, 28))

        image_array = np.array(resized_image)
        image_array = image_array.astype("float32") / 255.0
        image_array = image_array.reshape(1, 28, 28, 1)

        predictions = model.predict(image_array, verbose=0)[0]

        predicted_index = int(np.argmax(predictions))
        predicted_letter = class_mapping[str(predicted_index)]
        confidence = float(predictions[predicted_index])

        st.success(f"Predicted Letter: {predicted_letter}")
        st.metric("Confidence", f"{confidence:.2%}")

        st.write("Top 5 predictions:")

        top_indices = np.argsort(predictions)[-5:][::-1]

        for index in top_indices:
            letter = class_mapping[str(int(index))]
            score = float(predictions[index])
            st.write(f"**{letter}** — {score:.2%}")
            st.progress(score)

st.divider()

st.caption(
    "Note: This prototype is designed for images similar to "
    "the Sign Language MNIST dataset. Arbitrary phone photographs "
    "and live webcam recognition have not been validated."
)
