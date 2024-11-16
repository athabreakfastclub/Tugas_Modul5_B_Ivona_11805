import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Define the model path
model_path = r'"C:\Users\ASUS\Downloads\Introduction to Deep Learning (Praktek)\Introduction to Deep Learning (Praktek)\best_model_tf.h5"'

# Load the model
if os.path.exists(model_path):
    try:
        # Reduce verbosity of TensorFlow
        tf.get_logger().setLevel('ERROR')
        model = tf.keras.models.load_model(model_path, compile=False)

        # Class names for Fashion MNIST
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

        # Function to preprocess the image
        def preprocess_image(image):
            image = image.resize((28, 28))  # Resize to 28x28 pixels
            image = image.convert('L')  # Convert to grayscale
            image_array = np.array(image) / 255.0  # Normalize
            image_array = image_array.reshape(1, 28, 28, 1)  # Reshape into 4D array
            return image_array

        # Streamlit UI
        st.title("Fashion MNIST Image Classifier")
        st.write("Upload a fashion item image (e.g., shoes, bags, clothes), and the model will predict its class.")

        # File uploader for image input
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        # Display the uploaded image and "Predict" button
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # "Predict" button
            if st.button("Predict"):
                # Process image and predict
                processed_image = preprocess_image(image)
                predictions = model.predict(processed_image)[0]
                predicted_class = np.argmax(predictions)
                confidence = predictions[predicted_class] * 100  # Ensure confidence in percentage

                # Display prediction results
                st.write("### Prediction Results")
                st.write(f"Predicted Class: **{class_names[predicted_class]}**")
                st.write(f"Confidence: **{confidence:.2f}%**")

    except Exception as e:
        st.error(f"Error: {str(e)}")

else:
    st.error("Model file not found.")
