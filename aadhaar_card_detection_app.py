import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

# Path to your trained model
MODEL_PATH = r"C:\Users\anant\OneDrive\coding\h5\model.h5"

# Load the trained model
def load_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

# Function to preprocess the image for the model
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize to the input size expected by the model
    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    image_array = image_array / 255.0  # Normalize
    return image_array

# Function to continuously capture video and make predictions
def capture_and_predict_continuous(model):
    st.sidebar.title("Aadhaar Card Detection App")
    st.sidebar.write("This application uses a deep learning model to verify if an Aadhaar card is correctly positioned.")

    st.title("Aadhaar Card Detection")
    st.write("Position your Aadhaar card within the frame for detection.")

    # Start video capture
    cap = cv2.VideoCapture(0)

    # Set resolution and FPS (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        st.error("Error: Could not open video source.")
        return

    # Placeholder for displaying the video stream and results
    video_placeholder = st.empty()
    result_placeholder = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture image.")
            break

        # Convert the captured frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image for model input
        image = Image.fromarray(frame_rgb)
        processed_image = preprocess_image(image)

        # Predict using the model
        predictions = model.predict(processed_image)
        prediction = np.argmax(predictions[0])  # Assuming a binary classification (correct/incorrect)

        # Display the captured frame
        video_placeholder.image(frame_rgb, caption="Live Camera Feed", use_column_width=True)

        # Display result
        if prediction == 1:
            result_placeholder.success("Aadhaar card detected correctly!")
        else:
            result_placeholder.warning("Please position the Aadhaar card correctly.")

# Footer with custom message
def custom_footer():
    st.markdown(
        """
        <style>
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #f5f5dc;
            color: black;
            text-align: center;
            padding: 10px;
        }
        </style>
        <div class="footer">
            <p>Made with ❤️ by Anantu Pillai</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# Run the app
if __name__ == "__main__":
    model = load_model()
    capture_and_predict_continuous(model)
    custom_footer()

