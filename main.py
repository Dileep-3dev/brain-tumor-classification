import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import pandas as pd
import os
from collections import Counter

# Load the trained model
model = tf.keras.models.load_model('best_model.keras')

# Class labels
class_labels = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

# Function to preprocess the image for prediction
def preprocess_image(img):
    img = img.resize((224, 224))  # Resize to the size expected by the model
    img = np.array(img) / 255.0   # Normalize the image
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Function to apply image enhancement
def enhance_image(img, enhancement_type='contrast'):
    if enhancement_type == 'contrast':
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(2)
    elif enhancement_type == 'sharpness':
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(2)
    return img

# Function to display the model summary
def display_model_summary():
    with st.expander("Model Summary"):
        model_summary = []
        model.summary(print_fn=lambda x: model_summary.append(x))  # Capture summary output
        st.text("\n".join(model_summary))  # Display summary as text

# Function to analyze dataset distribution
def analyze_dataset(dataset_path, dataset_type):
    data_dir = os.path.join(dataset_path, dataset_type)
    class_counts = {}
    for class_label in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_label)
        if os.path.isdir(class_dir):
            class_counts[class_label] = len(os.listdir(class_dir))
    return class_counts

# Function to plot dataset distribution
def plot_dataset_distribution(dataset_distribution, dataset_type):
    labels = list(dataset_distribution.keys())
    values = list(dataset_distribution.values())

    fig, ax = plt.subplots()
    ax.bar(labels, values, color='skyblue')
    ax.set_title(f"{dataset_type.capitalize()} Dataset Distribution")
    ax.set_ylabel("Number of Samples")
    ax.set_xlabel("Class Labels")
    plt.xticks(rotation=45)
    st.pyplot(fig)

# Streamlit interface
def main():
    st.title("Brain Tumor Detection")
    st.write("Upload an MRI image to predict whether it contains a brain tumor.")

    # Dataset Analysis Section
    dataset_path = 'F:/New folderDileep/data'
    st.subheader("Dataset Analysis")
    if st.button("Show Dataset Insights"):
        train_distribution = analyze_dataset(dataset_path, "train")
        test_distribution = analyze_dataset(dataset_path, "test")

        st.write("### Training Dataset Distribution")
        plot_dataset_distribution(train_distribution, "train")

        st.write("### Testing Dataset Distribution")
        plot_dataset_distribution(test_distribution, "test")

    # Image Enhancement Options
    st.subheader("Prediction Section")
    enhancement_type = st.selectbox("Choose Image Enhancement", ["None", "Contrast", "Sharpness"])

    # File uploader widget
    uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Load and display the image
        img = Image.open(uploaded_file)
        if enhancement_type != "None":
            img = enhance_image(img, enhancement_type)
        st.image(img, caption="Uploaded Image", use_column_width=True)

        # Preprocess the image and make prediction
        processed_img = preprocess_image(img)
        prediction = model.predict(processed_img)
        predicted_class = class_labels[np.argmax(prediction)]
        confidence = np.max(prediction)

        # Display prediction result
        st.write(f"Prediction: **{predicted_class}**")
        st.write(f"Confidence: **{confidence * 100:.2f}%**")

        # Display prediction graph
        st.subheader("Prediction Confidence")
        fig, ax = plt.subplots()
        ax.bar(class_labels, prediction[0], color='lightgreen')
        ax.set_ylabel("Confidence")
        ax.set_title("Model Prediction Confidence")
        st.pyplot(fig)

        # Display pie chart of class prediction distribution
        fig_pie, ax_pie = plt.subplots()
        ax_pie.pie(prediction[0], labels=class_labels, autopct='%1.1f%%', startangle=90)
        ax_pie.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        ax_pie.set_title("Prediction Distribution")
        st.pyplot(fig_pie)

    # Display model summary
    st.subheader("Model Summary")
    display_model_summary()

if __name__ == "__main__":
    main()