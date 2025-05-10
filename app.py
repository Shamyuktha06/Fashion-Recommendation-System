import numpy as np
import pickle as pkl
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPool2D

from sklearn.neighbors import NearestNeighbors
import os
from numpy.linalg import norm
import streamlit as st
import base64
from io import BytesIO
from PIL import Image

# Function to add a background image from a local file
def add_bg_from_local(image_file):
    with open(image_file, "rb") as file:
        encoded_string = base64.b64encode(file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded_string}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Set the background
add_bg_from_local("C:/frsda deep proj/pic4.jpg")

st.header('Fashion Recommendation System')

# Explanation of features
st.write("""
Welcome to the **Fashion Recommendation System**! 

This application leverages a powerful deep learning model to help you discover visually similar fashion items based on the image you upload. Simply upload an image of a fashion item, such as a shirt, dress, or accessory, and the system will analyze its features and find similar items in our collection.
""")

# Load features and filenames
Image_features = pkl.load(open('Images_features.pkl', 'rb'))
filenames = pkl.load(open('filenames.pkl', 'rb'))

# Define the feature extraction function
def extract_features_from_images(upload_file, model):
    img = image.load_img(upload_file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_expand_dim = np.expand_dims(img_array, axis=0)
    img_preprocess = preprocess_input(img_expand_dim)
    result = model.predict(img_preprocess).flatten()
    norm_result = result / norm(result)
    return norm_result

# Load the pre-trained model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tf.keras.models.Sequential([model, GlobalMaxPool2D()])

# Set up Nearest Neighbors
neighbors = NearestNeighbors(n_neighbors=10, algorithm='brute', metric='euclidean')
neighbors.fit(Image_features)

# File uploader for user to upload an image
upload_file = st.file_uploader("Upload an Image of a Fashion Item")

# Option for the number of recommendations
num_recommendations = st.slider("Number of Recommendations", 1, 10, 5)

# Image Similarity Threshold Slider
similarity_threshold = st.slider("Adjust Similarity Threshold", 0.0, 1.0, 0.75)

# Favorites list to store saved images
favorites = []

if upload_file is not None:
    # Save the uploaded image
    os.makedirs('upload', exist_ok=True)
    file_path = os.path.join('upload', upload_file.name)
    with open(file_path, 'wb') as f:
        f.write(upload_file.getbuffer())

    # Display uploaded image
    st.subheader('Uploaded Image')
    st.image(upload_file)

    # Add spinner while processing
    with st.spinner("Finding recommendations..."):
        # Extract features from uploaded image and get recommendations
        input_img_features = extract_features_from_images(file_path, model)
        distances, indices = neighbors.kneighbors([input_img_features], n_neighbors=10)

    # Filter recommendations based on similarity threshold
    filtered_indices = [i for i, d in enumerate(distances[0]) if d <= similarity_threshold]

    if filtered_indices:
        # Display recommended images with distance scores and download option
        st.subheader('Filtered Recommended Images')
        recommendation_cols = st.columns(len(filtered_indices[:num_recommendations]))

        for i, idx in enumerate(filtered_indices[:num_recommendations]):
            with recommendation_cols[i]:
                # Load recommended image
                recommended_img_path = filenames[indices[0][idx]]
                recommended_img = Image.open(recommended_img_path)

                # Display image with distance score
                st.image(recommended_img, caption=f"Distance: {distances[0][idx]:.2f}", use_container_width=True)

                # Display filename as metadata
                st.write(f"Filename: {os.path.basename(recommended_img_path)}")

                # Save to Favorites button
                if st.button("Save to Favorites", key=f"favorite_{i}"):
                    favorites.append(recommended_img_path)
                    st.write(f"Saved {os.path.basename(recommended_img_path)} to Favorites!")

                # Download button for recommended image
                buffered = BytesIO()
                recommended_img.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                href = f'<a href="data:file/jpg;base64,{img_str}" download="{os.path.basename(recommended_img_path)}">Download Image</a>'
                st.markdown(href, unsafe_allow_html=True)
    else:
        st.write("No recommendations found that match the specified similarity threshold.")

# Display Favorites section if there are any saved recommendations
if favorites:
    st.subheader("Favorites")
    for fav_path in favorites:
        fav_img = Image.open(fav_path)
        st.image(fav_img, caption=f"Saved Item: {os.path.basename(fav_path)}", use_container_width=True)

