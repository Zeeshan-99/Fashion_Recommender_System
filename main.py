import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

# Load pre-trained model and data
feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# Set theme to dark
st.markdown(
    """
    <style>
    .reportview-container {
        background: #1E1E1E;
        color: #FFFFFF;
    }
    .sidebar .sidebar-content {
        background: #1E1E1E;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.title('Fashion Recommender System')

# File upload
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

# Function to save uploaded file
def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0

# Function for feature extraction
def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

# Function for recommendation
def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices

# Process uploaded file and make recommendations
if uploaded_file is not None:
    if st.button('Get Recommendations'):
        if save_uploaded_file(uploaded_file):
            # Display uploaded image
            display_image = Image.open(uploaded_file)
            st.image(display_image, caption='Uploaded Image', use_column_width=True)

            # Feature extraction
            features = feature_extraction(os.path.join("uploads", uploaded_file.name), model)
            st.text("Image Features: {}".format(features))

            # Recommendation
            indices = recommend(features, feature_list)

            # Display recommendations
            st.subheader("Recommended Products:")
            cols = st.columns(5)
            for i, col in enumerate(cols):
                col.image(filenames[indices[0][i]], use_column_width=True)
        else:
            st.error("Error occurred during file upload")
