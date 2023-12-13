import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Load saved models
age_model_inception = load_model('age_model_inception.h5')
gender_model_inception = load_model('gender_model_inception.h5')

# Process and predict function for InceptionV3 models
def process_and_predict_inception(file):
    # Load and resize the image to (299, 299)
    img = load_img(file, target_size=(299, 299))
    img_array = img_to_array(img)
    img_array = img_array.reshape((1, 299, 299, 3))
    img_array /= 255.0  # Normalize the image

    # Predict age and gender
    age = age_model_inception.predict(img_array)
    gender = np.round(gender_model_inception.predict(img_array))[0][0]
    gender = 'female' if gender == 1 else 'male'

    return int(age[0]), gender

# Streamlit app
st.title("Age and Gender Prediction App (InceptionV3)")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
    st.write("")
    st.write("Classifying...")

    age, gender = process_and_predict_inception(uploaded_file)

    # Stylish presentation using markdown and CSS
    st.markdown(f"**Predicted Age:** {age} years")
    
    # Use HTML and CSS for styling
    if gender == 'female':
        st.markdown('<p style="color:#FF69B4;font-size:20px;">Predicted Gender: Female</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p style="color:#6495ED;font-size:20px;">Predicted Gender: Male</p>', unsafe_allow_html=True)
