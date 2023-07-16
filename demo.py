# import module
import streamlit as st
import numpy as np
import spectral
import matplotlib.pyplot as plt
import numpy as np
import spectral.io.envi as envi
from sklearn.decomposition import PCA
from PIL import Image
import base64
from io import BytesIO


im = Image.open("mango.ico")

st.set_page_config(page_title="Ripeness Prediction", page_icon=im,layout="wide")





import base64
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('img4.jpg')    
    





# Title
st.title("Measuring Fruit Ripeness Using Hyperspectral Imaging")
st.text('')

# Selection box
fruits = st.selectbox("SELECT A FRUIT : ", ['KIWI', 'MANGO', 'PAPAYA', 'AVACADO'])

# print the selected hobby
st.write("Your Selected Fruit Is: ", fruits)



# Render a file uploader control
uploaded_file_1 = st.file_uploader("Choose An .hdr File", type=["hdr"])
uploaded_file_2 = st.file_uploader("Choose An .bin File", type=["bin"])



# Load the saved model
from tensorflow.keras.models import load_model
#from keras.models import load_model
# Load the saved model
model = load_model('fruit_model.h5')





def check_new_image(image, model):
    
    # Preprocess the image using PCA
    flat_image = np.reshape(image, (image.shape[0] * image.shape[1], image.shape[2]))
    pca = PCA(n_components=100)
    pca.fit(flat_image)
    flat_image = pca.transform(flat_image)
    reshaped_image = np.reshape(flat_image, (image.shape[0], image.shape[1], 100))
    
    # Predict the class of the image using the trained model
    prediction = model.predict(np.array([reshaped_image]))
    class_index = np.argmax(prediction)
    
    # Print the predicted class index
    print("Predicted class index: ", class_index)
    if class_index == 1:
        return "Un-ripe"
    elif class_index == 0:
        return "Ripe"
    else:
        return "Over-ripe"
    #return class_index

# display the name when the submit button is clicked
# .title() is used to get the input text string        
if(st.button('Predict')):
    if uploaded_file_1 is not None and uploaded_file_2 is not None:
        check_img = spectral.envi.open(uploaded_file_1.name,uploaded_file_2.name).load()
        #st.write(check_img.shape)
        st.success(check_new_image(check_img, model))
    else:
        st.write("Please upload image. Make sure your image is in HDR/BIN Format.")        




#Created by Tanay Nikam
