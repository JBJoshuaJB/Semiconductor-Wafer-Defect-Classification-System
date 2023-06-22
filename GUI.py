import streamlit as st
import tensorflow as tf
from keras.preprocessing import image
from keras.utils import load_img
from keras.applications.imagenet_utils import decode_predictions
from PIL import Image, ImageOps, ImageDraw
import pandas as pd
import numpy as np
import tempfile
import os
import openpyxl

img_width, img_height = 64, 64
model = tf.keras.models.load_model('defects_classification_model.h5')
classes = ['Contamination-Particle', 'Pattern defect', 'Probe Mark', 'Scratches', 'Others']

def extract_images(tif_file, output_folder):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    filename = tif_file.name
    # Load the .tif file
    with Image.open(tif_file) as tif_image:
        # Iterate over each frame/page in the .tif file
        for i in range(tif_image.n_frames):
            tif_image.seek(i)
            # Convert and save each frame as a .png file
            output_path = os.path.join(output_folder, f"{filename}_{i}.png")
            tif_image.save(output_path, "PNG")

st.image('Onsemi_logo.png', use_column_width = True)
st.title('Semiconductor Wafer Defect Classification System')
st.write('\n')
st.write('Upload an image of a semiconductor wafer to classify its defect!')
st.write('\n')

#Sidebar
st.sidebar.image('UTeM_logo.png', use_column_width = True)
st.sidebar.title(':red[TIFF File Extract]')
uploaded_file = st.sidebar.file_uploader("Choose a .tif file", type = "tif")
if uploaded_file is not None:
        #Get the uploaded file name without the extension
        filename = os.path.splitext(uploaded_file.name)[0]
        #Autoname the output folder based on the uploaded filename
        output_folder = f"{filename}(extracted)"
        #Add the Extract button
        extract_button = st.sidebar.button("Extract")
        if extract_button:
            #if output_folder.strip() != "":
                #Extract the images and save as .png
                 extract_images(uploaded_file, output_folder)
            
                 st.sidebar.success("Image extraction completed!")
        else:
            st.sidebar.warning("Waiting for extraction...")
            
st.sidebar.title(':blue[Image Upload]') 
uploaded_file = st.sidebar.file_uploader('Upload file here:', type = ['jpg', 'jpeg', 'png'])

if uploaded_file is None:
    st.text('')
    
else:
    df = pd.read_excel('Record.xlsx')
    ori_image = Image.open(uploaded_file)
    width, height = ori_image.size  #Get size of current image
    left = (width - 50) // 2  #Define coordinates for the crop
    top = (height - 50) // 2
    right = left + 50
    bottom = top + 50
    image = ori_image.crop((left, top, right, bottom))
    col1, col2 = st.columns(2)
    col1.subheader('Selected image:')
    image_with_rectangle = ori_image.copy() #Create copy of original image
    image_with_rectangle = image_with_rectangle.convert("RGB") #Convert the image to RGB mode for its appearance
    draw = ImageDraw.Draw(image_with_rectangle) #Draw rectangle on image
    draw.rectangle([(left, top), (right, bottom)], outline = 'red', width = 2) 
    del draw
    col1.image(image_with_rectangle, caption = uploaded_file.name, width=300)
    col2.subheader('Image of defect:')
    col2.image(image, width = 300)
    st.write('\n')
    with tempfile.NamedTemporaryFile(suffix = ".png", dir=tempfile.gettempdir(), delete=False) as temp_file:
        img_path = temp_file.name
        image.save(img_path)
        image = tf.keras.utils.load_img(img_path, target_size=(64, 64))

    image = tf.keras.utils.img_to_array(image)
    image = image / 255
    image = np.expand_dims(image, axis = 0)
    result = model.predict(image)

    type = ""

    if result[0][0] > result[0][1] and result[0][0] > result[0][2] and result[0][0] > result[0][3] and result[0][0] > result[0][4]:
        type = "Contamination particle"
    elif result[0][1] > result[0][0] and result[0][1] > result[0][2] and result[0][1] > result[0][3] and result[0][1] > result[0][4]:
        type = "Pattern defect"
    elif result[0][2] > result[0][0] and result[0][2] > result[0][1] and result[0][2] > result[0][3] and result[0][2] > result[0][4]:
        type = "Probe mark"
    elif result[0][3] > result[0][0] and result[0][3] > result[0][1] and result[0][3] > result[0][2] and result[0][3] > result[0][4]:
        type = "Scratches"
    elif result[0][4] > result[0][0] and result[0][4] > result[0][1] and result[0][4] > result[0][2] and result[0][4] > result[0][3]:
        type = "Others"

    st.write('\n')
    st.write('The defect is classified as:') 
    st.markdown('<span style = "color:#FF785B; font-size: 48px; font-weight: bold;">' + type + '</span>', unsafe_allow_html = True)
    st.write("The similarity score is approximately:", result)

    df.loc[len(df.index)] = [uploaded_file.name, type]

    df.to_excel('Record.xlsx', index = False)