import streamlit as st
from keras.models import load_model
from PIL import Image
from util import classify,set_background

set_background("./bg/3386851.jpg")
#set title
st.title('Pneumonia Classification')

#set header
st.header('Please upload a chest X-ray image')

#upload file
file = st.file_uploader('',type=['jpeg','jpg','png'],label_visibility='visible')

#load classifier
model = load_model('model/tuned_resnet.h5')

#load classname
with open('./model/labels.txt','r') as f:
    class_names = ['PNEUMONIA','NORMAL']
    f.close()
print(class_names)

#display image
if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image,use_column_width=True)

    #classify image
    class_name, confidence_score = classify(image,model,class_names)

    #write classification
    st.write("## {}".format(class_name))
    st.write("### score: {}".format(confidence_score))