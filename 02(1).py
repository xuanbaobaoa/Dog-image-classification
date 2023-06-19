import streamlit as st
import os
from fastai.vision.all import *
from PIL import Image
import pathlib
temp=pathlib.PosixPath
pathlib.PosixPath=pathlib.WindowsPath

path=os.path.dirname(os.path.abspath(__file__))
model_path=os.path.join(path,'export_1.pkl')
learn_inf =load_learner(model_path)

pathlib.PosixPath=temp
st.title('Welcome to use this app')
st.balloons()
#上传文件
uploaded_file=st.file_uploader("Choose an image...",type=["jpg","png","jpeg"])

if uploaded_file is not None:
    img=PILImage.create(uploaded_file)
    st.image(img.to_thumb(500,500),caption='Your Image')
    pred,pred_idx,probs =learn_inf.predict(img)
    st.write(f"Prediction: {pred}; Probability: {probs[pred_idx]:.04f}")
    if pred == 'n02099601-golden_retriever':
        image1 = Image.open('C:/Users/86198/Desktop/新建文件夹/golden_retriever1.jpg', 'r')
        image2 = Image.open('C:/Users/86198/Desktop/新建文件夹/golden_retriever2.jpg', 'r')
        image3 = Image.open('C:/Users/86198/Desktop/新建文件夹/golden_retriever3.jpg', 'r')
        image4 = Image.open('C:/Users/86198/Desktop/新建文件夹/golden_retriever4.jpg', 'r')
        image5 = Image.open('C:/Users/86198/Desktop/新建文件夹/golden_retriever5.jpg', 'r')
        image6 = Image.open('C:/Users/86198/Desktop/新建文件夹/golden_retriever6.jpg', 'r')
        st.write('Labrador_positive comments:')
        st.image(image1)
        st.image(image2)
        st.image(image3)
        st.write('Labrador_negative comments:')
        st.image(image4)
        st.image(image5)
        st.image(image6)
    if pred == 'n02111889-Samoyed':
        image1 = Image.open('C:/Users/86198/Desktop/新建文件夹/Samoyed1.jpg', 'r')
        image2 = Image.open('C:/Users/86198/Desktop/新建文件夹/Samoyed2.jpg', 'r')
        image3 = Image.open('C:/Users/86198/Desktop/新建文件夹/Samoyed3.jpg', 'r')
        image4 = Image.open('C:/Users/86198/Desktop/新建文件夹/Samoyed4.jpg', 'r')
        image5 = Image.open('C:/Users/86198/Desktop/新建文件夹/Samoyed5.jpg', 'r')
        image6 = Image.open('C:/Users/86198/Desktop/新建文件夹/Samoyed6.jpg', 'r')
        st.write('Samoyed_positive comments:')
        st.image(image1)
        st.image(image2)
        st.image(image3)
        st.write('Samoyed_negative comments:')
        st.image(image4)
        st.image(image5)
        st.image(image6)
    if pred == 'n02110185-Siberian_husky':
        image1 = Image.open('C:/Users/86198/Desktop/新建文件夹/hashiqi1.jpg', 'r')
        image2 = Image.open('C:/Users/86198/Desktop/新建文件夹/hashiqi2.jpg', 'r')
        image3 = Image.open('C:/Users/86198/Desktop/新建文件夹/hashiqi3.jpg', 'r')
        image4 = Image.open('C:/Users/86198/Desktop/新建文件夹/hashiqi4.jpg', 'r')
        image5 = Image.open('C:/Users/86198/Desktop/新建文件夹/hashiqi5.jpg', 'r')
        image6 = Image.open('C:/Users/86198/Desktop/新建文件夹/hashiqi6.jpg', 'r')
        st.write('positive comments:')
        st.image(image1)
        st.image(image2)
        st.image(image3)
        st.write('negative comments:')
        st.image(image4)
        st.image(image5)
        st.image(image6)

