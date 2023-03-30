import streamlit as st
import numpy as np
import pandas as pd
from fer import FER
from PIL import Image
import cv2
import altair as alt
# import the frameworks, packages and libraries
import streamlit as st
from PIL import Image
from io import BytesIO
import numpy as np
import cv2  # computer vision
from multiapp import MultiApp
import cv2
import numpy as np
import sys
from tqdm import tqdm


def Wel():
   st.markdown('<h1 style="color: black;font-family:cursive;"> Photo Notebook </h1',unsafe_allow_html=True)
   st.image("ss.jpg")
   st.write(
        """
        ### Photos have lot of Memories, lets play with that!!!!...
        """
    )

def page1():
    def getEmotions(img):
        detector  = FER(mtcnn=True)
        result = detector.detect_emotions(img)
        data  = result[0]['emotions']
        if data is None:
            st.write('No result')
            return False
        else:
            return data
    title_container=st.container()
    value,picture=st.columns(2,gap='small')
    #image= st.image("kiss.jpg")
    with title_container:
     with picture:
         st.image("kiss.jpg",width=44)
     with value:
         st.markdown('<h1 style="color: black;font-family: cursive;"> Emotions Bomb </h1',unsafe_allow_html=True)

    st.write(
        """
        ### Every picture says some story, lets generate yours....
        """
    )
    file = st.file_uploader('Please upload an image file', type = ['jpg','png'])
    if file is None:
      st.write("")
    else:
      image = Image.open(file)
      img = np.array(image)
      
      st.image(image,width=350)
      st.write(pd.DataFrame(getEmotions(img),index=[0]))

def Page2():
    def convertto_watercolorsketch(inp_img):
       
        Img = cv2.resize(inp_img, (740,480))
        GrayImg = cv2.cvtColor(src=Img, code=cv2.COLOR_BGR2GRAY)
        SmoothImg = cv2.medianBlur(src=GrayImg, ksize=5)

        Edges = cv2.adaptiveThreshold(src=SmoothImg, maxValue=255, \
        adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C, \
        thresholdType=cv2.THRESH_BINARY, blockSize=7, C=6)

        ColorImg = cv2.bilateralFilter(src=Img, d=9, sigmaColor=220, \
            sigmaSpace=200)

        img_water_color = cv2.bitwise_and(src1=ColorImg,src2=ColorImg,mask=Edges)
        return(img_water_color)
  
    def pencilsketch(inp_img):
       img_pencil_sketch, pencil_color_sketch = cv2.pencilSketch(
       inp_img, sigma_s=50, sigma_r=0.07, shade_factor=0.0825)
       return(img_pencil_sketch)
    
    def load_an_image(image):
       img = Image.open(image)
       return img
    def main():
       st.markdown('<h1 style="color: black;font-family: cursive;"> Lets Paint your Memories 🤳🏽 </h1',unsafe_allow_html=True)
       st.write("This will paint your photo in two different styles!!. Have Funnn!😀")
       st.subheader("Please Upload your image🤓")
      
       # image file uploader
       image_file = st.file_uploader("Upload Images", type=["png", "jpg", "jpeg"])
       if image_file is not None:
            option = st.selectbox('Hmmm, Which style you prefer? 😎',
                              ('Cartoonify Me',
                               'Pencil sketch Splash'))
            if option == 'Cartoonify Me':
               image = Image.open(image_file)
               final_sketch = convertto_watercolorsketch(np.array(image))
               im_pil = Image.fromarray(final_sketch)
               col1, col2 = st.columns(2)
               with col1:
                st.header("Original Image")
                st.image(load_an_image(image_file), width=250)
               with col2:
                st.header("Your Cartoon Version")
                st.image(im_pil, width=250)
                buf = BytesIO()
                img = im_pil
                img.save(buf, format="JPEG")
                byte_im = buf.getvalue()
                st.download_button(
                    label="Download image",
                    data=byte_im,
                    file_name="watercolorsketch.png",
                    mime="image/png"
                )
  
            if option == 'Pencil sketch Splash':
               image = Image.open(image_file)
               final_sketch = pencilsketch(np.array(image))
               im_pil = Image.fromarray(final_sketch)
               col1, col2 = st.columns(2)
               with col1:
                st.header("Original Image")
                st.image(load_an_image(image_file), width=250)
  
               with col2:
                st.header("Pencil Sketch Splash")
                st.image(im_pil, width=250)
                buf = BytesIO()
                img = im_pil
                img.save(buf, format="JPEG")
                byte_im = buf.getvalue()
                st.download_button(
                    label="Download image",
                    data=byte_im,
                    file_name="watercolorsketch.png",
                    mime="image/png")
    if __name__ == '__main__':
       main()


   

app = MultiApp()
app.add_app("Welcome Page",Wel)
app.add_app("Emotions Detector",page1)
app.add_app("Photo Editor",Page2)
app.run()  
