import streamlit as st
import easyocr
import numpy as np
from transformers import pipeline

from PIL import Image


reader=easyocr.Reader(['hi','en'])

qa_pipeline=pipeline('question-answering',model='distilbert-base-cased-distilled-squad',framework='pt')



def ocr_image(uploaded_image):
    image=Image.open(uploaded_image)
    image_np=np.array(image)
    
    result=reader.readtext(image_np,detail=0,paragraph=True)
    return " ".join(result)

def search_keyword(extracted_text,keyword):
    search_result=qa_pipeline({
        'context':extracted_text,
        'question':f"Where is{keyword} mentioned in text?"
    })
    return search_result['answer']

st.title("NLP,OCR and Document Search Web Application")

st.write("Upload an image containing text in Hindi and English,extract the text, and search for keywords or NLP imsights.")


uploaded_image=st.file_uploader("Upload an image file (JPEG,PNG)",type=["jpg","jpeg","png"])


if uploaded_image:
    st.image(uploaded_image,caption='Uploaded Image',use_column_width=True)
    st.write("Extracting text from the image....")
    extracted_text=ocr_image(uploaded_image)
    
    st.write("### Extracted Text")
    st.write(extracted_text)
    
    
    st.write("### Search in Extracted Text")
    keyword=st.text_input("Enter a keyword to search:")
    
    if keyword:
        search_result=search_keyword(extracted_text,keyword)
        st.write(f"Search Result:{search_result}")
        
        
st.write("@ Made by Tushar Sharma")
    
