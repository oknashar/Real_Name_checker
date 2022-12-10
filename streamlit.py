import streamlit as st 
import pandas as pd 
import numpy as np 
import time
import keras
import pickle
from shared_param import model_path,tokenizer_path
from test import model_and_tokenizer,predict_dl
from models import model

# import the model and tokenizer

modeel ,tokenizer = model_and_tokenizer(model_path,tokenizer_path,model)

# set title to the streamlit
st.title("اختبار للاسم هل هو حقيقي ام مزيف")

# field to upload text 
text_uploader = st.text_input("ضع الاسم هنا")

if text_uploader is not None:
    # add button to predict 
    pred_button = st.button("اختبر الاسم")


    # check if the predict button pressed 
    if pred_button:
        # split the uploaded text to list of 3 words 
        text=" ".join(text_uploader.split("*"))

        #start to calculate the time
        start_time = time.time()

        # make prediction
        result=predict_dl(modeel,tokenizer,text)

        #end the time and calulate all the time
        fulltime =time.time() - start_time

        #check the confidence of the model
        if result > 0.40:
            st.write("الاسم هو : " , text)
            st.write("الاسم حقيقي بنسبة : " , round(result,3))
            st.write("مدة الاختبار: " ,round(fulltime,3))
        else:
            st.write("الاسم هو : " , text)
            st.write("الاسم مزيف بنسبة : " , round(1- result,3))
            st.write("الاسم حقيقي بنسبة : " , round(result,3))
            st.write("مدة الاختبار : " ,round(fulltime,3))

    
