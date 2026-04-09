import streamlit as st
import pandas as pd

st.title("MLOps Streamlit Apps :coffee:")
isim=st.text_input('isminizi giriniz',max_chars=20)
#st.video("data/secret_of_success.mp4")
#st.camera_input("camera")
st.date_input("choose")
st.time_input("time")
st.text_input("enter password", type="password")
st.radio("status",("married","single","divorced"))
st.image("data\image_01.jpg")