import streamlit as st
import pickle
st.title('A model that can predict salary according to experience, exam and interwiev :heavy_dollar_sign:')
model=pickle.load(open('salary.pkl','rb'))
tecrube=st.number_input('Experience',1,10)
yazili=st.number_input('Exam',1,10)
mulakat=st.number_input('Interwiev',1,10)
if st.button('Tahmin et'):
    tahmin=model.predict([[tecrube,yazili,mulakat]])
    tahmin=round(tahmin[0][0],2)
    st.success(f'maas tahmini:{tahmin}')