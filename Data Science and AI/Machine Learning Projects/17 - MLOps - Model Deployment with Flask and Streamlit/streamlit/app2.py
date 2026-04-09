import streamlit as st 
import pandas as pd

st.title('Formdan veri alma ve CSV dosyasina yazma :broken_heart:')
isim=st.text_input("Isminizi Giriniz")
sifre=st.text_input("Sifrenizi Giriniz", type='password')
dob=st.date_input("Dogum Tarihini Giriniz")
yas=st.slider('yasinizi giriniz',1,100)
mesaj=st.text_area('Mesajinizi giriniz',height=80)
if st.button('Gonder'):
    ndf=pd.DataFrame({'Isim':[isim],
                      'Sifre':[sifre],
                      'dob':[dob],
                      'yas':[yas],
                      'mesaj':[mesaj],
                      
                      
                      })
    st.write(ndf)
    ndf.to_csv('katilimformu.csv',index=False)
st.divider()