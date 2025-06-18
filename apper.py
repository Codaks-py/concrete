
import streamlit as st
import pickle
import pandas as pd
import numpy as np


st.title('CONCRETE COMPRESSIVE STRENGHT PREDICTOR')
st.caption('This is a simple app that predicts the compressive strenght of a concrete based on features including the components that makes up a concrete.')

# st.subheader('What is Compressive strenght of a concrete?')
#st.caption('Compressive strength of concrete refers to its ability to withstand compressive forces, essentially how much pressure it can handle before failing or breaking.')




lm_pick = open('strenght.pkl', 'rb')
scaler_pick = open('strenght_scaler.pkl', 'rb')

lme = pickle.load(lm_pick)
scaler = pickle.load(scaler_pick)

#st.write(lm)
st.write()

with st.form('Form'):
    cement = st.number_input('The amount of cements used (Bags)', min_value=0.0)
    slag = st.number_input('The volume of the blast furnace slag', min_value=0.0)
    flyash = st.number_input('The volume of the Fly ash:',min_value=0.0)
    water = st.number_input('The amount of water used(Litres):', min_value=0.0)
    superplasticer = st.number_input('The weight of the admixture(Superplasticizer)', min_value=0.0)
    coarseagg =  st.number_input('The density of the Coarse aggregates:', min_value=0.0)
    fineagg = st.number_input('The density of the Fine aggregates: ', min_value=0.0)
    age = st.number_input('Time spent after molding: ', min_value=0.0)
    submitted = st.form_submit_button('Predict')

if submitted:
    features = pd.DataFrame({'cement': [cement], 
                             'slag' : [slag],
                              'flyash': [flyash],
                               'water' :[water],
                                'superplasticizer' : [superplasticer], 
                                'coarseaggregate':[coarseagg],
                                'fineaggregate': [fineagg], 
                                'age': [age]})
    dummy_features = pd.get_dummies(features)
    scaled_features = scaler.transform(dummy_features)

    prediction = lme.predict(scaled_features)


    st.write(f'Predicted Strenght:  {prediction[0]:.2f}')

