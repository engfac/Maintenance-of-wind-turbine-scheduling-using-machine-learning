# Core Pkgs
import streamlit as st

# EDA Pkgs
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

import xgboost as xg
# Utils
import os
import joblib
import hashlib
# passlib,bcrypt

# Data Viz Pkgs
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
# DB
from manage_db import *

# Password 
def generate_hashes(password):
	return hashlib.sha256(str.encode(password)).hexdigest()

feature_names_best = ['Wind speed', 'Generator speed', 'Blade angle', 'Wind direction', 'Ambient temperature']



# Load ML Models
def load_model(model_file):
	loaded_model = pickle.load(open(model_file,'rb'))
   
 
 
	return loaded_model

def verify_hashes(password,hashed_text):
	if generate_hashes(password) == hashed_text:
		return hashed_text
	return False

def main():
    """APP"""
    st.title(":blue[Time Scheduled Preventive maintenance for wind turbines:date:]")

    menu = ["Home", "Login", "SignUp"]
    submenu = ["Plot", "Prediction"]

    choice = st.sidebar.selectbox("Menu", menu)
    if choice == "Home":

        st.subheader("Preventive maintenance for wind turbines is one of the maintenance solutions that are most profitable in the long term.")
        st.text("                           ")
        st.subheader("_:blue[With preventive maintenance, we achieve:]_")
        st.text("  1). Reduction of maintenance costs.")
        st.text("  2). Reduction of unforeseen breakdowns.")
        st.text("  3). Increased availability.")
        st.text("  4). Increase in production.")
        st.text("  5). Extension of operating life.")
        st.text("                           ")
        st.text("                           ")
        st.text("Prevention is one of the best ways to save financially!")

    elif choice == "Login":
            username = st.sidebar.text_input("Username")
            password = st.sidebar.text_input("Password",type='password')
            if st.sidebar.checkbox("Login"):
                create_usertable()
                hashed_pswd = generate_hashes(password)
                result = login_user(username,verify_hashes(password,hashed_pswd))
                if result:
                    st.success("Welcome {}".format(username))
                    
                
                    activity=st.selectbox("Activity",submenu)
                    if activity == "Plot":
                        st.subheader("Data vis plot")
                        df=pd.read_csv("c:/Users/DELL/Desktop/31/mlApp2/mlApp/data/D_data.csv")
                        #st.dataframe(df)
                        warnings.filterwarnings('ignore')
                        #Freq Dist plot
                        freq_df =pd.read_csv
                        
                        x = df.drop("Active power", axis = 1)
                        y = df["Active power"] 

                        st.dataframe(x)
                        st.subheader("Box plot")
                        df.plot.box(grid='True')
                        #df['Wind speed'].plot(kind='bar')
                        st.pyplot()
                        
                        
                    elif activity == "Prediction":
                        st.subheader("Prediction analytics")
                        
                        WindSpeed=st.number_input("Wind speed",step=1.,format="%.6f")
                        GeneratorSpeed=st.number_input("Generator speed",step=1.,format="%.6f")
                        BladeAngle=st.number_input("Blade angle",step=1.,format="%.6f")
                        WindDirection=st.number_input("Wind direction",step=1.,format="%.6f")
                        AmbientTemperature=st.number_input("Ambient temperature",step=1.,format="%.6f")
                        feature_list=[WindSpeed,GeneratorSpeed,BladeAngle,WindDirection,AmbientTemperature]
                        #st.write(feature_list)
                        pretty_result={"Wind speed":WindSpeed,"Generator Speed":GeneratorSpeed,"Blade Angle":BladeAngle,"Wind Direction":WindDirection,"Ambient Temperature":AmbientTemperature}
                        st.json(pretty_result)
                        single_sample=np.array(feature_list).reshape(1,-1)
                        
                        #ML
                        model_choice = st.selectbox("Select Model",["KNN","SVR","XGBoost"])
                        
                        if st.button("Predict"):
                            if model_choice == "KNN":
                                load_model1=load_model("c:/Users/DELL/Desktop/31/mlApp2/mlApp/ml_models/loaded_model_KNN")
                                prediction=load_model1.predict(single_sample)
                                st.write(prediction)

                            elif model_choice=="SVR":
                                load_model2=load_model("c:/Users/DELL/Desktop/31/mlApp2/mlApp/ml_models/loaded_model_SVR")
                                prediction1=load_model2.predict(single_sample)
                                st.write(prediction1)
                                
                                
                        
                else:
                    st.warning("Incorrect Username/Password")
                
    elif choice == "SignUp":
        new_username = st.text_input("User name")
        new_password = st.text_input("Password", type='password') 
        
        
        confirm_password = st.text_input("Confirm Password",type='password')
        if new_password == confirm_password:
            st.success("Password Confirmed")
        else:
            st.warning("Passwords not the same")

        if st.button("Submit"):
            create_usertable()
            hashed_new_password = generate_hashes(new_password)
            add_userdata(new_username,hashed_new_password)
            st.success("You have successfully created a new account")
            st.info("Login to Get Started")
           
            
if __name__ == '__main__':
    main()
