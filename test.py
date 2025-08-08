import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def load_data():
    df=pd.read_csv(r"residential_energy_usage.csv",parse_dates=["Date"])
    df=df.sort_values("Date").drop_duplicates("Date")
    df["Days"]=df["Date"].dt.day_name()
    return df

def train_model(df):
    df["Datess"]=df["Date"].dt.dayofyear
    x=df[["Datess"]]
    y=df["Appliance_Usage_kWh"]
    ml=LinearRegression()
    ml.fit(x,y)
    return ml


def future_prediction(ml,sd):
    future_days=pd.DataFrame({"Datess":list(range(sd, sd+8))})
    future_prd=ml.predict(future_days)
    future_df=pd.DataFrame(
        {"Fut_Datess":future_days["Datess"],"Predicted Energy Usage":future_prd})
    return future_df

def plot_usage(df): 
    plt.figure(figsize=(9,5))
    plt.plot(df["Date"],df["Appliance_Usage_kWh"],label="Daily usage")
    plt.plot(df["Date"],df["Appliance_Usage_kWh"].rolling(window=7).mean(),label="Weekly Average")
    plt.legend()
    st.pyplot(plt)



st.set_page_config(page_title="🏠Residential Energy Consumption... ",layout="centered")

st.title("🏠 Residential Energy Cosumption  &Forecasting Dashboards")
df=load_data()
st.header("Add Todays Upgrade")
with st.form("Enter form"):
    new_date=st.date_input("Date")
    new_usage=st.number_input("Energy usage",min_value=0.0,step=0.1)
    submit=st.form_submit_button("Add the entry")
    if submit:
        new_row=pd.DataFrame(
            {"Date":[pd.to_datetime(new_date)],"Appliance_Usage_kWh":[new_usage],
             "Days":[pd.to_datetime(new_date).day_name()]})
        
        df_up=pd.concat([df,new_row],ignore_index=True)
        df_up.to_csv("residential_energy_usage.csv",index=False)
        st.rerun()
             
st.header("Energy usage trend")
plot_usage(df)

st.header(" 7 days prediction ")
ml=train_model(df)
sd=df["Date"].dt.dayofyear.max()
future_df=future_prediction(ml,sd)
st.dataframe(future_df)
