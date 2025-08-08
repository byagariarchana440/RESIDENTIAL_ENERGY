import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df=pd.read_csv("residential_energy_usage.csv",parse_dates=["Date"])

df["7Days_Avg"]=df["Appliance_Usage_kWh"].rolling(window=7).mean()
#df["Day_name"]=df["Date"].dt.day_name()

df["dates"]=df["Date"].dt.dayofyear

x = df[["Date"]]
y=df["Appliance_Usage_kWh"]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

ml=LinearRegression()
ml.fit(x_train,y_train)

datess=pd.DataFrame({"dates":range(214,220)})

prd=ml.predict(datess)

print(prd)

