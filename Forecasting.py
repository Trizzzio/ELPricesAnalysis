import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

#Cleaning price and production data DK
df_price=pd.read_csv("data/GUI_ENERGY_PRICES_DK.csv")
df_production=pd.read_csv("data/AGGREGATED_GENERATION_PER_TYPE_DK.csv")

df_price_clean=df_price.copy()
df_production_clean=df_production.copy()

print(df_price_clean.columns)
print(df_production_clean.columns)

##Fixing time, dropping columns and observations, setting index, ...
df_price_clean["datetime"]=pd.to_datetime(df_price_clean['MTU (CET/CEST)'].str.split('-').str[0],format="mixed", dayfirst=True)
df_production_clean['MTU (CET/CEST)']=df_production_clean['MTU (CET/CEST)'].str.replace(r"\s*\(.*\)", "", regex=True)
df_production_clean["datetime"]=pd.to_datetime(df_production_clean['MTU (CET/CEST)'].str.split('-').str[0],format="mixed", dayfirst=True)

df_price_clean=df_price_clean.drop(columns=["Sequence",'MTU (CET/CEST)',
    "Intraday Price (EUR/MWh)",
    "Intraday Period (CET/CEST)",
    "Area"])

df_production_clean=df_production_clean.drop(columns=["Area"])

df_price_clean.rename(columns={'Day-ahead Price (EUR/MWh)':"price"},inplace=True)
df_price_clean["price"]=pd.to_numeric(df_price_clean["price"])
df_production_clean.rename(columns={'Production Type':"type"},inplace=True)
df_production_clean.rename(columns={'Generation (MW)':"mw"},inplace=True)
df_production_clean["mw"]=df_production_clean["mw"].replace("n/e",np.nan)
df_production_clean["mw"]=df_production_clean["mw"].replace("-",np.nan)
df_production_clean["mw"]=pd.to_numeric(df_production_clean["mw"])

df_price_clean=df_price_clean.set_index("datetime")
df_price_clean=df_price_clean.sort_index()

cutoff=df_price_clean.index.max()
df_production_clean=df_production_clean[df_production_clean["datetime"]<=cutoff]

df_production_clean=df_production_clean.set_index("datetime")
df_production_clean=df_production_clean.sort_index()

##Pivoting production file and merging

df_production_wide=df_production_clean.pivot_table(
    index="datetime",
    columns="type",
    values="mw",aggfunc="mean"
)

print(df_production_wide.head())
print(df_price_clean.head())

df_merged=df_price_clean.join(df_production_wide,how="inner")

print(df_merged.head())

####ARIMA

#check for stationarity

result = adfuller(df_merged["price"])
print(f"ADF STatistic: {result[0]}")
print(f"p-value: {result[1]}")

#Plot ACF and PACF to shooce p and q
fig, ax=plt.subplots(2,1,figsize=(12,6))
plot_acf(df_merged["price"],ax=ax[0],lags=40)
plot_pacf(df_merged["price"],ax=ax[1],lags=40)
plt.tight_layout()
plt.show()

#Fit ARIMA
model = ARIMA(df_merged["price"],order=(2,1,2))
model_fit = model.fit()
print(model_fit.summary())

#Forecast
n_steps=30
forecast=model_fit.get_forecast(steps=n_steps)
mean_forecast=forecast.predicted_mean
conf_int=forecast.conf_int()

forecast_index = pd.date_range(start=df_merged.index[-1]+pd.Timedelta(days=1),periods=n_steps,freq=("D"))

#Plot
plt.figure(figsize=(12,6))
plt.plot(df_merged["price"], label="Historical")
plt.plot(forecast_index, mean_forecast, label="Forecast", color="green")
plt.fill_between(forecast_index, conf_int.iloc[:,0],conf_int.iloc[:,1], color="green", alpha=0.3)
plt.title("ARIMA Forecast")
plt.legend()
plt.show()






