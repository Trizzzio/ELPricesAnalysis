import pandas as pd
import numpy as np
import datetime as datetime

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


