import pandas as pd

#Load csv file

df = pd.read_csv("data/GUI_ENERGY_PRICES.csv")

#data inspection

print(df.head)
print("\nColumns:")
print(df.columns)

#Ceck for missing values
print("\nMissing values:")
print(df.isnull().sum())


#Clean up code
#copy data to keep original clean
df_clean=df.copy()

#Fix time
df_clean['datetime'] = pd.to_datetime(df_clean['MTU (CET/CEST)'].str.split('-').str[0], dayfirst=True)

#Drop unwanted columns
df_clean=df_clean.drop(columns=[
'MTU (CET/CEST)',
    'Sequence',
    'Intraday Period (CET/CEST)',
    'Intraday Price (EUR/MWh)'
])

#Rename columns
df_clean.rename(columns={"Day-ahead Price (EUR/MWh)":"price"},inplace=True)

#Set datetime as index and sort
df_clean=df_clean.set_index("datetime")
df_clean=df_clean.sort_index()

print(df_clean.head())