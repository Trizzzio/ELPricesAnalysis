import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from Datamanipulation import df_clean

#Check out basic statistics
print(df_clean.describe())

#Check time coverage
print(f"Start: {df_clean.index.min()}, End: {df_clean.index.max()}")
print(f"Frequency: {pd.infer_freq(df_clean.index)}")

#Add time features for analysis
df_clean["hour"]=df_clean.index.hour
df_clean["weekday"]=df_clean.index.dayofweek
df_clean["month"]=df_clean.index.month

print(df_clean.head)

#Hourly price profile

sns.set(style="whitegrid")
hourly_avg=df_clean.groupby("hour")["price"].mean()
weekday_avg=df_clean.groupby("weekday")["price"].mean()

plt.figure(figsize=(10,5))
sns.lineplot(x=hourly_avg.index, y=hourly_avg.values)
plt.title("Average Hourly Prices")
plt.xlabel("Hours")
plt.ylabel("Hourly Prices")
plt.xticks(range(0,24))
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,5))
sns.lineplot(x=weekday_avg.index, y=weekday_avg.values)
plt.title("Average Weekday Prices")
plt.xlabel("Weekdays")
plt.ylabel("Prices")
plt.xticks(range(0,6))
plt.tight_layout()
plt.show()