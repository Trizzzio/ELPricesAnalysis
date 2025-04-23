import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt


df=pd.read_csv("data/merged_prices_production.csv")
df["datetime"] = pd.to_datetime(df["datetime"])
df=df.set_index("datetime")
print(df.head())
print(df.columns)

df=pd.read_csv("data/merged_prices_production.csv")
df["datetime"] = pd.to_datetime(df["datetime"])
df=df.set_index("datetime")
print(df.head())
print(df.columns)

##Linear regression to set up work pipeline

def create_features(df, target_col="price", lags=[1,2,3], rolling=[24]):
    df=df.copy()

#Lag features
    for lag in lags:
        df[f'{target_col}_lag{lag}'] = df[target_col].shift(lag)

#Rolling means
    for r in rolling:
        df[f'{target_col}_roll{r}']=df[target_col].shift(1).rolling(window=r).mean()

#Time-based featrues

    df['hour']=df.index.hour
    df['dayofweek']=df.index.dayofweek
    df['month']=df.index.month

    return df.dropna()

df_ml=create_features(df,target_col='price')
print(df_ml.columns)
print(df_ml.head())

print(f"Total rows after dropping NaNs: {len(df)}")

##Random Forest

#Define targets and features

target = 'price'
features = [col for col in df_ml.columns if col != target]

X=df_ml[features]
y=df_ml[target]



#Time-based split

split_index = int(len(df_ml)*0.8)
print(f"Split index: {split_index}")
X_train, X_test=X.iloc[:split_index],X.iloc[split_index:]
y_train, y_test=y.iloc[:split_index],y.iloc[split_index:]

print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)

##Fit Random Forest

rf=RandomForestRegressor(n_estimators=100,max_depth=10,random_state=42)
rf.fit(X_train,y_train)

y_pred_rf=rf.predict(X_test)

mae_rf=mean_absolute_error(y_test,y_pred_rf)
mse_rf=mean_squared_error(y_test,y_pred_rf)

print(f"Mean absolute error: {mae_rf:.2f}")
print(f"Mean squared error: {mse_rf:.2f}")

#Plot

importances=rf.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(10,6))
plt.title("Feature Importances - Random Forest")
plt.bar(range(len(features)),importances[indices],align='center')
plt.xticks(range(len(features)),[features[i] for i in indices],rotation=90)
plt.tight_layout()
plt.show()

