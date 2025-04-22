import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


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

##Split in Train/Test

X=df_ml.drop(columns=['price'])  #Explanatory Varaibles
y=df_ml['price']

print(X.columns)

X_train, X_test, y_train, y_test=train_test_split(X,y, shuffle=False, test_size=0.2)

#Train Linear, Ridge, and Lasso

#Linear
lr=LinearRegression()
lr.fit(X_train,y_train)
y_pred_lr=lr.predict(X_test)

#Ridge
ridge=Ridge(alpha=1.0)
ridge.fit(X_train,y_train)
y_pred_ridge=ridge.predict(X_test)

#Lasso
lasso=Lasso(alpha=0.1)
lasso.fit(X_train,y_train)
y_pred_lasso=lasso.predict(X_test)

##Show reszults

def plot_preds(y_test, preds, label):
    plt.plot(y_test.index, y_test, label='Actual')
    plt.plot(y_test.index, preds, label=label)
    plt.legend()
    plt.title(f'{label} Forecast vs Actual')
    plt.show()

print(f"Linear MS)={mean_squared_error(y_test, y_pred_lr):.2f}")
print(f"Ridge MS)={mean_squared_error(y_test, y_pred_ridge):.2f}")
print(f"Lasso MS)={mean_squared_error(y_test, y_pred_lasso):.2f}")

plot_preds(y_test, y_pred_lr,"Linear Regression")
plot_preds(y_test, y_pred_ridge,"Ridge Regression")
plot_preds(y_test, y_pred_lasso,"Lasso Regression")