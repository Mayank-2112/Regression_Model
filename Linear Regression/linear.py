#importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#reading dataset
df=pd.read_csv('C:\Regression_Model\Linear Regression\HW.csv')
#print(df.head())

#scatter plot
plt.scatter(df['Height'],df['Weight'])
plt.xlabel('Height')
plt.ylabel('Weight')
#plt.show()

#Correlation
# print(df.corr())

#Seaborn for visualization

# import seaborn as sns
# sns.pairplot(df)
# plt.show()

# Independent and dependent variables

X=df[['Height']]
y=df['Weight']

# print(np.array(X).shape)
# print(np.array(y).shape)

# Train Test Split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.25,random_state=42)

#Standardization

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)


# Apply Simple Linear Regression
from sklearn.linear_model import LinearRegression

regression = LinearRegression()
regression.fit(X_train,y_train)
print(regression.coef_)
print(regression.intercept_)

#plot Training data - Best fit line
plt.scatter(X_train,y_train)
plt.plot(X_train,regression.predict(X_train))
# plt.show()

# Prediction for test data
y_pred=regression.predict(X_test)

#Performance Metrics
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

mse=mean_squared_error(y_test,y_pred)
mae=mean_absolute_error(y_test,y_pred)
rmse=np.sqrt(mse)
print(rmse)
print(mse)
print(mae)

score=r2_score(y_test,y_pred)
print(score)

#display adjusted r_squared
r_adj= 1-(1-score)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)
print(r_adj)

#OLS Linear Regression

import statsmodels.api as sm

model=sm.OLS(y_train,X_train).fit()
pred=model.predict(X_test)
# print(model.summary())

#prediction for new data
print(regression.predict(scaler.transform([[72]])))