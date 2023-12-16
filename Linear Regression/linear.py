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
print(df.corr())

#Seaborn for visualization

import seaborn as sns
sns.pairplot(df)
plt.show()
