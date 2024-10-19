import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
data=pd.read_csv("exercise1.csv")

print(data.head())
print(data.head())

print(data.describe())

print(data.isnull())
print(data.isnull().sum())

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data['Gender']=le.fit_transform(data['Gender'].values)


x=data[['Duration','Heart_Rate','Body_Temp']]
y=data['Calories']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

plt.figure(dpi=125)
sns.regplot(x=y_test, y=y_pred, scatter_kws={"color": "blue"}, line_kws={"color": "red"})
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('Regression Line for y_test vs y_pred')
plt.show()


plt.figure(dpi=125)
sns.heatmap(np.round(data.corr(numeric_only=True),2),annot=True)
plt.show()

plt.scatter(data['Height'],data['Calories'])
plt.xlabel('Height')  # Label for x-axis
plt.ylabel('Calories')
plt.show()

plt.scatter(data['Gender'],data['Calories'])
plt.xlabel('Gender')  # Label for x-axis
plt.ylabel('Calories')
plt.show()

plt.scatter(data['Heart_Rate'],data['Calories'])
plt.xlabel('Heart_Rate')  # Label for x-axis
plt.ylabel('Calories')
plt.show()

plt.scatter(data['Duration'],data['Calories'])
plt.xlabel('Duration')  # Label for x-axis
plt.ylabel('Calories')
plt.show()
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"mean squared error",mse)
print(f"r^2 error",r2)





