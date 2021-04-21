import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
%matplotlib inline
url = "http://bit.ly/w-data"
s_data = pd.read_csv(url)
print("Data imported successfully")
s_data.head(10)
s_data.plot(x='Hours', y='Scores', style='.', color ='green')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.grid()
plt.show()
X = s_data.iloc[:, :1].values  
y = s_data.iloc[:, 1:].values  
from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, y_train) 

print("Training complete.")
line = regressor.coef_*X+regressor.intercept_
# Plotting for the test data
plt.scatter(X, y,color ='g')
plt.plot(X, line);
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.grid()
plt.show()
print(X_test) # Testing data - In Hours
y_pred = regressor.predict(X_test) # Predicting the scores
df = pd.DataFrame({'Actual': y_test.ravel(), 'Predicted': y_pred.ravel()})  
df 
hours = 9.25
own_pred = regressor.predict([[hours]])
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))
from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_pred)) 