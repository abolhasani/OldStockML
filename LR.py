import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

IAF = pd.read_csv('IAF2.csv')
#print(IAF)
X = IAF[['OpenInt', 'High', 'Low', 'Open', 'Volume']]
y = IAF['Close']
#X = X.drop(['Low', 'OpenInt'])

plt.figure()
plt.plot(y)
plt.title('Iran Aluminium Stock Price')
plt.ylabel('Price')
plt.xlabel('Dates')
#plt.legend(['Prediction', 'Real'], loc='upper left')
plt.show()

#_train = IAF[:10000]
#d_test = IAF[10000:]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=False)
#print(y_test)
#print(X_test)

'''scaler = Normalizer().fit(X_train)
normalized_X = scaler.transform(X_train)
normalized_X_test = scaler.transform(X_test)
#print(normalized_X_test)'''


lr = LinearRegression(normalize=False)

lr.fit(X, y)

y_pred = lr.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(mse, r2)

print(y_pred,y_test)
plt.figure()
plt.plot(y_pred)
plt.plot(y_test)
plt.title('Prediction vs Real Stock Price')
plt.ylabel('Price')
plt.xlabel('Dates')
plt.legend(['Prediction', 'Real'], loc='upper left')
plt.show()