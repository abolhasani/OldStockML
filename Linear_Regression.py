import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt   #Data visualisation libraries
import seaborn as sns

IAF = pd.read_csv('IAF2.csv')
#sns.pairplot(IAF)
#sns.distplot(IAF['Close'])
X = IAF[['OpenInt', 'High', 'Low', 'Open', 'Volume']]
y = IAF['Close']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7) #, random_state=101)
lm = LinearRegression()
lm.fit(X_train,y_train)
predictions = lm.predict(X_test)
#sc = lm.score(predictions, y_test)
#plt.scatter(y_test,predictions)
#plt.show()
plt.savefig('LR.png', dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        metadata=None)
#print(sc)
#print(y_test);
#print(X_test);
plt.figure()
plt.plot(predictions)
#plt.plot(y)
plt.title('Prediction vs Real Stock Price')
plt.ylabel('Price')
plt.xlabel('Days')
plt.legend(['Prediction', 'Real'], loc='upper left')
plt.show()
#print(predictions)
#print(IAF['Close'])