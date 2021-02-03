import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import preprocessing, linear_model
from sklearn.utils import shuffle
from matplotlib import style
import pandas as pd
import pickle
import time

#To track the execution duration
t1 = time.time()

# reading the file
data = pd.read_csv('FuelConsumptionCo2.csv')
print(data.head())

# data processing
data = data[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]
predict = 'CO2EMISSIONS'
x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

# Creating a train-test split
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
best = 0
for _ in range(30):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    accuracy = linear.score(x_test, y_test)
    print(accuracy)
    if accuracy > best:
        best = accuracy
        
# saving the model so we don't retrain the model each time we run the file
    with open('carmodel.pickle', 'wb') as f:
        pickle.dump(linear, f)

# Loading the saved model
pickle_in = open('carmodel.pickle', 'rb')
linear = pickle.load(pickle_in)

# printing the coefficient and the intercept of the model
print('coefficient:', linear.coef_)
print('intercept :', linear.intercept_)

# making prediction based on the model
predicted = linear.predict(x_test)
for i in range(len(predicted)):
    print('predictions:', predicted[i], 'data:', x_test[i], 'actual data', y_test[i])

# plotting a scatter diagram to visualize the effect of Enginesize on CO2 emission
style.use('ggplot')
p = 'ENGINESIZE'
plt.scatter(data[p],data['CO2EMISSIONS'])
plt.xlabel('Engine size')
plt.ylabel('Co2 emissions')
plt.show()

# to show the execution duration of the model
t2 = time.time()
print('time:',t2-t1)
