# import numpy as np
# from keras.layers import Dense
# from keras.models import Sequential
 
# predictors = np.loadtxt('', delimiter=',') #add path to csv file
# n_cols = predictors.shape[1]

# model = Sequential()
# model.add(Dense(100,activation='relu',input_shape=(n_cols,))) # no of columns, number of columns
# model.add(Dense(100,activation= 'relu'))
# model.add(Dense(1))  

# #Compiling and Fitting

# model.compile(loss = 'mean_squared_error',optimizer = "adam")
# model.fit(features,targets,epochs=10) #features and targets are specified according to the csv file


import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1/(1+np.exp(-z))

input = np.linspace(-20,20,100)
plt.plot(input,sigmoid(input))
plt.show()

