# from keras.utils.np_utils import to_categorical
# from keras import Sequential
# from keras.layers import Activation,Dense
# import pandas as pd

# data = pd.read_csv("heart.csv")
# predictor = data.drop(['output'],axis=1).values
# target = to_categorical(data.output)

# model = Sequential()
# model.add(Dense(100,activation='relu',input_shape = (n_cols,)))
# model.add(Dense(100,activation='relu'))
# model.add(Dense(100,activation='relu'))
# model.add(Dense(2,activation='softmax'))
# model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
# model.fit(predictor,target)

import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
import pandas as pd

df=pd.read_csv("heart.csv")
target = df["output"].values
target=to_categorical(target)

#Drop those features which are not required while model training
df.drop(["output"],axis=1).values
features = df.values

model=Sequential()
model.add(Dense(32,activation="relu",input_shape=(14,)))#Hidden Layer
model.add(Dense(2,activation="softmax"))
model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])
model.fit(features,target,epochs=49)



