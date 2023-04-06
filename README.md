# **Classification model for heart attack prediction based on patient's medical history**
This project uses a neural network to predict whether a patient is more prone to a heart attack or not based on their medical history.
This is a simple implementation of neural network in order to predict the output value which is 0 or 1.

### **Requirements**
To run this project, you will need :

```
Python 3.6 or higher
NumPy
Pandas
TensorFlow
````
### **How to run this model**
 To install the required dependencies, you can run the following command:



### **Usage**
To run the project, you can use the following command:
```
git clone "https://github.com/Pratik-Behera/neural-network-tutorial.git"
```

```
python classification.py
```

##### This will run the neural network and generate a prediction for each patient in the dataset. The predictions will be output to a CSV file named predictions.csv.

### **Dataset**
This dataset is downloaded from kaggle website
Link: https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset?resource=download
### **Result**
```
Epoch 1/49
10/10 [==============================] - 0s 1ms/step - loss: 4.6991 - accuracy: 0.6469
Epoch 2/49
10/10 [==============================] - 0s 778us/step - loss: 3.4825 - accuracy: 0.6502
Epoch 3/49
10/10 [==============================] - 0s 779us/step - loss: 3.0834 - accuracy: 0.6403
Epoch 4/49
10/10 [==============================] - 0s 778us/step - loss: 2.6564 - accuracy: 0.6634
Epoch 5/49
10/10 [==============================] - 0s 778us/step - loss: 2.5674 - accuracy: 0.6337
Epoch 6/49
10/10 [==============================] - 0s 667us/step - loss: 2.0438 - accuracy: 0.6535
Epoch 7/49
10/10 [==============================] - 0s 779us/step - loss: 1.7565 - accuracy: 0.6700
Epoch 8/49
10/10 [==============================] - 0s 778us/step - loss: 1.8614 - accuracy: 0.6832
Epoch 9/49
10/10 [==============================] - 0s 778us/step - loss: 1.4555 - accuracy: 0.7195
Epoch 10/49
10/10 [==============================] - 0s 778us/step - loss: 1.2896 - accuracy: 0.6997
Epoch 11/49
10/10 [==============================] - 0s 778us/step - loss: 1.1677 - accuracy: 0.7162
Epoch 12/49
10/10 [==============================] - 0s 779us/step - loss: 1.0284 - accuracy: 0.7228
Epoch 13/49
10/10 [==============================] - 0s 778us/step - loss: 0.9577 - accuracy: 0.7261
Epoch 14/49
10/10 [==============================] - 0s 778us/step - loss: 0.9444 - accuracy: 0.7096
Epoch 15/49
10/10 [==============================] - 0s 778us/step - loss: 0.8296 - accuracy: 0.7459
Epoch 16/49
10/10 [==============================] - 0s 778us/step - loss: 0.7293 - accuracy: 0.7459
Epoch 17/49
10/10 [==============================] - 0s 778us/step - loss: 0.7111 - accuracy: 0.7492
Epoch 18/49
10/10 [==============================] - 0s 778us/step - loss: 0.5842 - accuracy: 0.7657
Epoch 19/49
10/10 [==============================] - 0s 778us/step - loss: 0.5026 - accuracy: 0.7855
Epoch 20/49
10/10 [==============================] - 0s 778us/step - loss: 0.4066 - accuracy: 0.8119
Epoch 21/49
10/10 [==============================] - 0s 779us/step - loss: 0.4801 - accuracy: 0.8053
Epoch 22/49
10/10 [==============================] - 0s 778us/step - loss: 0.4336 - accuracy: 0.8185
Epoch 23/49
10/10 [==============================] - 0s 778us/step - loss: 0.3483 - accuracy: 0.8482
Epoch 24/49
10/10 [==============================] - 0s 779us/step - loss: 0.2928 - accuracy: 0.8845
Epoch 25/49
10/10 [==============================] - 0s 667us/step - loss: 0.2699 - accuracy: 0.8779
Epoch 26/49
10/10 [==============================] - 0s 667us/step - loss: 0.2735 - accuracy: 0.8944
Epoch 27/49
10/10 [==============================] - 0s 778us/step - loss: 0.2859 - accuracy: 0.8977
Epoch 28/49
10/10 [==============================] - 0s 778us/step - loss: 0.2638 - accuracy: 0.8944
Epoch 29/49
10/10 [==============================] - 0s 778us/step - loss: 0.2260 - accuracy: 0.9010
Epoch 30/49
10/10 [==============================] - 0s 778us/step - loss: 0.2288 - accuracy: 0.9010
Epoch 31/49
10/10 [==============================] - 0s 667us/step - loss: 0.2287 - accuracy: 0.9076
Epoch 32/49
10/10 [==============================] - 0s 667us/step - loss: 0.2678 - accuracy: 0.8911
Epoch 33/49
10/10 [==============================] - 0s 667us/step - loss: 0.2687 - accuracy: 0.8944
Epoch 34/49
10/10 [==============================] - 0s 667us/step - loss: 0.2750 - accuracy: 0.8812
Epoch 35/49
10/10 [==============================] - 0s 667us/step - loss: 0.2674 - accuracy: 0.8812
Epoch 36/49
10/10 [==============================] - 0s 667us/step - loss: 0.2886 - accuracy: 0.8911
Epoch 37/49
10/10 [==============================] - 0s 667us/step - loss: 0.2264 - accuracy: 0.9175
Epoch 38/49
10/10 [==============================] - 0s 667us/step - loss: 0.1907 - accuracy: 0.9340
Epoch 39/49
10/10 [==============================] - 0s 779us/step - loss: 0.1785 - accuracy: 0.9307
Epoch 40/49
10/10 [==============================] - 0s 778us/step - loss: 0.1857 - accuracy: 0.9373
Epoch 41/49
10/10 [==============================] - 0s 778us/step - loss: 0.1810 - accuracy: 0.9373
Epoch 42/49
10/10 [==============================] - 0s 778us/step - loss: 0.1843 - accuracy: 0.9340
Epoch 43/49
10/10 [==============================] - 0s 778us/step - loss: 0.1675 - accuracy: 0.9472
Epoch 44/49
10/10 [==============================] - 0s 779us/step - loss: 0.1790 - accuracy: 0.9406
Epoch 45/49
10/10 [==============================] - 0s 778us/step - loss: 0.1715 - accuracy: 0.9439
Epoch 46/49
10/10 [==============================] - 0s 667us/step - loss: 0.1670 - accuracy: 0.9406
Epoch 47/49
10/10 [==============================] - 0s 667us/step - loss: 0.1725 - accuracy: 0.9406
Epoch 48/49
10/10 [==============================] - 0s 667us/step - loss: 0.1548 - accuracy: 0.9505
Epoch 49/49
10/10 [==============================] - 0s 778us/step - loss: 0.1717 - accuracy: 0.9505
```
