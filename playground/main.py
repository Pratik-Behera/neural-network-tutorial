import numpy as np
weights = np.array([1,2])
input_data = np.array([3,4])
target_value = 6
learning_rate = 0.01
predicted_value =(weights*input_data).sum()
error = predicted_value - target_value

print(error)

#Slope calculation and update weights 
gradient = 2 * input_data * error
weights_updated = weights - learning_rate * gradient
predicted_value_updated = (weights_updated * input_data).sum()
error_updated = predicted_value_updated - target_value
print("Updated error value",error_updated)

