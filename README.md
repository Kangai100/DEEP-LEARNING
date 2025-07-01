# DEEP-LEARNING
!pip install tensorflow
import numpy as np # used for Converting data to formats suitable for machine learning models
import tensorflow as tf #  is an open-source deep learning framework developed by Google. which is used Building, training, and evaluating deep learning models
from tensorflow import keras # is a high-level API built on top of TensorFlow
from matplotlib import pyplot as plt # Matplotlib is used for visualization 
import seaborn as sn # seaborn is also used for visualization
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data() # This loads the MNIST dataset and returns a tuple
x_train.shape # this shows the  NumPy array shape
x_test.shape
x_train[0]
x_train = x_train/255
x_test = x_test/255
index = 0
plt.imshow(x_train[index], cmap = plt.cm.binary)
print(y_train[index])
index = 2
plt.imshow(x_train[index], cmap = plt.cm.binary)
print(y_train[index])
x_train_flat = x_train.reshape(len(x_train),(28 * 28))
x_test_flat = x_test.reshape(len(x_test),(28 * 28))
x_train_flat.shape
model = keras.Sequential((
    keras.layers.Dense(128, input_shape = (784,) , activation = 'relu'),
    keras.layers.Dense(64, activation = 'sigmoid'),
    keras.layers.Dense(32, activation = 'sigmoid'),
    keras.layers.Dense(10, activation = 'softmax'),
))

model.compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
               )
model.fit(x_train_flat, y_train, epochs = 5)
model.evaluate(x_test_flat, y_test)
y_pred = model.predict(x_test_flat)
y_pred_labels = [np.argmax(i) for i in y_pred]
confusion_matrix = tf.math.confusion_matrix(labels = y_test, predictions = y_pred_labels)
sn.heatmap(confusion_matrix, annot = True , fmt = 'd')
