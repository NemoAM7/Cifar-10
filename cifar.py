import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
import random
(X_train, Y_train), (X_test, Y_test) = datasets.cifar10.load_data()
Y_train = Y_train.reshape(-1,)
Y_test = Y_test.reshape(-1,)
X_train  = X_train / 255
X_test = X_test / 255
classes = ["airplane", "automobile","bird","cat","deer","dog","frog","horse", "ship","truck"]
def plot_sample(X,y,index):
    plt.figure(figsize = (15,2))
    plt.imshow(X[index])
    plt.xlabel(classes[y[index]])
    plt.show()
model = models.Sequential([
    #cnn
    layers.Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu', input_shape = (32,32,3)),
    layers.MaxPooling2D((2,2)),

    #dense
    layers.Flatten(),
    layers.Dense(64, activation = 'relu'),
    layers.Dense(10, activation = 'softmax')
])
model.compile(optimizer = 'SGD',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)
model.fit(X_train,Y_train,epochs = 10)
model.evaluate(X_test,Y_test )

