import numpy as np
from keras.datasets import imdb
from keras import models
from keras import layers
import matplotlib.pyplot as plt


def vectorize_sequences(sequences, dimension=10000):
    # Create an all-zero matrix
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results


(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# Our vectorized training data
X_train = vectorize_sequences(train_data)

# Vectorize test data
X_test = vectorize_sequences(test_data)

# Vectorized labels
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=5, batch_size=512, validation_data=(X_test, y_test))

acc = history.history['acc']
val_acc = history.history['val_acc']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'b', label='Training Accuracy')
plt.plot(epochs, acc, 'bo')

plt.plot(epochs, val_acc, 'g', label='Validation Accuracy')
plt.plot(epochs, val_acc, 'go')

plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.legend()
plt.show()