from tensorflow import keras
import matplotlib.pyplot as plt
import random



mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0



# Define the network architecture using Keras
model = keras.Sequential([
 keras.layers.Flatten(input_shape=(28, 28)),
 keras.layers.Dense(128, activation="relu"),
 keras.layers.Dense(10, activation="softmax")
])

model.summary()
model.compile(optimizer="sgd", loss="sparse_categorical_crossentropy", metrics=['accuracy'])
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10)

test_loss, test_acc = model.evaluate(x_test, y_test)
print("Loss = %.3f" % test_loss)
print("Accuracy = %.3f" % test_acc)

# Choose a random image from the test set
n = random.randint(0, len(x_test) - 1)
plt.imshow(x_test[n], cmap='gray')
plt.show()

# Predict the label for the chosen image
predicted_value = model.predict(x_test[n].reshape(1, 28, 28))
print("Predicted Value:", predicted_value.argmax())
# Plot training history - accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training history - loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
