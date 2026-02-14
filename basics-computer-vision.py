import tensorflow as tf
import numpy as np

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy') > 0.95):
            print("\nModel trained successfully!")
            self.model.stop_training = True

callbacks = myCallback()
data = tf.keras.datasets.fashion_mnist

(training_images, training_labels), (test_images, test_labels) = data.load_data()

training_images = training_images / 255.0
test_images = test_images / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=50, callbacks=[callbacks])

classifications = model.predict(test_images)

for i in range(100):
    predicted_label = np.argmax(classifications[i])
    actual_label = test_labels[i]

    if predicted_label != actual_label:
        print(f"Error at index {i}: Model thought it was {predicted_label}, but it is {actual_label}")
