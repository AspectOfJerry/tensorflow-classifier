import tensorflow as tf
from tensorflow import keras
from keras import layers

print("Tensorflow version: ", tf.__version__)
print("Available devices ", tf.config.get_visible_devices())

# https://discuss.tensorflow.org/t/tensorflow-not-detecting-gpu/16295/8
physical_devices = tf.config.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)  # Set memory growth for GPU

# CIFAR-10 example dataset
(train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

# define model architecture
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(10)
])

# compile model
model.compile(optimizer="adam",
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])

epochs = 2

# train with GPU
with tf.device("/GPU:0"):
    model.fit(train_images, train_labels, epochs=epochs, validation_data=(test_images, test_labels))

# evaluate model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

print("\nTest accuracy:", test_acc)
print(model.summary())

input("Press Enter to exit...")
