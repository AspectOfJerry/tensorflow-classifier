import tensorflow as tf
from tensorflow import keras
from keras import layers

# console colors
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"

print("Tensorflow version: ", tf.__version__)
print("Available devices ", tf.config.get_visible_devices())

use_gpu = True
gpu_id = 0

physical_devices = tf.config.list_physical_devices("GPU")
if (len(physical_devices) > gpu_id):
    tf.config.experimental.set_memory_growth(physical_devices[gpu_id], True)  # set memory growth for GPU

# CIFAR-10 example dataset
(train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

# define model architecture
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation = "relu", input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation = "relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation = "relu"),
    layers.Flatten(),
    layers.Dense(64, activation = "relu"),
    layers.Dense(10)
])

# compile model
model.compile(optimizer = "adam",
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
              metrics = ["accuracy"])

epochs = 1

checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath = "./artifacts/checkpoints/model_checkpoint.h5",
    save_best_only = True,  # only save best model based on validation performance
    monitor = "val_accuracy",  # metric to monitor for saving model
    mode = "max",  # monitoring mode max for val_accuracy
    verbose = 1
)

# train with GPU with checkpoints
print(GREEN + "\nTraining the model with checkpoints every 5 epochs..." + RESET)

with tf.device("/GPU:0"):
    model.fit(train_images,
                train_labels, 
                epochs=epochs, 
                validation_data=(test_images, test_labels),
                callbacks=[checkpoint_callback])

# evaluate model
print(GREEN + "\nEvaluating the model..." + RESET)
test_loss, test_acc = model.evaluate(test_images, test_labels)

print(BLUE + "\nTest accuracy: " + str(test_acc) + RESET)

model.save("./artifacts/model.h5")
print(YELLOW + "\nSaved model to ./artifacts/model.h5" + RESET)

input(GREEN + "Press Enter to exit..." + RESET)
