import tensorflow as tf
from tensorflow import keras
from keras import layers

# console colors
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"


# load the model
model = keras.models.load_model("./artifacts/model.h5")

# Load and preprocess CIFAR-10 example data
(_, _), (test_images, test_labels) = keras.datasets.cifar10.load_data()
test_images = test_images / 255.0

# Make predictions using the loaded model
print(GREEN + "\nMaking predictions..." + RESET)
predictions = model.predict(test_images)

# Process predictions (e.g., find the class with the highest probability)
predicted_labels = tf.argmax(predictions, axis=1)

# Compare predicted labels with actual labels
correct_predictions = tf.equal(predicted_labels, test_labels)
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

print(BLUE + "Model accuracy:", accuracy.numpy() + RESET)
