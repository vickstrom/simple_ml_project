import os
from tensorflow import keras
from tensorflow.keras import layers

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

# Download dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()   

# Preprocess data
nr_train_samples, x_size, y_size = x_train.shape
nr_test_samples, _, _ = x_test.shape

x_train = x_train.reshape(nr_train_samples, x_size*y_size).astype("float32") / 255
x_test = x_test.reshape(nr_test_samples, x_size*y_size).astype("float32") / 255

# Create model
inputs = keras.Input(shape=(x_size*y_size))
layer1 = layers.Dense(128, activation=keras.activations.sigmoid)(inputs)
layer2 = layers.Dense(64)(layer1)
outputs = layers.Dense(10)(layer2)

model = keras.Model(inputs=inputs, outputs=outputs)

# Compile model
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.SGD(),
    metrics=["accuracy"]
)

# Train model
result = model.fit(x_train, y_train, batch_size=64, epochs=5, validation_split=0.2, verbose=0)
test_scores = model.evaluate(x_test, y_test, verbose=0)

loss, accuracy = test_scores
print("Test loss:", loss)
print("Test accuracy:", accuracy)