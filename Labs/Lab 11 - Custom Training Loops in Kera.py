# Lab 11 - Custom Training Loops in Keras

# Import necessary libraries
import tensorflow as tf 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Flatten 

# Step 1: Set Up the Environment
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data() 
x_train, x_test = x_train / 255.0, x_test / 255.0 
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32) 

# Step 2: Define the Model
model = Sequential([ 
    Flatten(input_shape=(28, 28)), 
    Dense(128, activation='relu'), 
    Dense(10) 
]) 

# Step 3: Define Loss Function and Optimizer
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) 
optimizer = tf.keras.optimizers.Adam() 

# Step 4: Implement the Custom Training Loop
for epoch in range(5): 
    for x_batch, y_batch in train_dataset: 
        with tf.GradientTape() as tape: 
            logits = model(x_batch, training=True) 
            loss = loss_fn(y_batch, logits) 
        grads = tape.gradient(loss, model.trainable_weights) 
        optimizer.apply_gradients(zip(grads, model.trainable_weights)) 
    print(f'Epoch {epoch + 1}: Loss = {loss.numpy()}')


# Import necessary libraries
import tensorflow as tf 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Flatten 

# Step 1: Set Up the Environment
(x_train, y_train), _ = tf.keras.datasets.mnist.load_data() 
x_train = x_train / 255.0 
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32) 

# Step 2: Define the Model
model = Sequential([ 
    Flatten(input_shape=(28, 28)), 
    Dense(128, activation='relu'), 
    Dense(10) 
]) 

# Step 3: Define Loss Function, Optimizer, and Metric
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) 
optimizer = tf.keras.optimizers.Adam() 
accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy() 

# Step 4: Implement the Custom Training Loop with Accuracy Tracking
epochs = 5 
for epoch in range(epochs): 
    for x_batch, y_batch in train_dataset: 
        with tf.GradientTape() as tape: 
            logits = model(x_batch, training=True) 
            loss = loss_fn(y_batch, logits) 
        grads = tape.gradient(loss, model.trainable_weights) 
        optimizer.apply_gradients(zip(grads, model.trainable_weights)) 
        accuracy_metric.update_state(y_batch, logits) 
    print(f'Epoch {epoch + 1}: Loss = {loss.numpy()} Accuracy = {accuracy_metric.result().numpy()}') 
    accuracy_metric.reset_state()


# Import necessary libraries
import tensorflow as tf 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Flatten 
from tensorflow.keras.callbacks import Callback 

# Step 1: Set Up the Environment
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data() 
x_train = x_train / 255.0 
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32) 

# Step 2: Define the Model
model = Sequential([ 
    tf.keras.Input(shape=(28, 28)),  # Updated Input layer syntax
    Flatten(), 
    Dense(128, activation='relu'), 
    Dense(10) 
]) 

# Step 3: Define Loss Function, Optimizer, and Metric
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) 
optimizer = tf.keras.optimizers.Adam() 
accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy() 

# Step 4: Implement the Custom Callback
class CustomCallback(Callback): 
    def on_epoch_end(self, epoch, logs=None): 
        print(f'End of epoch {epoch + 1}, loss: {logs.get("loss")}, accuracy: {logs.get("accuracy")}') 

# Step 5: Implement the Custom Training Loop with Custom Callback
custom_callback = CustomCallback() 

for epoch in range(5): 
    for x_batch, y_batch in train_dataset: 
        with tf.GradientTape() as tape: 
            logits = model(x_batch, training=True) 
            loss = loss_fn(y_batch, logits) 
        grads = tape.gradient(loss, model.trainable_weights) 
        optimizer.apply_gradients(zip(grads, model.trainable_weights)) 
        accuracy_metric.update_state(y_batch, logits) 
    custom_callback.on_epoch_end(epoch, logs={'loss': loss.numpy(), 'accuracy': accuracy_metric.result().numpy()}) 
    accuracy_metric.reset_state()  # Updated method


import json
import os
import keras_tuner as kt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# Step 1: Load your dataset
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

# Step 2: Define the model-building function
def build_model(hp):
    model = Sequential()
    # Tune the number of units in the first Dense layer
    model.add(Dense(units=hp.Int('units', min_value=32, max_value=512, step=32),
                    activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Binary classification example
    model.compile(optimizer=Adam(hp.Float('learning_rate', 1e-4, 1e-2, sampling='LOG')),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# Step 3: Initialize a Keras Tuner RandomSearch tuner
tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=10,  # Set the number of trials
    executions_per_trial=1,  # Set how many executions per trial
    directory='tuner_results',  # Directory for saving logs
    project_name='hyperparam_tuning'
)

# Step 4: Run the tuner search (make sure the data is correct)
tuner.search(X_train, y_train, validation_data=(X_val, y_val), epochs=5)

# Step 5: Save the tuning results as JSON files
try:
    for i in range(10):
        # Fetch the best hyperparameters from the tuner
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        
        # Results dictionary to save hyperparameters and score
        results = {
            "trial": i + 1,
            "hyperparameters": best_hps.values,  # Hyperparameters tuned in this trial
            "score": None  # Add any score or metrics if available
        }

        # Save the results as JSON
        with open(os.path.join('tuning_results', f"trial_{i + 1}.json"), "w") as f:
            json.dump(results, f)

except IndexError:
    print("Tuning process has not completed or no results available.")


























