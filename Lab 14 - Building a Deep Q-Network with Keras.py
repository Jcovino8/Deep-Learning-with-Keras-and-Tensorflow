# Lab 14 - Building a Deep Q-Network with Keras

import os

# Create sample directory structure if it does not exist
base_dir = 'sample_data'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'validation')
class1_train = os.path.join(train_dir, 'class1')
class2_train = os.path.join(train_dir, 'class2')
class1_val = os.path.join(val_dir, 'class1')
class2_val = os.path.join(val_dir, 'class2')

# Create directories if they do not exist
for dir_path in [train_dir, val_dir, class1_train, class2_train, class1_val, class2_val]:
    os.makedirs(dir_path, exist_ok=True)

print("Directory structure created. Add your images to these directories.")

# Import the necessary library
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Modify data generator to include validation data
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    'sample_data',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    'sample_data',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# Function to modify the reward to encourage longer episodes
def modify_reward(reward, next_state):
    # Penalize large pole angles
    pole_angle = abs(next_state[2])  # Extract the pole angle from the state
    penalty = 1 if pole_angle > 0.1 else 0  # Apply penalty if angle is large
    return reward - penalty  # Adjust reward

# Inside the training loop
# Example usage in a reinforcement learning training loop:
# reward = modify_reward(reward, next_state)  # Use the modified reward


# Import necessary libraries
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Modify data generator to include validation data
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    'sample_data',  
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    'sample_data',  
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# Early stopping parameters
consecutive_success_threshold = 100
success_episode_length = 195
consecutive_success_count = 0
episode_lengths = []  # Initialize episode lengths list

# Example of training loop (this should be your actual loop)
for episode in range(1000):  # Replace with actual loop condition
    # Training logic goes here
    episode_length = 200  # Example value, replace with actual calculation
    episode_lengths.append(episode_length)
    
    # Early stopping check
    if len(episode_lengths) > consecutive_success_threshold and all(
        length >= success_episode_length for length in episode_lengths[-consecutive_success_threshold:]
    ):
        print("Early stopping: Agent consistently reaches max episode length.")
        break  # This break is now correctly inside the loop

# Modify data generator to include validation data
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    'sample_data',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    'sample_data',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

def decay_epsilon(epsilon, episode, switch_episode=100):
    if episode < switch_episode:
        return max(epsilon - 0.01, 0.01)  # Linear decay
    else:
        return max(epsilon * 0.99, 0.01)  # Exponential decay

# Inside the training loop
epsilon = decay_epsilon(epsilon, e)  # Adjust epsilon based on the current episode















































