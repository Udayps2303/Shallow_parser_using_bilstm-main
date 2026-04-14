import tensorflow as tf
from tensorflow.keras.models import load_model

# Replace 'your_model.keras' with the path to your saved model file
model = load_model('random_bilstm_weights-12-3.99.keras')

# Display the model's architecture and details
model.summary()
