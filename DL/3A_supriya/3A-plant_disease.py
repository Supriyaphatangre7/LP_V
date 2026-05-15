
# Train

# Upload ZIP File
from google.colab import files
uploaded = files.upload()

# Extract ZIP File
import zipfile

with zipfile.ZipFile("plant.zip", 'r') as zip_ref:
    zip_ref.extractall()

# Check Dataset Folders
import os
print(os.listdir("dataset"))

# Import TensorFlow Libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense

# Load Dataset
train_data = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset",
    image_size=(128,128),
    batch_size=32,
    label_mode='binary'
)

# Normalize Images
train_data = train_data.map(lambda x, y: (x / 255.0, y))

# Build CNN Model
model = Sequential([

    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),

    Dense(128, activation='relu'),

    Dense(1, activation='sigmoid')
])

# Compile Model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train Model
model.fit(train_data, epochs=10)

# Save Model
model.save("plant_disease_model.h5")

print("Model Trained Successfully")


#Test 


# Upload Test Image
from google.colab import files
uploaded = files.upload()

# Import Libraries
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load Trained Model
model = tf.keras.models.load_model("plant_disease_model.h5")

# Load Test Image
# Replace test_leaf.jpg with your image name
img = image.load_img(
    "test_leaf.JPG",
    target_size=(128,128)
)

# Convert Image to Array
img_array = image.img_to_array(img)

# Add Batch Dimension
img_array = np.expand_dims(img_array, axis=0)

# Normalize Image
img_array = img_array / 255.0

# Predict
prediction = model.predict(img_array)

# Print Prediction Value
print("Prediction Value:", prediction[0][0])

# Final Result
if prediction[0][0] > 0.5:
    print("Healthy Plant")
else:
    print("Diseased Plant")
