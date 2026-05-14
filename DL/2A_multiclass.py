import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# Upload dataset
uploaded = files.upload()

# Read dataset
data = pd.read_csv("letter-recognition.data", header=None)

# Features and target
X = data.iloc[:, 1:]
y = data.iloc[:, 0]

# Encode target labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Convert to categorical
y_cat = to_categorical(y_encoded)

# Feature scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.2, random_state=42
)

# Build Deep Neural Network
model = Sequential([
    Dense(64, activation='relu', input_shape=(16,)),
    Dense(26, activation='softmax')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
model.fit(X_train, y_train, epochs=20)

# Evaluate model
loss, acc = model.evaluate(X_test, y_test)

print("Accuracy:", acc)

# Predictions
y_pred = model.predict(X_test)

# Convert predictions
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

# Confusion Matrix
cm = confusion_matrix(y_test_classes, y_pred_classes)

# Display confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')

plt.title("Confusion Matrix")
plt.show()