import pandas as pd
import matplotlib.pyplot as plt

from google.colab import files

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Upload dataset
uploaded = files.upload()

# Load dataset
data = pd.read_csv("IMDB Dataset.csv")

# Reviews and labels
X = data["review"]

y = data["sentiment"].map({
    "positive": 1,
    "negative": 0
})

# Convert text into numbers
vectorizer = CountVectorizer(max_features=5000)

X = vectorizer.fit_transform(X).toarray()

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Build Neural Network
model = Sequential([
    Dense(16, activation='relu', input_shape=(5000,)),
    Dense(1, activation='sigmoid')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train model
model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=128
)

# Evaluate model
loss, acc = model.evaluate(X_test, y_test)

print("Accuracy:", acc)

# Predictions
y_pred = model.predict(X_test)

y_pred = (y_pred > 0.5).astype("int32")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)

disp.plot(cmap='Blues')

plt.title("Confusion Matrix")

plt.show()