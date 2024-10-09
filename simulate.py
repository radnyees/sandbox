import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0,8,8*500)

brain_signal = np.zeros_like(t)

brain_signal[1000:1050] = 1  # Spike at 2 seconds
brain_signal[2000:2050] = 1  # Spike at 4 seconds
brain_signal[3000:3050] = 1  # Spike at 6 seconds
brain_signal[4000:4050] = 1  # Spike at 8 seconds

plt.plot(t, brain_signal)
plt.title("Simulated Brain Signal: Movement Intention (8 seconds)")
plt.xlabel("Time (s)")
plt.ylabel("Movement Intention (0 or 1)")

plt.savefig("simulated_8secbrainsignal.jpg")
plt.show()

# Process brain signal to detect movement intention
for i, signal in enumerate(brain_signal):
    if signal == 1:
        print(f"Movement intention detected at time {i/500} seconds")
        # Here you would send a command to the electrical stimulation device

# Simulate muscle stimulation
for i, signal in enumerate(brain_signal):
    if signal == 1:
        print(f"Electrical stimulation triggered at time {i/500} seconds to activate muscle.")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Step 1: Simulate EEG Data
# We will create 1000 samples, each with 10 features (like different frequency power bands)
X = np.random.rand(1000, 10)  # Simulated EEG data: 1000 samples, each with 10 features

# Step 2: Simulate Labels for Classification
# Binary labels: 0 = no movement intention, 1 = movement intention
y = np.random.randint(0, 2, 1000)

# Step 3: Split Data into Training and Test Sets
# Train set: 80%, Test set: 20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train a Random Forest Classifier
clf = RandomForestClassifier()  # Using Random Forest for classification
clf.fit(X_train, y_train)  # Train the classifier

# Step 5: Make Predictions
y_pred = clf.predict(X_test)

# Step 6: Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)  # Calculate accuracy of the model
print(f"Model Accuracy: {accuracy * 100:.2f}%")  # Print the accuracy in percentage

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

# Step 1: Simulate EEG Data
# We will create 1000 samples, each with 10 features (like different frequency power bands)
X = np.random.rand(1000, 10)  # Simulated EEG data: 1000 samples, each with 10 features

# Step 2: Simulate Labels for Classification
# Binary labels: 0 = no movement intention, 1 = movement intention
y = np.random.randint(0, 2, 1000)

# Step 3: Split Data into Training and Test Sets
# Train set: 80%, Test set: 20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train a Random Forest Classifier
clf = LogisticRegression()  # Using Random Forest for classification
clf.fit(X_train, y_train)  # Train the classifier

# Step 5: Make Predictions
y_pred = clf.predict(X_test)

# Step 6: Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)  # Calculate accuracy of the model
print(f"Model Accuracy: {accuracy * 100:.2f}%")  # Print the accuracy in percentage



from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Step 1: Simulate EEG Data
# We will create 1000 samples, each with 10 features (like different frequency power bands)
X = np.random.rand(1000, 10)  # Simulated EEG data: 1000 samples, each with 10 features

# Step 2: Simulate Labels for Classification
# Binary labels: 0 = no movement intention, 1 = movement intention
y = np.random.randint(0, 2, 1000)

# Step 3: Split Data into Training and Test Sets
# Train set: 80%, Test set: 20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train a Random Forest Classifier
clf = DecisionTreeClassifier()  # Using Random Forest for classification
clf.fit(X_train, y_train)  # Train the classifier

# Step 5: Make Predictions
y_pred = clf.predict(X_test)

# Step 6: Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)  # Calculate accuracy of the model
print(f"Model Accuracy: {accuracy * 100:.2f}%")  # Print the accuracy in percentage
