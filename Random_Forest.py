import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
import matplotlib.pyplot as plt
import numpy as np
import time

# Load data from the file
file_path = "message.txt"
with open(file_path, "r") as file:
    data = file.read()

# Extract the dictionary from the data
data = data.split("Ratios: ", 1)[1].strip()
data = data.rstrip("}")

# Regular expression to match the dictionary entries
entry_pattern = re.compile(r"\('([^']+)', '([^']+)'\): \{([^}]+)\}")

# Function to parse the features string
def parse_features(features_str):
    feature_dict = {}
    for feature in features_str.split(", "):
        key, value = feature.split(": ")
        feature_dict[key.strip("'")] = float(value)
    return feature_dict

# Method to create a new dictionary with feature_id and ratios tuple
def feature_id_and_ratios(ratios):
    new_data = {}
    for match in entry_pattern.finditer(ratios):
        person_id, feature_id, features_str = match.groups()
        features = parse_features(features_str)
        ratio_tuple = (
            features.get("eye_distance_ratio"),
            features.get("Eye length ratio"),
            features.get("nose_ratio"),
            features.get("lip_size_ratio"),
            features.get("lip_length_ratio"),
            features.get("eyebrow_length_ratio"),
            features.get("aggressive_ratio"),
            features.get("eyebrow_length_ratio"),
            features.get("aggressive_ratio")
        )
        new_data[feature_id] = (ratio_tuple, 'man' if person_id.startswith('m-') else 'woman')
    return new_data

# Generate feature_id and ratios tuple dictionary
ratios_tuple_dict = feature_id_and_ratios(data)

# Prepare the data for training
X = []
y = []
for feature_id, (ratios, label) in ratios_tuple_dict.items():
    X.append(ratios)
    y.append(label)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Measure the training time
start_time = time.time()
# Train the Random Forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
training_time = time.time() - start_time

# Measure the prediction time
start_time = time.time()
# Predict on the test set
y_pred = rf.predict(X_test)
prediction_time = time.time() - start_time

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Print predictions for example ratios
example_ratios = X_test[:80]
example_predictions = rf.predict(example_ratios)
print(f"Example Ratios Predictions: {example_predictions}")

# Convert labels to binary format for precision and recall calculations
def convert_labels(labels):
    return [1 if label == 'man' else 0 for label in labels]

final_precision = precision_score(convert_labels(y_test), convert_labels(y_pred))
final_recall = recall_score(convert_labels(y_test), convert_labels(y_pred))
print(f"Final Precision: {final_precision:.4f}")
print(f"Final Recall: {final_recall:.4f}")

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Print the timing results
print(f"Training Time: {training_time:.4f} seconds")
print(f"Prediction Time: {prediction_time:.4f} seconds")

# Plot accuracy, precision, and recall
metrics = ['Accuracy', 'Precision', 'Recall']
values = [accuracy, final_precision, final_recall]

plt.figure(figsize=(10, 6))
plt.bar(metrics, values, color=['blue', 'green', 'red'])
plt.ylim(0, 1)
plt.title('Model Performance Metrics')
plt.xlabel('Metrics')
plt.ylabel('Values')
plt.show()

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Woman', 'Man'], rotation=45)
plt.yticks(tick_marks, ['Woman', 'Man'])

# Print confusion matrix values on the plot
thresh = conf_matrix.max() / 2.
for i, j in np.ndindex(conf_matrix.shape):
    plt.text(j, i, format(conf_matrix[i, j], 'd'),
             horizontalalignment="center",
             color="white" if conf_matrix[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()