import re
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import matplotlib.pyplot as plot
import numpy as np

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

# Train the k-NN classifier
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train, y_train)

# Predict on the test set
y_pred = knn.predict(X_test)

# Evaluate the model

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print(f"Prediction for example ratios: {y_pred}")




#Because the sklearn built-in precision_score needs to be 0 or 1,
#we have to normalize the data
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

# Plot the confusion matrix
plot.figure(figsize=(8, 6))
plot.imshow(conf_matrix, interpolation='nearest', cmap=plot.cm.Blues)
plot.title('Confusion Matrix')
plot.colorbar()
tick_marks = np.arange(2)
plot.xticks(tick_marks, ['Woman', 'Man'], rotation=45)
plot.yticks(tick_marks, ['Woman', 'Man'])

# Print confusion matrix values on the plot
thresh = conf_matrix.max() / 2.
for i, j in np.ndindex(conf_matrix.shape):
    plot.text(j, i, format(conf_matrix[i, j], 'd'),
             horizontalalignment="center",
             color="white" if conf_matrix[i, j] > thresh else "black")

plot.tight_layout()
plot.ylabel('True label')
plot.xlabel('Predicted label')
plot.show()
