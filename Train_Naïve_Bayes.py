import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

# Load features and labels
features = np.load('features.npy')
labels = np.load('labels.npy')

# Split data into training and test sets, and the test set accounts for 30% of the total data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

# Train Naive Bayes classifier
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predict the labels on test set
y_pred = classifier.predict(X_test)

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Calculate accuracy, precision, and recall
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')

print(f"Accuracy: {"{:.2f}%".format(accuracy * 100)}")
print(f"Precision: {"{:.2f}%".format(precision * 100)}")
print(f"Recall: {"{:.2f}%".format(recall * 100)}")
