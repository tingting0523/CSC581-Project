import numpy
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import matplotlib.pyplot as plot
import re

print('Artificial Neural Network')

#activation function
def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

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
    for match in re.finditer(r"\('([^']+)', '([^']+)'\): \{([^}]+)\}", ratios):
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

numpy.random.seed(42)

# Load data from the file
file_path = "message.txt"
print(file_path,"file_path")
with open(file_path, "r") as file:
    data = file.read()

# Extract the dictionary from the data
data = data.split("Ratios: ", 1)[1].strip()
data = data.rstrip("}")

# Generate feature_id and ratios tuple dictionary
ratios_tuple_dict = feature_id_and_ratios(data)

# Prepare the data for training
X = []
y = []
for feature_id, (ratios, label) in ratios_tuple_dict.items():
    X.append(ratios)
    y.append(label)


# Convert labels to binary
def convert_labels(labels):
    return [1 if label == 'man' else 0 for label in labels]


y = convert_labels(y)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to numpy arrays
X_train = numpy.array(X_train)
X_test = numpy.array(X_test)

#these 2 lines of code are necessary to reshape the rtainging into a 2D array
y_train = numpy.array(y_train).reshape(-1, 1)
y_test = numpy.array(y_test).reshape(-1, 1)

# Initialize weights randomly
numpy.random.seed(42) #Set a random seed to ensure consistent results each time you run it
inumpyut_layer_neurons = X_train.shape[1] # The number of neurons in the input layer, i.e. the dimension of the feature
hidden_layer_neurons = 64
output_neurons = 1

weights_inumpyut_hidden = numpy.random.randn(inumpyut_layer_neurons, hidden_layer_neurons) * 0.01
bias_hidden = numpy.zeros((1, hidden_layer_neurons))
weights_hidden_output = numpy.random.randn(hidden_layer_neurons, output_neurons) * 0.01
bias_output = numpy.zeros((1, output_neurons))

# Set the learning rate and the amount of iterations max
learning_rate = 0.01
iterations = 5000

# Training process
for iteration in range(iterations):
    # Forward propagation
    hidden_layer_inumpyut = numpy.dot(X_train, weights_inumpyut_hidden) + bias_hidden
    hidden_layer_activation = sigmoid(hidden_layer_inumpyut)
    
    output_layer_inumpyut = numpy.dot(hidden_layer_activation, weights_hidden_output) + bias_output
    predicted_output = sigmoid(output_layer_inumpyut)
    
    # Calculate error
    error = y_train - predicted_output
    d_predicted_output = error * sigmoid_derivative(predicted_output)
    
    # Backpropagation
    error_hidden_layer = d_predicted_output.dot(weights_hidden_output.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_activation)
    
    # Update weights and biases
    weights_hidden_output += hidden_layer_activation.T.dot(d_predicted_output) * learning_rate
    bias_output += numpy.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate
    weights_inumpyut_hidden += X_train.T.dot(d_hidden_layer) * learning_rate
    bias_hidden += numpy.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

    # Print precision and recall every 1000 iterations
    if iteration % 1000 == 0 or iteration == iterations - 1:
        hidden_layer_inumpyut_test = numpy.dot(X_test, weights_inumpyut_hidden) + bias_hidden
        hidden_layer_activation_test = sigmoid(hidden_layer_inumpyut_test)

        output_layer_inumpyut_test = numpy.dot(hidden_layer_activation_test, weights_hidden_output) + bias_output
        predicted_output_test = sigmoid(output_layer_inumpyut_test)

        predictions = [1 if p > 0.5 else 0 for p in predicted_output_test]
        True_Positive = numpy.sum((predictions == 1) & (y_test.flatten() == 1))
        False_Positive = numpy.sum((predictions == 1) & (y_test.flatten() == 0))
        False_Negative = numpy.sum((predictions == 0) & (y_test.flatten() == 1))

        precision = True_Positive / (True_Positive + False_Positive) if (True_Positive + False_Positive) > 0 else 0
        recall = True_Positive / (True_Positive + False_Negative) if (True_Positive + False_Negative) > 0 else 0
        print(f"Iteration {iteration}")


# Prediction and Evaluation
hidden_layer_inumpyut_test = numpy.dot(X_test, weights_inumpyut_hidden) + bias_hidden
hidden_layer_activation_test = sigmoid(hidden_layer_inumpyut_test)

output_layer_inumpyut_test = numpy.dot(hidden_layer_activation_test, weights_hidden_output) + bias_output
predicted_output_test = sigmoid(output_layer_inumpyut_test)

predictions = [1 if p > 0.5 else 0 for p in predicted_output_test]
accuracy = numpy.mean(predictions == y_test.flatten()) * 100


# Print predictions for the whole dataset
print("Predictions for the test dataset:")
for i in range(len(X_test)):
    print(f"Predicted: {'man' if predictions[i] == 1 else 'woman'} - Actual: {'man' if y_test[i][0] == 1 else 'woman'}")

# Print final precision and recall scores
final_precision = precision_score(y_test, predictions)
final_recall = recall_score(y_test, predictions)
print(f"Final Precision: {final_precision:.4f}")
print(f"Final Recall: {final_recall:.4f}")
print(f"Test Accuracy: {accuracy:.2f}%")

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, predictions)
print("Confusion Matrix:")
print(conf_matrix)

# Plot the confusion matrix
plot.figure(figsize=(8, 6))
plot.imshow(conf_matrix, interpolation='nearest', cmap=plot.cm.Blues)
plot.title('Confusion Matrix')
plot.colorbar()
tick_marks = numpy.arange(2)
plot.xticks(tick_marks, ['Woman', 'Man'], rotation=45)
plot.yticks(tick_marks, ['Woman', 'Man'])

# Print confusion matrix values on the plot
thresh = conf_matrix.max() / 2.
for i, j in numpy.ndindex(conf_matrix.shape):
    plot.text(j, i, format(conf_matrix[i, j], 'd'))

plot.tight_layout()
plot.ylabel('True label')
plot.xlabel('Predicted label')
plot.show()