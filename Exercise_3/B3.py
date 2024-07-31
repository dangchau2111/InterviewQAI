import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics import accuracy_score

# Load MNIST dataset
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist.data, mnist.target.astype(int).values  # Convert to numpy array

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
        
    def forward(self, X):
        # Forward pass
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = np.tanh(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.z2
        return self.a2

    def backward(self, X, anchor_embed, positive_embed, negative_embed, learning_rate=0.01):
        # Backward pass
        m = X.shape[0]
        dz2 = anchor_embed - positive_embed + negative_embed
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * (1 - np.tanh(self.z1)**2)
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        # Update weights and biases
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

   
    def triplet_loss(anchor, positive, negative, margin=1.0):
        # Compute Euclidean distance between anchor and positive
        pos_dist = np.sum(np.square(anchor - positive), axis=1)
        # Compute Euclidean distance between anchor and negative
        neg_dist = np.sum(np.square(anchor - negative), axis=1)
        # Compute Triplet Loss value
        loss = np.maximum(pos_dist - neg_dist + margin, 0)
        return np.mean(loss)

   
    def create_triplets(x, y, num_triplets=1000):
        triplets = []
        for _ in range(num_triplets):
            anchor_idx = np.random.randint(0, x.shape[0])
            anchor_label = y[anchor_idx]
            positive_idx = np.random.choice(np.where(y == anchor_label)[0])
            negative_idx = np.random.choice(np.where(y != anchor_label)[0])
            triplets.append((x[anchor_idx], x[positive_idx], x[negative_idx]))
        return np.array(triplets)

# Initialize model
model = NeuralNetwork(input_size=28*28, hidden_size=128, output_size=128)

# Create triplet data
triplets = NeuralNetwork.create_triplets(X_train, y_train)

# Hyperparameters
initial_learning_rate = 0.01
decay_rate = 0.001
epochs = 10

# Train the model with learning rate decay
for epoch in range(epochs):
    # Update learning rate with decay
    learning_rate = initial_learning_rate / (1 + decay_rate * epoch)
    
    total_loss = 0
    for anchor, positive, negative in triplets:
        anchor_embed = model.forward(anchor.reshape(1, -1))
        positive_embed = model.forward(positive.reshape(1, -1))
        negative_embed = model.forward(negative.reshape(1, -1))
        
        # Calculate loss
        batch_loss = NeuralNetwork.triplet_loss(anchor_embed, positive_embed, negative_embed)
        total_loss += batch_loss
        
        # Perform backward pass and update weights
        model.backward(anchor.reshape(1, -1), anchor_embed, positive_embed, negative_embed, learning_rate)
    
    print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(triplets)}, Learning Rate: {learning_rate}')

# Generate embeddings for training and testing data
x_train_embed = np.array([model.forward(x.reshape(1, -1)).flatten() for x in X_train])
x_test_embed = np.array([model.forward(x.reshape(1, -1)).flatten() for x in X_test])

# Predict labels for test data
def predict_labels(test_embeddings, train_embeddings, train_labels):
    distances = cosine_distances(test_embeddings, train_embeddings)
    closest = np.argmin(distances, axis=1)
    return train_labels[closest]

predicted_labels = predict_labels(x_test_embed, x_train_embed, y_train)

# Calculate accuracy
accuracy = accuracy_score(y_test, predicted_labels)
print(f'Accuracy: {accuracy * 100:.2f}%')
