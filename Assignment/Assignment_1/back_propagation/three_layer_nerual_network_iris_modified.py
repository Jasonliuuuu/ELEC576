import numpy as np
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class NeuralNetwork(object):
    def __init__(self, nn_input_dim, nn_hidden_dim, nn_output_dim, actFun_type='leaky_relu', reg_lambda=0.01, seed=0):
        self.nn_input_dim = nn_input_dim
        self.nn_hidden_dim = nn_hidden_dim
        self.nn_output_dim = nn_output_dim
        self.actFun_type = actFun_type
        self.reg_lambda = reg_lambda
        
        np.random.seed(seed)
        self.W1 = np.random.randn(self.nn_input_dim, self.nn_hidden_dim) / np.sqrt(self.nn_input_dim)
        self.b1 = np.zeros((1, self.nn_hidden_dim))
        self.W2 = np.random.randn(self.nn_hidden_dim, self.nn_output_dim) / np.sqrt(self.nn_hidden_dim)
        self.b2 = np.zeros((1, self.nn_output_dim))

    def actFun(self, z, type):
        if type == 'leaky_relu':
            return np.where(z > 0, z, 0.01 * z)
        elif type == 'tanh':
            return np.tanh(z)
        elif type == 'sigmoid':
            return 1 / (1 + np.exp(-z))
        else:
            raise ValueError("Unknown activation function type")

    def diff_actFun(self, z, type):
        if type == 'leaky_relu':
            return np.where(z > 0, 1, 0.01)
        elif type == 'tanh':
            return 1 - np.tanh(z)**2
        elif type == 'sigmoid':
            sig = 1 / (1 + np.exp(-z))
            return sig * (1 - sig)
        else:
            raise ValueError("Unknown activation function type")

    def feedforward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.actFun(self.z1, type=self.actFun_type)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        exp_scores = np.exp(self.z2)
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def calculate_loss(self, X, y):
        num_examples = len(X)
        self.feedforward(X)
        correct_logprobs = -np.log(self.probs[range(num_examples), np.argmax(y, axis=1)])
        data_loss = np.sum(correct_logprobs)
        data_loss += self.reg_lambda / 2 * (np.sum(np.square(self.W1)) + np.sum(np.square(self.W2)))
        return (1. / num_examples) * data_loss

    def predict(self, X):
        self.feedforward(X)
        return np.argmax(self.probs, axis=1)

    def backprop(self, X, y):
        num_examples = len(X)
        delta3 = self.probs
        delta3[range(num_examples), np.argmax(y, axis=1)] -= 1
        dW2 = np.dot(self.a1.T, delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = np.dot(delta3, self.W2.T) * self.diff_actFun(self.z1, self.actFun_type)
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0, keepdims=True)
        return dW1, dW2, db1, db2

    def fit_model(self, X, y, epsilon=0.001, num_passes=20000, print_loss=True):
        for i in range(0, num_passes):
            self.feedforward(X)
            dW1, dW2, db1, db2 = self.backprop(X, y)

            dW2 += self.reg_lambda * self.W2
            dW1 += self.reg_lambda * self.W1

            self.W1 += -epsilon * dW1
            self.b1 += -epsilon * db1
            self.W2 += -epsilon * dW2
            self.b2 += -epsilon * db2

            if print_loss and i % 1000 == 0:
                print("Loss after iteration %i: %f" % (i, self.calculate_loss(X, y)))

def main():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    encoder = OneHotEncoder(sparse_output=False)
    y_encoded = encoder.fit_transform(y.reshape(-1, 1))

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Increase hidden units to 50 and use Leaky ReLU
    model = NeuralNetwork(nn_input_dim=4, nn_hidden_dim=50, nn_output_dim=3, actFun_type='leaky_relu', reg_lambda=0.001)

    model.fit_model(X_train, y_train)

    predictions = model.predict(X_test)
    print("Predictions:", predictions)
    print("True labels:", np.argmax(y_test, axis=1))

if __name__ == "__main__":
    main()
