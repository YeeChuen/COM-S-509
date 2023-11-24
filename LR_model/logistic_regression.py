# Author: Yee Chuen Teoh
import numpy as np
import copy
import warnings

warnings.filterwarnings('ignore')

class LogisticRegression():

    def __init__(self, learning_rate, iteration, bias = True, classifier = "binomial"):
        self.learning_rate = learning_rate
        self.iteration = iteration

        self.cost_list = []

        self.bias = bias

        if classifier not in ["binomial", "multinomial"] :
            print("LR model Error: input for 'classifier' parameter needs to be 'binomial' or 'multinomial'.")
        self.classifier = classifier
        if self.classifier == "multinomial":
            self.learning_rate *= 0.01

    def one_hot_encoding(self,Y):
        OneHotEncoding = []
        encoding = []
        for i in range(len(Y)):
            if(Y[i] == 0): encoding = np.array([1,0]) #Class 1, if y = 0
            elif(Y[i] == 1): encoding = np.array([0,1]) #Class 2, if y = 1
            else: encoding = np.array([Y[i],Y[i]]) #Class 2, if y = 1

            OneHotEncoding.append(encoding)

        return np.array(OneHotEncoding)

    def fit(self, X_train, y_train):
        m, n = X_train.shape # m --> number of data (row), n --> number of feature map (col)

        
        if self.classifier == "multinomial":
            self.w = np.array([[0,0] for _ in range(n)])        # weight per feature
        elif self.classifier == "binomial":
            self.w = np.zeros(n)        # weight per feature
        self.b = 0                  # bias
        X = copy.deepcopy(X_train)  # datas, 2d
        y = copy.deepcopy(y_train)  # class/outcome, 2d

        for i in range(self.iteration):
            z = np.array(np.dot(X, self.w) + self.b, dtype=np.float32) # z = wX + b
            y_hat = self.function(z) # y_hat is probabilistic prediction for y
            
            Ti = self.one_hot_encoding(y)

            # ignore cost at the moment
            #cost = self.cost_function(m, y, y_hat)
            #self.cost_list.append(cost)

            if self.classifier == "multinomial":
                self.gradient_descent(m, y_hat, Ti, X)
            elif self.classifier == "binomial":
                self.gradient_descent(m, y_hat, y, X)


            if i % 10 == 0:
                # print(y_hat)
                # print(f"loss: {cost}")
                continue

    def cost_function(self, m, y, y_hat):
        cost = -(1/m) * np.sum( y*np.log(y_hat) + (1-y) * np.log(1-y_hat))
        return cost 

    def gradient_descent(self, m, y_hat, y, X):
        dW = (1/m)*np.dot((y_hat-y).T, X)
        dB = (1/m)*np.sum(y_hat - y)
        self.w = self.w - self.learning_rate*dW.T

        if self.bias:
            self.b = self.b - self.learning_rate*dB
        else:
            self.b = 0

    def function(self, z):
        if self.classifier == "binomial":   
            sigmoid = []
            for value in z:
                if value >= 0:
                    a = np.exp(-value)
                    sigmoid.append( 1 / (1 + a))
                else:
                    a = np.exp(value)
                    sigmoid.append( a / (1 + a))
            return np.array(sigmoid)
        elif self.classifier == "multinomial":
            return (np.exp(z).T / np.sum(np.exp(z), axis=1)).T

    def predict(self, X, prob):
        if self.classifier == "binomial":
            z = np.array(np.dot(X, self.w) + self.b, dtype=np.float32) # z = wX + b
            y_pred = self.function(z) # y prediction
            y_pred = np.where(y_pred > prob, 1, 0) # y = 1 if y[i]>0.5, else 0
            return y_pred
        elif self.classifier == "multinomial":
            z = np.array(np.dot(X, self.w) + self.b, dtype=np.float32) # z = wX + b
            y_pred = self.function(z) # y prediction
            y_pred = np.where(y_pred > prob, 1, 0) # y = 1 if y[i]>0.5, else 0
            return y_pred
    
    def score(self, X, y_true, prob = 0.5):
        y_pred = self.predict(X, prob)

        if self.classifier == "multinomial":
            y_true = self.one_hot_encoding(y_true)
        accuracy = np.mean(y_pred == y_true)
        return accuracy
    
    def get_params(self):
        return self.w, self.b, self.cost_list

# from github
import copy
import numpy as np
from sklearn.metrics import accuracy_score

class CustomLogisticRegression():
    def __init__(self):
        self.losses = []
        self.train_accuracies = []

    def fit(self, x, y, epochs):
        x = self._transform_x(x)
        y = self._transform_y(y)

        self.weights = np.zeros(x.shape[1])
        self.bias = 0

        for i in range(epochs):
            x_dot_weights = np.matmul(self.weights, x.transpose()) + self.bias
            pred = self._sigmoid(x_dot_weights)
            loss = self.compute_loss(y, pred)
            error_w, error_b = self.compute_gradients(x, y, pred)
            self.update_model_parameters(error_w, error_b)

            pred_to_class = [1 if p > 0.5 else 0 for p in pred]
            self.train_accuracies.append(accuracy_score(y, pred_to_class))
            self.losses.append(loss)

    def compute_loss(self, y_true, y_pred):
        # binary cross entropy
        y_zero_loss = y_true * np.log(y_pred + 1e-9)
        y_one_loss = (1-y_true) * np.log(1 - y_pred + 1e-9)
        return -np.mean(y_zero_loss + y_one_loss)

    def compute_gradients(self, x, y_true, y_pred):
        # derivative of binary cross entropy
        difference =  y_pred - y_true
        gradient_b = np.mean(difference)
        gradients_w = np.matmul(x.transpose(), difference)
        gradients_w = np.array([np.mean(grad) for grad in gradients_w])

        return gradients_w, gradient_b

    def update_model_parameters(self, error_w, error_b):
        self.weights = self.weights - 0.1 * error_w
        self.bias = self.bias - 0.1 * error_b

    def predict(self, x):
        x_dot_weights = np.matmul(x, self.weights.transpose()) + self.bias
        probabilities = self._sigmoid(x_dot_weights)
        return [1 if p > 0.5 else 0 for p in probabilities]

    def _sigmoid(self, x):
        return np.array([self._sigmoid_function(value) for value in x])

    def _sigmoid_function(self, x):
        if x >= 0:
            z = np.exp(-x)
            return 1 / (1 + z)
        else:
            z = np.exp(x)
            return z / (1 + z)

    def _transform_x(self, x):
        x = copy.deepcopy(x)
        if hasattr(x, 'values'):
            return x.values
        return x

    def _transform_y(self, y):
        y = copy.deepcopy(y)
        if hasattr(y, 'values'):
            return y.values.reshape(y.shape[0], 1)
        return y.reshape(y.shape[0], 1)
    
    def score(self, x_test, y_test):
        pred = self.predict(x_test)
        score = accuracy_score(y_test, pred)
        return score
    
class ShortLogisticRegression():

    def __init__(self, learning_rate, iterations):
        self.learning_rate = learning_rate
        self.iterations = iterations
        pass

    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))
    
    def fit(self, X, Y):
    
        m = X.shape[0]
        n = X.shape[1]
        
        self.W = np.zeros((n,1))
        self.B = 0
        
        self.cost_list = []
        
        for i in range(self.iterations):
            
            Z = np.dot(self.W.T, X.T) + self.B
            A = self.sigmoid(Z)
            
            # cost function
            cost = -(1/m)*np.sum( Y*np.log(A) + (1-Y)*np.log(1-A))
            
            # Gradient Descent
            dW = (1/m)*np.dot(A-Y, X)
            dB = (1/m)*np.sum(A - Y)
            
            self.W = self.W - self.learning_rate*dW.T
            self.B = self.B - self.learning_rate*dB
            
            # Keeping track of our cost function value
            self.cost_list.append(cost)
    
    def score(self, X, Y):
        Z = np.dot(self.W.T, X.T) + self.B
        A = self.sigmoid(Z)
        
        A = A > 0.5
        
        A = np.array(A, dtype = 'int64')
        
        accuracy = np.mean(A == Y)

        return accuracy
            
class PELogisticRegression:

    def __init__(self, learning_rate=0.001, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # init parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        # gradient descent
        for _ in range(self.n_iters):
            # approximate y with linear combination of weights and x, plus bias
            linear_model = np.dot(X, self.weights) + self.bias
            # apply sigmoid function
            y_predicted = self._sigmoid(linear_model)

            # compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            # update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def score(self, X, Y):
        A = self.predict(X)
        accuracy = np.mean(A == Y)
        return accuracy
