import numpy as np
from typing import Optional


class RegressionNeuralNetwork:
    def __init__(
            self,
            architecture: list[int],
            activation: list[str] | str,
            alpha: float = 0.01,
            epochs: int = 1000,
            batch_size: int | None = None,

    ):
        # check for input
        if not isinstance(architecture, list):
            raise ValueError("The architecture must be a list of neurons per each layer!")
        
        if isinstance(activation, list):
            if len(architecture) != len(activation):
                raise ValueError("Architecture and Activation must have the same length, an activation function per each layer!")
            
            if not np.all(np.isin(activation, ['relu', 'sigmoid', 'softplus'])):
                raise ValueError(f"Invalid activation functions, expected : ['relu', 'sigmoid', 'softplus'], got {np.unique(activation)}!")
            
        #elif isinstance(activation, str) and (activation not in ['relu', 'sigmoid', 'softplus']):
            #raise ValueError(f"Invalid activation functions, expected : ['relu', 'sigmoid', 'softplus'], got {activation}!")
        
        else:
            raise ValueError('Activation must be a string or a list of strings!')

        self.activation = activation
        self.alpha = alpha
        self.epochs = epochs
        self.batch_size = batch_size
        self.architecture = architecture

        self.network = None

class NeuralNetwork:
    def __init__(
            self,
            architecture: list[int],
            activation: list[str] | str,
            alpha: float = 0.01,
            epochs: int = 1000,
            batch_size: int | None = None,

    ):
        # check for input
        if not isinstance(architecture, list):
            raise ValueError("The architecture must be a list of neurons per each layer!")
        
        if isinstance(activation, list):
            if len(architecture) != len(activation):
                raise ValueError("Architecture and Activation must have the same length, an activation function per each layer!")
            
            if not np.all(np.isin(activation, ['relu', 'sigmoid', 'softplus'])):
                raise ValueError(f"Invalid activation functions, expected : ['relu', 'sigmoid', 'softplus'], got {np.unique(activation)}!")
            
        #elif isinstance(activation, str) and (activation not in ['relu', 'sigmoid', 'softplus']):
            #raise ValueError(f"Invalid activation functions, expected : ['relu', 'sigmoid', 'softplus'], got {activation}!")
        
        else:
            raise ValueError('Activation must be a string or a list of strings!')

        self.activation = activation
        self.alpha = alpha
        self.epochs = epochs
        self.batch_size = batch_size
        self.architecture = architecture

        self.network = None
    
    # ____ Public Methods ______________________________________________________________________________
    def fit(
            self,
            X: np.ndarray,
            y: np.ndarray,
            loss: str
    ):
        X, y = self.check_input(X, y)
        if loss not in ['mse', 'entropy']:
            raise ValueError(f"Invalid loss function, expected ['mse', 'entropy'], got {loss}!")
        
        self.network = self.build_network(X, y, loss)

        return self
    
    # ____ Internal Methods ______________________________________________________________________________
    def forward(self, X: np.ndarray, index: int= 0):
            if index == len(self.architecture) -1 :
                return
            
            activations = {
                'relu': self.relu,
                'sigmoid': self.sigmoid,
                'softplus': self.softplus
            }
            n_inputs = self.architecture[index]
            n_neurons = self.architecture[index+1]

            weights = np.random.randn(n_neurons, n_inputs)
            biases = np.zeros(n_neurons)
            y = weights @ X + biases
            outputs = activations[self.activation[index]](y)
            
            next_layer = self.forward(outputs, index+1)

            return Network(outputs, weights, biases, next_layer)
    
    def backward(self, X: np.ndarray, y: np.ndarray, loss: str):
        pass
    
    def build_network(self, X: np.ndarray, y: np.ndarray, loss: str):
        for epoch in range(self.epochs):
            indices = np.random.permutation(len(X))
            X = X[indices]
            y = y[indices]
            for i in range(0, len(X), self.batch_size):
                batches = X[i: i+self.batch_size]
                match loss:
                    case "binary_classification_cost":
                        pass
                    case "multi_class_classification_cost":
                        pass
                    case "simple_regression_cost":
                        pass
                    case "multi_output_regression_cost":
                        pass
                    case _ :
                        raise  ValueError('Invalid loss function!')
                # backprop
                self.backward()
                # update
    
    # ____ Activation Functions ______________________________________________________________________________
    def relu(self, X: np.ndarray):
        X[X < 0] = 0
        return X

    def sigmoid(self, X: np.ndarray):
        return 1 / (1 + np.exp(-X))

    def softplus(self, X:np.ndarray):
        return np.log(1 + np.exp(X))

    # ____ Activation Functions Der______________________________________________________________________________
    def d_relu(self, X: np.ndarray):
        X[X < 0] = 0
        X[X > 0] = 1
        return X

    def d_sigmoid(self, X: np.ndarray):
        return self.sigmoid(X) * (1 - self.sigmoid(X))

    def d_softplus(self, X: np.ndarray):
        return self.sigmoid(X)
    
    # ____ Cost Functions ______________________________________________________________________________
    def binary_classification_cost(self, y_true: np.ndarray, y_pred: np.ndarray):
        return y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
    
    def multi_class_classification_cost(self, y_true: np.ndarray, y_pred: np.ndarray):
        return - np.log(y_pred[y_true == 1])

    def simple_regression_cost(self, y_true: np.ndarray, y_pred: np.ndarray):
        return (y_true - y_pred)**2

    def multi_output_regression_cost(self, y_true: np.ndarray, y_pred: np.ndarray):
        return np.sum(self.simple_regression_cost(y_true, y_pred)) / len(y_true)
    
    # ____ Validation Methods ______________________________________________________________________________
    def check_input(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X)
        y = np.asarray(y)

        # Checking for dimensions and shapes
        if X.ndim == 1:
            X = X.reshape(-1,1)

        if X.ndim > 2:
            raise ValueError(f"X must be 1D or 2D array, got {X.ndim}D array!")
        
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X has {X.shape[0]} samples, but y has {y.shape[0]} samples!")
        
        if y.shape != (X.shape[0], self.architecture[-1]):
            raise ValueError(f"The target feature must be of shape {(self.architecture[-1], X.shape[0])}!")

        # Checking for dtypes
        if not np.issubdtype(X.dtype, np.number) or not np.issubdtype(y.dtype, np.number):
            raise ValueError("X and y must contain numeric values!")
        
        return X, y
    


class Network:
    def __init__(self, outputs, weights, biases, next_layer):
        self.outputs = outputs
        self.weights = weights
        self.biases = biases
        self.next_layer = next_layer