import numpy as np
from dataclasses import dataclass

# ____ Data Classes ______________________________________________________________________________
@dataclass
class Parameters:
    weights: np.ndarray
    biases: np.ndarray

@dataclass
class Cache:
    Z : np.ndarray
    A : np.ndarray

# ____ Activation Functions ______________________________________________________________________________
ACTIVATIONS = {
        'relu': (lambda Z: np.maximum(0, Z),
                 lambda Z: (Z > 0).astype(int)),
        'sigmoid': (lambda Z: 1 / (1 + np.exp(-Z)),
                    lambda Z: (S := 1 / (1 + np.exp(-Z))) * (1 - S)),
        'softplus': (lambda Z: np.log(1 + np.exp(Z)),
                     lambda Z: 1 / (1 + np.exp(-Z)))
    }

class NeuralNetwork:
    def __init__(
            self,
            architecture: list[int],
            activation: list[str],
            batch_size: int,
            alpha: float = 0.01,
            epochs: int = 1000

    ):
        # check for input
        if not isinstance(architecture, list):
            raise ValueError("The architecture must be a list of neurons per each layer!")
        
        if isinstance(activation, list):
            if len(architecture) != len(activation):
                raise ValueError("Architecture and Activation must have the same length, an activation function per each layer!")
            
            if not np.all(np.isin(activation, ['relu', 'sigmoid', 'softplus'])):
                raise ValueError(f"Invalid activation functions, expected : ['relu', 'sigmoid', 'softplus'], got {np.unique(activation)}!")
        
        else:
            raise ValueError('Activation must be a list of strings!')

        self.activation = activation
        self.alpha = alpha
        self.epochs = epochs
        self.batch_size = batch_size
        self.architecture = architecture

        self.parameters = self.initialize_parameters()
    # ____ Public Methods ______________________________________________________________________________
    def fit(
            self,
            X: np.ndarray,
            y: np.ndarray,
            loss_func: str
    ):
        X, y = self.check_input(X, y)
        if loss_func not in ['mse', 'cross_entropy']:
            raise ValueError(f"Invalid loss function, expected ['mse', 'cross_entropy'], got {loss_func}!")
        
        self.build_network(X, y, loss_func)

        return self
    
    def predict(self, X: np.ndarray):
        predictions, _ = self.forward(X)
        return predictions
    
    # ____ Internal Methods ______________________________________________________________________________
    def forward(self, X: np.ndarray):
            cache = []
            A_prev = X
            for i, params in enumerate(self.parameters):
                f, _ = ACTIVATIONS[self.activation[i]]
                Z = A_prev @ params.weights.T + params.biases
                A = f(Z)
                cache.append(Cache(Z, A))
                A_prev = A
            return A_prev, cache
            
    
    def initialize_parameters(self):
        params = []

        for i in range(len(self.architecture) - 1):
            n_inputs = self.architecture[i]
            n_neurons = self.architecture[i+1]
            
            if self.activation[i] == 'relu':
                weights = np.random.randn(n_neurons, n_inputs) * np.sqrt(2 / n_inputs)
            else:
                weights = np.random.randn(n_neurons, n_inputs) * np.sqrt(1 / n_inputs)
            biases = np.zeros(n_neurons)

            params.append(Parameters(weights, biases))
        return params

    def backward(self, X: np.ndarray, y: np.ndarray, cache: list[Cache]):
        m = X.shape[0]
        L = len(self.parameters)
        delta = cache[-1].A - y
        grads = [None] * L
        for i in reversed(range(L)):
            A_prev = cache[i - 1].A if i > 0 else X

            dW = A_prev.T @ delta / m 
            db = delta.mean(axis=0)
      
            grads[i] = (dW.T, db)

            if i > 0:
                _, f_prime = ACTIVATIONS[self.activation[i - 1]]
                delta = (delta @ self.parameters[i].weights) * f_prime(cache[i - 1].Z)
        return grads
    
    def update_parameters(self, new_params: list[np.ndarray]):
        for params, (dw, db) in zip(self.parameters, new_params):
            params.weights -= dw * self.alpha
            params.biases -= db * self.alpha
    
    def build_network(self, X: np.ndarray, y: np.ndarray, loss_func: str):
        self.loss_history = []
        for _ in range(self.epochs):
            indices = np.random.permutation(len(X))
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for i in range(0, len(X), self.batch_size):
                X_batch = X_shuffled[i: i+self.batch_size]
                y_batch = y_shuffled[i: i+self.batch_size]
                predictions, cache = self.forward(X_batch)
                match loss_func:
                    case "mse":
                        loss = self.mse(y_batch, predictions).sum() / len(y_batch)
                        self.loss_history.append(float(loss))
                    case "cross_entropy":
                        loss = self.cross_entropy(y_batch, predictions).sum() / len(y_batch)
                        self.loss_history.append(float(loss))
                    case _ :
                        raise  ValueError('Invalid loss function!')
                new_params = self.backward(X_batch, y_batch, cache)
                self.update_parameters(new_params)
                    
    
    # ____ Cost Functions ______________________________________________________________________________
    def cross_entropy(self, y_true: np.ndarray, y_pred: np.ndarray):
        return - np.log(y_pred[y_true == 1])

    def mse(self, y_true: np.ndarray, y_pred: np.ndarray):
        return np.sum((y_true - y_pred)**2, axis=1) / y_true.shape[-1]
    
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
        
        if X.shape[-1] != self.architecture[0]:
            raise ValueError(f"The input layer has {self.architecture[0]} neurons, but X has {X.shape[-1]} features")
        
        if y.shape != (X.shape[0], self.architecture[-1]):
            raise ValueError(f"The target feature must be of shape {(X.shape[0], self.architecture[-1])}!")

        # Checking for dtypes
        if not np.issubdtype(X.dtype, np.number) or not np.issubdtype(y.dtype, np.number):
            raise ValueError("X and y must contain numeric values!")
        
        return X, y
