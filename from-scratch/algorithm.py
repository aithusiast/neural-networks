import numpy as np
from dataclasses import dataclass

# ____ Data Classes ______________________________________________________________________________
@dataclass
class Parameters:
    weights: np.ndarray
    biases: np.ndarray

@dataclass
class Cache:
    Z: np.ndarray
    A: np.ndarray

# ____ Numerically Stable Activations ______________________________________________
def _sigmoid(Z: np.ndarray):
    out = np.empty_like(Z, dtype=float)
    pos = Z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-Z[pos]))
    exp_z = np.exp(Z[~pos])
    out[~pos] = exp_z / (1.0 + exp_z)
    return out

def _softplus(Z: np.ndarray):
    return np.maximum(Z, 0) + np.log1p(np.exp(-np.abs(Z)))

def _softmax(Z: np.ndarray):
    Z_shift = Z - np.max(Z, axis=1, keepdims=True)
    expZ = np.exp(Z_shift)
    return expZ / np.sum(expZ, axis=1, keepdims=True)

# ____ Activation Functions ______________________________________________________________________
ACTIVATIONS = {
    'relu':     (lambda Z: np.maximum(0, Z), lambda Z: (Z > 0).astype(float)),
    'sigmoid':  (_sigmoid, lambda Z: _sigmoid(Z) * (1 - _sigmoid(Z))),
    'softplus': (_softplus, _sigmoid),
    'softmax':  (_softmax, None),
    'linear':   (lambda Z: Z, lambda Z: np.ones_like(Z))
}

# ____ Main Class __________________________________________________________________________________
class NeuralNetwork:
    def __init__(
            self,
            architecture: list[int],
            activation: list[str],
            batch_size: int,
            alpha: float = 0.01,
            epochs: int = 1000
    ):
        # parameters checks
        if not isinstance(architecture, list):
            raise ValueError("The architecture must be a list of neurons per each layer!")

        if not all(isinstance(n, int) and not isinstance(n, bool) and n > 0 for n in architecture):
            raise ValueError("Every entry in architecture must be a positive integer!")

        if not isinstance(activation, list):
            raise ValueError('Activation must be a list of strings!')

        if len(architecture) - 1 != len(activation):
            raise ValueError("There must be an activation per each layer!")

        if not np.all(np.isin(activation, ['relu', 'sigmoid', 'softplus', 'softmax', 'linear'])):
            raise ValueError(
                f"Invalid activation functions, expected : "
                f"['relu', 'sigmoid', 'softplus', 'softmax', 'linear'], got {np.unique(activation)}!"
            )

        if 'softmax' in activation[:-1]:
            raise ValueError("'softmax' can only be used as the final layer's activation!")
        
        if 'linear' in activation[:-1]:
            raise ValueError("Only the output layer can have a linear activation!")


        int_args = {'batch_size': batch_size, 'epochs': epochs}
        for name, val in int_args.items():
            if isinstance(val, bool) or not isinstance(val, int):
                raise ValueError(f'{name} must be a non-null positive integer')
            if val <= 0:
                raise ValueError(f'Invalid {name} value, expected {name} > 0, got {val}')

        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
            raise ValueError('alpha must be a numeric value')
        if alpha <= 0:
            raise ValueError(f'Invalid alpha value, expected alpha > 0, got {alpha}')

        self.activation = activation
        self.alpha = alpha
        self.epochs = epochs
        self.batch_size = batch_size
        self.architecture = architecture

        self.parameters = self.initialize_parameters()

    # ____ Public Methods __________________________________________________________________________
    def fit(
            self,
            X: np.ndarray,
            y: np.ndarray,
            loss_func: str
    ):
        X, y = self.check_input(X, y)

        if loss_func not in ['mse', 'cross_entropy']:
            raise ValueError(f"Invalid loss function, expected ['mse', 'cross_entropy'], got {loss_func}!")

        if self.activation[-1] == 'softmax' and loss_func != 'cross_entropy':
            raise ValueError("The output layer with a softmax activation requires the 'cross_entropy' loss!")
        
        if loss_func == 'cross_entropy' and self.activation[-1] not in ('sigmoid', 'softmax'):
            raise ValueError(
                "'cross_entropy' requires the final layer's activation to be "
                "'sigmoid' (binary) or 'softmax' (multi-class), "
                f"got '{self.activation[-1]}'!"
            )
        
        if self.activation[-1] == 'linear' and loss_func != 'mse':
            raise ValueError("The output layer with a linear activation requires the 'mse' loss!")

        self.build_network(X, y, loss_func)
        return self

    def predict(self, X: np.ndarray):
        X = self.check_prediction_input(X)
        predictions, _ = self.forward(X)
        return predictions

    # ____ Internal Methods _________________________________________________________________________
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
            n_neurons = self.architecture[i + 1]

            if self.activation[i] == 'relu':
                weights = np.random.randn(n_neurons, n_inputs) * np.sqrt(2 / n_inputs)
            else:
                weights = np.random.randn(n_neurons, n_inputs) * np.sqrt(1 / n_inputs)
            biases = np.zeros(n_neurons)

            params.append(Parameters(weights, biases))
        return params

    def backward(self, X: np.ndarray, y: np.ndarray, cache: list[Cache], loss_func: str):
        m = X.shape[0]
        L = len(self.parameters)
        grads = [None] * L

        match loss_func:
            case 'cross_entropy':
                delta = cache[-1].A - y

            case 'mse':
                _, f_prime = ACTIVATIONS[self.activation[-1]]
                n_features = y.shape[-1]
                delta = (2 / n_features) * (cache[-1].A - y) * f_prime(cache[-1].Z)

            case _:
                raise ValueError('Invalid loss function!')

        for i in reversed(range(L)):
            A_prev = cache[i - 1].A if i > 0 else X

            dW = A_prev.T @ delta / m
            db = delta.mean(axis=0)

            grads[i] = (dW.T, db)

            if i > 0:
                _, f_prime = ACTIVATIONS[self.activation[i - 1]]
                delta = (delta @ self.parameters[i].weights) * f_prime(cache[i - 1].Z)

        return grads

    def update_parameters(self, new_params: list[tuple[np.ndarray, np.ndarray]]):
        for params, (dw, db) in zip(self.parameters, new_params):
            params.weights -= dw * self.alpha
            params.biases -= db * self.alpha

    def build_network(self, X: np.ndarray, y: np.ndarray, loss_func: str):
        self.loss_history = []
        m = X.shape[0]

        losses = {
            'mse': self.mse,
            'cross_entropy': self.cross_entropy
        }
        loss_fn = losses[loss_func]  

        for _ in range(self.epochs):
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_loss = 0
            for i in range(0, len(X), self.batch_size):
                X_batch = X_shuffled[i: i + self.batch_size]
                y_batch = y_shuffled[i: i + self.batch_size]
                y_pred, cache = self.forward(X_batch)

                loss = loss_fn(y_batch, y_pred)
                epoch_loss += loss * len(X_batch)

                new_params = self.backward(X_batch, y_batch, cache, loss_func)
                self.update_parameters(new_params)

            epoch_loss /= m
            self.loss_history.append(float(epoch_loss))

    # ____ Cost Functions ___________________________________________________________________________
    def cross_entropy(self, y_true: np.ndarray, y_pred: np.ndarray):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        if self.activation[-1] == 'softmax':
            return np.mean(np.sum(-y_true * np.log(y_pred), axis=1))
        return np.mean(-(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))

    def mse(self, y_true: np.ndarray, y_pred: np.ndarray):
        return np.mean(np.sum((y_true - y_pred) ** 2, axis=1) / y_true.shape[-1])

    # ____ Validation Methods _______________________________________________________________________
    def check_input(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.ndim > 2:
            raise ValueError(f"X must be 1D or 2D array, got {X.ndim}D array!")

        if y.ndim == 1 and self.architecture[-1] == 1:
            y = y.reshape(-1, 1)

        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X has {X.shape[0]} samples, but y has {y.shape[0]} samples!")

        if X.shape[-1] != self.architecture[0]:
            raise ValueError(f"The input layer has {self.architecture[0]} neurons, but X has {X.shape[-1]} features")

        if y.shape != (X.shape[0], self.architecture[-1]):
            raise ValueError(f"The target feature must be of shape {(X.shape[0], self.architecture[-1])}!")

        if not np.issubdtype(X.dtype, np.number) or not np.issubdtype(y.dtype, np.number):
            raise ValueError("X and y must contain numeric values!")

        return X, y

    def check_prediction_input(self, X: np.ndarray):
        X = np.asarray(X)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.ndim > 2:
            raise ValueError(f"X must be 1D or 2D array, got {X.ndim}D array!")

        if X.shape[-1] != self.architecture[0]:
            raise ValueError(f"The input layer has {self.architecture[0]} neurons, but X has {X.shape[-1]} features")

        if not np.issubdtype(X.dtype, np.number):
            raise ValueError("X must contain numeric values!")

        return X