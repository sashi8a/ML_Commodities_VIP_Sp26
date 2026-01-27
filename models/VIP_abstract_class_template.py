"""
Base Template for Forecasting Models

This is a simple template showing what methods your model needs to have.
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseForecastModel(ABC):
    """
    Simple base class for forecasting models.

    Your model should inherit from this and implement all the methods below.

    Example:
        class MyModel(BaseForecastModel):
            def __init__(self, task_type, **params):
                super().__init__(task_type, **params)
                # Your setup code here

            def fit(self, X_train, y_train):
                # Your training code here
                pass

            def predict(self, X):
                # Your prediction code here
                return predictions

            def evaluate(self, X_test, y_test):
                # Your evaluation code here
                return metrics
    """

    def __init__(self, task_type: str, **hyperparameters):
        """
        Initialize your model.

        Args:
            task_type: Either 'regression' or 'classification'
            **hyperparameters: Your model's parameters (e.g., learning_rate=0.01)

        Students should:
            - Store the task_type
            - Store any hyperparameters
            - Initialize your model architecture/components
        """
        self.task_type = task_type
        self.hyperparameters = hyperparameters

    @abstractmethod
    def fit(self, X_train, y_train):
        """
        Train your model on the training data.

        Args:
            X_train: Training features, shape (n_samples, n_features)
            y_train: Training targets, shape (n_samples,)

        You should implement:
            - Data preprocessing if needed
            - Model training logic
            - Store trained parameters/weights

        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Make predictions on new data.

        Args:
            X: Features to predict on, shape (n_samples, n_features)

        Returns:
            predictions: Your predictions, shape (n_samples,)

        You should implement:
            - Apply same preprocessing as in fit()
            - Generate predictions using trained model
            - Return predictions as numpy array
        """
        pass

    @abstractmethod
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance on test data.

        Args:
            X_test: Test features
            y_test: True test labels/values

        Returns:
            metrics: Dictionary with metric names and values
                    e.g., {'rmse': 0.5, 'mae': 0.3, 'r2': 0.85}

        You should implement:
            - Get predictions on test data
            - Compute appropriate metrics based on task_type:
              * Regression: RMSE, MAE, R², MAPE
              * Classification: Accuracy, Precision, Recall, F1
            - Return metrics as a dictionary

        """
        pass

    @abstractmethod
    def save(self, filepath: str):
        """
        Save your trained model to a file.

        Args:
            filepath: Path where to save (e.g., 'models/my_model.pkl')

        You should implement:
            - Save model parameters/weights
            - Save any preprocessing parameters (mean, std, etc.)
            - Save hyperparameters
        """
        pass

    @abstractmethod
    def load(self, filepath: str):
        """
        Load a previously saved model.

        Args:
            filepath: Path to the saved model file

        You should implement:
            - Load model parameters/weights
            - Load preprocessing parameters
            - Restore the model to a usable state

        """
        pass









    '''


    """
Example: Simple Linear Regression Model

This shows you exactly what to implement.
"""

import numpy as np
import pickle


class SimpleLinearRegression(BaseForecastModel):
    """
    A simple linear regression model: y = X @ weights + bias

    This is a minimal example showing all required methods.
    """

    def __init__(self, task_type='regression', learning_rate=0.01):
        # Call parent constructor
        super().__init__(task_type=task_type, learning_rate=learning_rate)

        # Initialize model parameters
        self.weights = None
        self.bias = None
        self.learning_rate = learning_rate

    def fit(self, X_train, y_train):
        """Train using closed-form solution."""
        # Add bias column to X
        X_with_bias = np.column_stack([np.ones(len(X_train)), X_train])

        # Closed-form solution: weights = (X^T X)^-1 X^T y
        self.weights = np.linalg.lstsq(X_with_bias, y_train, rcond=None)[0]

        # Split weights into bias and coefficients
        self.bias = self.weights[0]
        self.weights = self.weights[1:]

    def predict(self, X):
        """Make predictions."""
        if self.weights is None:
            raise ValueError("Model not trained! Call fit() first.")

        # y = X @ weights + bias
        predictions = X @ self.weights + self.bias
        return predictions

    def evaluate(self, X_test, y_test):
        """Compute evaluation metrics."""
        predictions = self.predict(X_test)

        # Calculate metrics
        mse = np.mean((y_test - predictions) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_test - predictions))

        # R² score
        ss_total = np.sum((y_test - np.mean(y_test)) ** 2)
        ss_residual = np.sum((y_test - predictions) ** 2)
        r2 = 1 - (ss_residual / ss_total)

        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }

    def save(self, filepath: str):
        """Save the model."""
        save_dict = {
            'weights': self.weights,
            'bias': self.bias,
            'learning_rate': self.learning_rate,
            'task_type': self.task_type
        }

        with open(filepath, 'wb') as f:
            pickle.dump(save_dict, f)

        print(f"Model saved to {filepath}")

    def load(self, filepath: str):
        """Load the model."""
        with open(filepath, 'rb') as f:
            save_dict = pickle.load(f)

        self.weights = save_dict['weights']
        self.bias = save_dict['bias']
        self.learning_rate = save_dict['learning_rate']

        print(f"Model loaded from {filepath}")


# ============================================================================
# Example of execution on synthetic data
# ============================================================================

if __name__ == "__main__":
    # 1. Create some data
    X_train = np.random.randn(100, 5)
    y_train = X_train @ np.array([1, 2, 3, 4, 5]) + np.random.randn(100) * 0.1

    X_test = np.random.randn(20, 5)
    y_test = X_test @ np.array([1, 2, 3, 4, 5]) + np.random.randn(20) * 0.1

    # 2. Create and train model
    model = SimpleLinearRegression(learning_rate=0.01)
    model.fit(X_train, y_train)

    # 3. Make predictions
    predictions = model.predict(X_test)
    print("Predictions:", predictions[:5])

    # 4. Evaluate
    metrics = model.evaluate(X_test, y_test)
    print("\nMetrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    # 5. Save and load
    model.save('linear_model.pkl')

    new_model = SimpleLinearRegression()
    new_model.load('linear_model.pkl')

    # Test loaded model
    new_predictions = new_model.predict(X_test)
    print(f"\nLoaded model works: {np.allclose(predictions, new_predictions)}")

    '''

