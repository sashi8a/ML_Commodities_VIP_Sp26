import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
from VIP_abstract_class_template import BaseForecastModel

class _BiGRUModel(nn.Module):
    """
    Internal PyTorch Module for Bi-GRU with Attention and Skip Connections.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.2):
        super(_BiGRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        
        # Bidirectional GRU layers
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention Mechanism
        # We will use a simple dot-product attention
        self.attention = nn.Linear(hidden_dim * 2, 1)
        
        # Skip Connection (Residual)
        # We project input to hidden_dim * 2 to match GRU output if dims differ
        self.skip_projection = nn.Linear(input_dim, hidden_dim * 2)
        
        # Output Layer
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        
        # GRU Output: (batch_size, seq_len, hidden_dim * 2)
        gru_out, _ = self.gru(x)
        
        # Attention scores
        # (batch_size, seq_len, 1)
        attn_scores = self.attention(gru_out)
        attn_weights = torch.softmax(attn_scores, dim=1)
        
        # Context vector (weighted sum of GRU outputs)
        # (batch_size, hidden_dim * 2)
        context_vector = torch.sum(attn_weights * gru_out, dim=1)
        
        # Skip Connection
        # We take the last input of the sequence for the skip connection
        # (batch_size, input_dim) -> (batch_size, hidden_dim * 2)
        skip_input = x[:, -1, :] 
        skip_out = self.skip_projection(skip_input)
        
        # Combine Context and Skip
        combined = context_vector + skip_out
        
        # Output
        out = self.fc(combined)
        return out

class BiGRUAttentionModel(BaseForecastModel):
    """
    Bi-GRU with Attention and Skip Connections for Time Series Forecasting.
    """
    def __init__(self, task_type='regression', input_dim=1, hidden_dim=64, num_layers=2, dropout=0.2, learning_rate=0.001, epochs=10, batch_size=32):
        super().__init__(task_type=task_type, input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout, learning_rate=learning_rate, epochs=epochs, batch_size=batch_size)
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        
        self.model = None
        self.criterion = None
        self.optimizer = None
        
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def _build_model(self):
        self.model = _BiGRUModel(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=1, # output dimension is 1 since we are predicting one value
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)
        
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def fit(self, X_train, y_train):
        """
        Train the model using PyTorch training loop.
        X_train shape: (num_samples, seq_len, input_dim)
        y_train shape: (num_samples,)
        """
        
        if self.model is None:
            self._build_model()
            
        # Convert to Tensor
        X_tensor = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(self.device)
        
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model.train()
        for epoch in range(self.epochs):
            batch_losses = []
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                batch_losses.append(loss.item())
            
            # Optional: Print progress
            if (epoch + 1) % max(1, self.epochs // 10) == 0:
                print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {np.mean(batch_losses):.4f}")

    def predict(self, X):
        """
        Make predictions.
        X shape: (num_samples, seq_len, input_dim)
        """
        if self.model is None:
             raise ValueError("Model not trained! Call fit() first.")
             
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy()
            
        return predictions.flatten()

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model.
        """
        predictions = self.predict(X_test)
        
        mse = np.mean((y_test - predictions) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_test - predictions))
        
        # R² score
        ss_total = np.sum((y_test - np.mean(y_test)) ** 2)
        ss_residual = np.sum((y_test - predictions) ** 2)
        r2 = 1 - (ss_residual / ss_total) if ss_total != 0 else 0
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }

    def save(self, filepath: str):
        """
        Save model state and hyperparameters.
        """
        state = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'hyperparameters': self.hyperparameters,
             # Also save the explicitly stored attributes in case they were modified or needed
            'config': {
                'input_dim': self.input_dim,
                'hidden_dim': self.hidden_dim,
                'num_layers': self.num_layers,
                'dropout': self.dropout,
                'learning_rate': self.learning_rate,
                'epochs': self.epochs,
                'batch_size': self.batch_size
            }
        }
        torch.save(state, filepath)
        print(f"Model saved to {filepath}")

    def load(self, filepath: str):
        """
        Load model state.
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Restore hyperparameters
        config = checkpoint.get('config', {})
        self.input_dim = config.get('input_dim', self.input_dim)
        self.hidden_dim = config.get('hidden_dim', self.hidden_dim)
        self.num_layers = config.get('num_layers', self.num_layers)
        self.dropout = config.get('dropout', self.dropout)
        self.learning_rate = config.get('learning_rate', self.learning_rate)
        # Note: epochs and batch_size are mainly for training, so restoring them is optional but good practice
        self.epochs = config.get('epochs', self.epochs)
        self.batch_size = config.get('batch_size', self.batch_size)
        
        # Rebuild model matches the saved config
        self._build_model()
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded from {filepath}")


if __name__ == '__main__':
    # Synthetic Data Test
    print("Generating synthetic data...")
    # Generate random sequences: (num_samples, seq_len, input_dim)
    N = 1000
    seq_len = 10
    input_dim = 1
    
    X = np.random.randn(N, seq_len, input_dim)
    # Simple target: sum of the sequence plus some noise
    y = np.sum(X, axis=1).flatten() + np.random.randn(N) * 0.1
    
    # Split
    split = int(0.8 * N)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    print("Initializing model...")
    model = BiGRUAttentionModel(
        input_dim=input_dim,
        hidden_dim=32,
        num_layers=1,
        epochs=5,
        batch_size=32
    )
    
    print("Training model...")
    model.fit(X_train, y_train)
    
    print("Evaluating model...")
    metrics = model.evaluate(X_test, y_test)
    print("Metrics:", metrics)
    
    print("Testing Save/Load...")
    model.save('test_bi_gru_model.pth')
    
    loaded_model = BiGRUAttentionModel(input_dim=input_dim, hidden_dim=32, num_layers=1)
    loaded_model.load('test_bi_gru_model.pth')
    
    # Check predictions match
    pred_orig = model.predict(X_test[:5])
    pred_load = loaded_model.predict(X_test[:5])
    
    print("Original Predictions:", pred_orig)
    print("Loaded Predictions:  ", pred_load)
    print("Match:", np.allclose(pred_orig, pred_load))
