"""
Model builders for the HIV-2 Drug Resistance Benchmark.
Factory pattern: build_model(name) returns sklearn-compatible estimator.
"""
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_config import RANDOM_SEED


def build_model(name, n_features=None):
    """
    Build a model by name. Returns sklearn-compatible estimator.
    
    Args:
        name: model key (e.g., "T4_Ridge", "T2_XGB")
        n_features: input dimensionality (needed for DL models)
    
    Returns: model with .fit(X, y) and .predict(X) interface
    """
    if name == "T0_mean":
        return ConstantMeanPredictor()
    
    elif name == "T1_hiv2eu":
        return HIV2EURuleBased()
    
    elif name == "T2_LR":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(
            max_iter=2000, C=1.0, random_state=RANDOM_SEED,
            class_weight="balanced"
        )
    
    elif name == "T2_RF":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=400, max_depth=20, random_state=RANDOM_SEED,
            n_jobs=-1, class_weight="balanced"
        )
    
    elif name == "T2_XGB":
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(
            n_estimators=400, max_depth=8, learning_rate=0.05,
            random_state=RANDOM_SEED, subsample=0.8
        )
    
    elif name == "T2_SVM":
        from sklearn.svm import SVC
        return SVC(
            kernel="rbf", C=1.0, gamma="scale",
            class_weight="balanced", probability=True,
            random_state=RANDOM_SEED
        )
    
    elif name == "T3_CNN":
        return CNNRegressor(n_features=n_features or 1980)
    
    elif name == "T3_LSTM":
        return LSTMRegressor(n_features=n_features or 1980)
    
    elif name == "T4_Ridge":
        from sklearn.linear_model import Ridge
        return Ridge(alpha=1.0)
    
    elif name == "T4_EN":
        from sklearn.linear_model import ElasticNet
        return ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000,
                         random_state=RANDOM_SEED)
    
    elif name == "T4_XGB":
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(
            n_estimators=400, max_depth=12, learning_rate=0.05,
            random_state=RANDOM_SEED, subsample=0.8
        )
    
    elif name == "T4_RF":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(
            n_estimators=400, max_depth=20, random_state=RANDOM_SEED,
            n_jobs=-1
        )
    
    elif name == "T5_ensemble":
        return EnsembleRegressor()
    
    else:
        raise ValueError(f"Unknown model: {name}")


# ── Custom model classes ────────────────────────────────────────────

class ConstantMeanPredictor:
    """Trivial baseline: predicts training mean for all test samples."""
    
    def __init__(self):
        self.mean_ = None
    
    def fit(self, X, y):
        self.mean_ = np.mean(y)
        return self
    
    def predict(self, X):
        return np.full(len(X), self.mean_)


class HIV2EURuleBased:
    """
    Rule-based baseline using HIV-2EU-style mutation penalty lookup.
    For regression benchmark, uses mutation count as proxy score.
    """
    
    def __init__(self):
        self.wt_ = None
    
    def fit(self, X, y):
        # Learn scaling from training data
        self.scale_ = np.std(y) if np.std(y) > 0 else 1.0
        self.offset_ = np.mean(y)
        return self
    
    def predict(self, X):
        # X is binary mutation matrix (N, 1980)
        # Simple: count mutations, scale to target range
        mutation_counts = X.sum(axis=1) / 20.0  # average mutations per position
        return mutation_counts * self.scale_ + self.offset_


class EnsembleRegressor:
    """Simple averaging ensemble of Ridge + RF + XGB."""
    
    def __init__(self):
        from sklearn.linear_model import Ridge
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        
        self.models_ = [
            Ridge(alpha=1.0),
            RandomForestRegressor(n_estimators=200, max_depth=15, 
                                 random_state=RANDOM_SEED, n_jobs=-1),
            GradientBoostingRegressor(n_estimators=200, max_depth=8,
                                     learning_rate=0.05, random_state=RANDOM_SEED)
        ]
        self.weights_ = [0.3, 0.35, 0.35]  # Slight tree bias
    
    def fit(self, X, y):
        for model in self.models_:
            model.fit(X, y)
        return self
    
    def predict(self, X):
        preds = np.array([m.predict(X) for m in self.models_])
        return np.average(preds, axis=0, weights=self.weights_)


# ── Deep Learning Models (PyTorch) ──────────────────────────────────

class CNNRegressor:
    """1D-CNN for sequence regression with sklearn-like interface."""
    
    def __init__(self, n_features=1980, seq_length=99, n_aa=20,
                 max_epochs=200, lr=1e-3, patience=20, batch_size=32):
        self.n_features = n_features
        self.seq_length = seq_length
        self.n_aa = n_aa
        self.max_epochs = max_epochs
        self.lr = lr
        self.patience = patience
        self.batch_size = batch_size
        self.model_ = None
        self.device_ = "cpu"
    
    def _build_model(self):
        import torch
        import torch.nn as nn
        
        class CNN(nn.Module):
            def __init__(self, n_aa, seq_length):
                super().__init__()
                self.conv = nn.Sequential(
                    nn.Conv1d(n_aa, 64, kernel_size=3, padding=1),
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    nn.Conv1d(64, 128, kernel_size=5, padding=2),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    nn.Conv1d(128, 256, kernel_size=7, padding=3),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.AdaptiveMaxPool1d(1),
                )
                self.head = nn.Sequential(
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(128, 1),
                )
            
            def forward(self, x):
                # x: (batch, seq_length * n_aa) → reshape to (batch, n_aa, seq_length)
                x = x.view(x.size(0), self.n_aa, -1)
                x = self.conv(x).squeeze(-1)
                return self.head(x).squeeze(-1)
        
        model = CNN(self.n_aa, self.seq_length)
        return model
    
    def fit(self, X, y):
        import torch
        from torch.utils.data import TensorDataset, DataLoader
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_ = device
        
        # Reshape to (N, seq_length, n_aa) for conv input
        X_t = torch.FloatTensor(X).to(device)
        y_t = torch.FloatTensor(y).to(device)
        
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model_ = self._build_model().to(device)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        criterion = torch.nn.MSELoss()
        
        best_loss = float("inf")
        patience_counter = 0
        best_state = None
        
        for epoch in range(self.max_epochs):
            self.model_.train()
            epoch_loss = 0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                pred = self.model_(X_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(loader)
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.model_.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break
        
        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.model_.eval()
        return self
    
    def predict(self, X):
        import torch
        
        self.model_.eval()
        X_t = torch.FloatTensor(X).to(self.device_)
        
        with torch.no_grad():
            preds = self.model_(X_t).cpu().numpy()
        
        return preds


class LSTMRegressor:
    """BiLSTM for sequence regression with sklearn-like interface."""
    
    def __init__(self, n_features=1980, seq_length=99, n_aa=20,
                 hidden=128, n_layers=2, max_epochs=200, lr=1e-3,
                 patience=20, batch_size=32):
        self.n_features = n_features
        self.seq_length = seq_length
        self.n_aa = n_aa
        self.hidden = hidden
        self.n_layers = n_layers
        self.max_epochs = max_epochs
        self.lr = lr
        self.patience = patience
        self.batch_size = batch_size
        self.model_ = None
        self.device_ = "cpu"
    
    def _build_model(self):
        import torch
        import torch.nn as nn
        
        class BiLSTM(nn.Module):
            def __init__(self, n_aa, hidden, n_layers):
                super().__init__()
                self.lstm = nn.LSTM(
                    n_aa, hidden, num_layers=n_layers,
                    batch_first=True, bidirectional=True, dropout=0.3
                )
                self.head = nn.Sequential(
                    nn.Linear(hidden * 2, 128),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(128, 1),
                )
            
            def forward(self, x):
                # x: (batch, seq_length * n_aa) → reshape to (batch, seq_length, n_aa)
                x = x.view(x.size(0), -1, self.n_aa)
                _, (h_n, _) = self.lstm(x)
                # Concatenate final hidden states from both directions
                h = torch.cat([h_n[-2], h_n[-1]], dim=1)
                return self.head(h).squeeze(-1)
        
        return BiLSTM(self.n_aa, self.hidden, self.n_layers)
    
    def fit(self, X, y):
        import torch
        from torch.utils.data import TensorDataset, DataLoader
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_ = device
        
        X_t = torch.FloatTensor(X).to(device)
        y_t = torch.FloatTensor(y).to(device)
        
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model_ = self._build_model().to(device)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        criterion = torch.nn.MSELoss()
        
        best_loss = float("inf")
        patience_counter = 0
        best_state = None
        
        for epoch in range(self.max_epochs):
            self.model_.train()
            epoch_loss = 0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                pred = self.model_(X_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(loader)
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.model_.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break
        
        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.model_.eval()
        return self
    
    def predict(self, X):
        import torch
        
        self.model_.eval()
        X_t = torch.FloatTensor(X).to(self.device_)
        
        with torch.no_grad():
            preds = self.model_(X_t).cpu().numpy()
        
        return preds


if __name__ == "__main__":
    print("Model builders loaded. Available models:")
    for name in ["T0_mean", "T1_hiv2eu", "T2_LR", "T2_RF", "T2_XGB", "T2_SVM",
                  "T3_CNN", "T3_LSTM", "T4_Ridge", "T4_EN", "T4_XGB", "T4_RF", "T5_ensemble"]:
        print(f"  {name}")
