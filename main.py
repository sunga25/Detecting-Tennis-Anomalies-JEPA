import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
import optuna

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
RANDOM_SEED = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 200

def load_data(start_year=2000, end_year=2017):
    dfs = []
    for year in range(start_year, end_year + 1):
        file_path = f'atp_matches_{year}.csv'
        try:
            df = pd.read_csv(file_path, low_memory=False)
            required_columns = ['tourney_id', 'surface', 'winner_id', 'loser_id', 'winner_name', 'loser_name', 
                                'winner_age', 'loser_age', 'winner_rank', 'loser_rank', 'tourney_date']
            if not all(col in df.columns for col in required_columns):
                logging.warning(f"File {file_path} is missing some required columns. Skipping this file.")
                continue
            dfs.append(df)
            logging.info(f"Data loaded successfully from {file_path}")
        except FileNotFoundError:
            logging.warning(f"File not found: {file_path}")
        except pd.errors.EmptyDataError:
            logging.warning(f"Empty file: {file_path}")
        except Exception as e:
            logging.error(f"Error loading data from {file_path}: {str(e)}")

    if not dfs:
        raise ValueError("No data files were successfully loaded.")

    combined_df = pd.concat(dfs, ignore_index=True)
    if combined_df.empty:
        raise ValueError("The combined DataFrame is empty after processing all files.")
    return combined_df

def preprocess_data(df):
    label_encoders = {}
    for col in ['tourney_id', 'surface', 'winner_id', 'loser_id']:
        df[col] = df[col].astype(str)
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    df['tourney_date'] = pd.to_datetime(df['tourney_date'], format='%Y%m%d', errors='coerce')
    df = df.dropna(subset=['tourney_date'])

    return df, label_encoders

def engineer_features(df):
    numeric_cols = ['winner_age', 'loser_age', 'winner_rank', 'loser_rank']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['age_difference'] = df['winner_age'] - df['loser_age']
    df['rank_difference'] = df['loser_rank'] - df['winner_rank']

    numeric_columns = numeric_cols + ['age_difference', 'rank_difference']
    df = df.dropna(subset=numeric_columns)

    return df, numeric_columns

class JointEmbeddedModel(nn.Module):
    def __init__(self, categorical_dims, numerical_dim, embedding_dim, hidden_dim, dropout_rate=0.3):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(dim, embedding_dim) for dim in categorical_dims])
        self.fc1 = nn.Linear(len(categorical_dims) * embedding_dim + numerical_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x_cat, x_num):
        embedded = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat(embedded + [x_num], dim=1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        return self.fc3(x).squeeze()

def create_dataloader(X, y, batch_size=64):
    x_cat, x_num = X
    # Ensure tensors are not empty
    if len(x_cat) == 0 or len(x_num) == 0:
        raise ValueError("Input data for dataloader is empty.")
    dataset = TensorDataset(torch.tensor(x_cat, dtype=torch.long),
                            torch.tensor(x_num, dtype=torch.float32),
                            torch.tensor(y, dtype=torch.float32))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def train_model(model, dataloader, val_data, epochs=20, learning_rate=0.001, weight_decay=0, patience=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience, verbose=True)
    scaler = GradScaler() if device.type == 'cuda' else None

    best_val_loss = float('inf')
    early_stopping_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x_cat, x_num, y in dataloader:
            x_cat, x_num, y = x_cat.to(device), x_num.to(device), y.to(device)
            optimizer.zero_grad()
            if scaler:
                with autocast(device_type='cuda'):
                    outputs = model(x_cat, x_num)
                    loss = criterion(outputs, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(x_cat, x_num)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        val_predictions = evaluate_model(model, val_data[0])
        val_loss = np.mean((val_predictions - val_data[1]) ** 2)
        scheduler.step(val_loss)
        logging.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                logging.info(f"Early stopping triggered after {epoch+1} epochs")
                break

def evaluate_model(model, X):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    x_cat, x_num = X

    if len(x_cat.shape) == 1:
        x_cat = x_cat.reshape(1, -1)
    if len(x_num.shape) == 1:
        x_num = x_num.reshape(1, -1)

    x_cat = torch.tensor(x_cat, dtype=torch.long).to(device)
    x_num = torch.tensor(x_num, dtype=torch.float32).to(device)

    with torch.no_grad():
        outputs = model(x_cat, x_num)
    return outputs.cpu().numpy()

def objective(trial):
    embedding_dim = trial.suggest_int('embedding_dim', 8, 64)
    hidden_dim = trial.suggest_int('hidden_dim', 32, 256)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    weight_decay = trial.suggest_float('weight_decay', 1e-8, 1e-3, log=True)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)

    model = JointEmbeddedModel(categorical_dims, numerical_dim, embedding_dim, hidden_dim, dropout_rate)
    dataloader = create_dataloader(X_train, y_train, batch_size=batch_size)
    train_model(model, dataloader, (X_val, y_val), epochs=10, learning_rate=learning_rate, weight_decay=weight_decay)

    val_predictions = evaluate_model(model, X_val)
    val_loss = np.mean((val_predictions - y_val) ** 2)
    return val_loss

def enhanced_anomaly_detection(model, X, df_subset, eps=0.5, min_samples=5, threshold=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    x_cat, x_num = X
    if len(x_cat.shape) == 1:
        x_cat = x_cat.reshape(-1, len(categorical_columns))
    if len(x_num.shape) == 1:
        x_num = x_num.reshape(-1, len(numeric_columns))

    x_cat = torch.tensor(x_cat, dtype=torch.long).to(device)
    x_num = torch.tensor(x_num, dtype=torch.float32).to(device)
    with torch.no_grad():
        embedded = [emb(x_cat[:, i]) for i, emb in enumerate(model.embeddings)]
        embeddings = torch.cat(embedded, dim=1).cpu().numpy()
        outputs = model(x_cat, x_num).cpu().numpy()

    scaler = StandardScaler()
    embeddings = scaler.fit_transform(embeddings)

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(embeddings)

    df_subset['anomaly'] = labels
    df_subset['expected_rank_difference'] = outputs

    if threshold is None:
        threshold = np.std(df_subset['rank_difference'] - df_subset['expected_rank_difference']) * 2

    df_subset['positive_anomaly'] = (df_subset['rank_difference'] - df_subset['expected_rank_difference']) > threshold
    df_subset['negative_anomaly'] = (df_subset['expected_rank_difference'] - df_subset['rank_difference']) > threshold

    anomalies = df_subset[(df_subset['positive_anomaly']) | (df_subset['negative_anomaly'])]

    positive_anomalies = anomalies[anomalies['positive_anomaly']]
    negative_anomalies = anomalies[anomalies['negative_anomaly']]

    logging.info(f"Positive Anomalies: {len(positive_anomalies)}")
    logging.info(f"Negative Anomalies: {len(negative_anomalies)}")

    # Count positive and negative anomalies per player, year, and tournament
    player_positive_anomalies = pd.concat([
        positive_anomalies['winner_name'],
        positive_anomalies['loser_name']
    ]).value_counts()

    player_negative_anomalies = pd.concat([
        negative_anomalies['winner_name'],
        negative_anomalies['loser_name']
    ]).value_counts()

    year_anomalies = anomalies['tourney_date'].dt.year.value_counts()
    tournament_anomalies = anomalies['tourney_id'].value_counts()

    # Save player anomalies counts to CSV
    player_positive_anomalies.to_csv('players_with_most_positive_anomalies.csv', header=['positive_anomalies'])
    player_negative_anomalies.to_csv('players_with_most_negative_anomalies.csv', header=['negative_anomalies'])
    year_anomalies.to_csv('years_with_most_anomalies.csv', header=['anomalies'])
    tournament_anomalies.to_csv('tournaments_with_most_anomalies.csv', header=['anomalies'])

    # Plotting DBSCAN results
    plt.figure(figsize=(10, 6))
    reduced_embeddings = TSNE(n_components=2).fit_transform(embeddings)
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.colorbar(label='Cluster Labels (Anomalies in -1)')
    plt.title('DBSCAN Clustering of Embeddings for Anomaly Detection')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.savefig('anomaly_detection_plot.png')
    plt.close()

    return anomalies

if __name__ == "__main__":
    try:
        df = load_data()
        df, label_encoders = preprocess_data(df)
        df, numeric_columns = engineer_features(df)

        categorical_columns = ['tourney_id', 'surface', 'winner_id', 'loser_id']
        X_cat = df[categorical_columns].values
        X_num = df[numeric_columns].values
        y = df['rank_difference'].values

        X_cat_train, X_cat_test, X_num_train, X_num_test, y_train, y_test, train_indices, test_indices = train_test_split(
            X_cat, X_num, y, df.index, test_size=TEST_SIZE, random_state=RANDOM_SEED)

        categorical_dims = [len(label_encoders[col].classes_) for col in categorical_columns]
        numerical_dim = len(numeric_columns)

        X_train = (X_cat_train, X_num_train)
        X_val = (X_cat_test[:VALIDATION_SIZE], X_num_test[:VALIDATION_SIZE])
        y_val = y_test[:VALIDATION_SIZE]

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=20)

        best_params = study.best_params
        logging.info(f"Best Hyperparameters: {best_params}")

        model = JointEmbeddedModel(categorical_dims, numerical_dim, best_params['embedding_dim'], 
                                   best_params['hidden_dim'], best_params['dropout_rate'])
        dataloader = create_dataloader(X_train, y_train, batch_size=best_params['batch_size'])
        train_model(model, dataloader, (X_val, y_val), epochs=20, learning_rate=best_params['learning_rate'], 
                    weight_decay=best_params['weight_decay'])

        model.load_state_dict(torch.load('best_model.pt'))
        test_predictions = evaluate_model(model, (X_cat_test, X_num_test))
        test_mse = np.mean((test_predictions - y_test) ** 2)
        logging.info(f"Final Test MSE: {test_mse}")

        anomalies = enhanced_anomaly_detection(model, (X_cat_test, X_num_test), df.loc[test_indices])

        # Save test predictions
        np.save('test_predictions.npy', test_predictions)

        # Save anomalies to CSV
        anomalies.to_csv('anomalies.csv', index=False)

        logging.info("Test predictions and anomalies saved successfully.")

        torch.save(model.state_dict(), 'final_model.pt')

        logging.info("Script execution completed successfully.")
    except Exception as e:
        logging.error(f"An error occurred during script execution: {str(e)}")
