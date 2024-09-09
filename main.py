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
            # Basic validation
            required_columns = ['tourney_id', 'surface', 'winner_id', 'loser_id', 'winner_age', 'loser_age', 'winner_rank', 'loser_rank', 'tourney_date']
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
    return combined_df

def preprocess_data(df):
    label_encoders = {}
    for col in ['tourney_id', 'surface', 'winner_id', 'loser_id']:
        # Convert column to string type
        df[col] = df[col].astype(str)
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Convert tourney_date to datetime, handling potential errors
    df['tourney_date'] = pd.to_datetime(df['tourney_date'], errors='coerce')

    # Handle potential NaN values after conversion
    df = df.dropna(subset=['tourney_date'])

    return df, label_encoders

def engineer_features(df):
    # Convert age and rank columns to numeric, coercing errors to NaN
    numeric_cols = ['winner_age', 'loser_age', 'winner_rank', 'loser_rank']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Calculate differences, which will result in NaN if either value is NaN
    df['age_difference'] = df['winner_age'] - df['loser_age']
    df['rank_difference'] = df['winner_rank'] - df['loser_rank']

    numeric_columns = numeric_cols + ['age_difference', 'rank_difference']

    # Drop rows with NaN values in numeric columns
    df = df.dropna(subset=numeric_columns)

    return df, numeric_columns

class JointEmbeddedModel(nn.Module):
    def __init__(self, categorical_dims, numerical_dim, embedding_dim, hidden_dim):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(dim, embedding_dim) for dim in categorical_dims])
        self.fc1 = nn.Linear(len(categorical_dims) * embedding_dim + numerical_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        self.relu = nn.ReLU()

    def forward(self, x_cat, x_num):
        embedded = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat(embedded + [x_num], dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x).squeeze()

def create_dataloader(X, y, batch_size=64):
    x_cat, x_num = X
    dataset = TensorDataset(torch.tensor(x_cat, dtype=torch.long),
                            torch.tensor(x_num, dtype=torch.float32),
                            torch.tensor(y, dtype=torch.float32))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def train_model(model, dataloader, epochs=20, learning_rate=0.001, weight_decay=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    scaler = GradScaler()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x_cat, x_num, y in dataloader:
            x_cat, x_num, y = x_cat.to(device), x_num.to(device), y.to(device)
            optimizer.zero_grad()
            with autocast():
                outputs = model(x_cat, x_num)
                loss = criterion(outputs, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        scheduler.step(avg_loss)
        logging.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

def evaluate_model(model, X, y):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    x_cat, x_num = X
    x_cat = torch.tensor(x_cat, dtype=torch.long).to(device)
    x_num = torch.tensor(x_num, dtype=torch.float32).to(device)
    y = torch.tensor(y, dtype=torch.float32).to(device)
    with torch.no_grad():
        outputs = model(x_cat, x_num)
        loss = nn.MSELoss()(outputs, y)
    return loss.item()

def objective(trial):
    embedding_dim = trial.suggest_int('embedding_dim', 8, 64)
    hidden_dim = trial.suggest_int('hidden_dim', 32, 256)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-8, 1e-3)

    model = JointEmbeddedModel(categorical_dims, numerical_dim, embedding_dim, hidden_dim)
    dataloader = create_dataloader(X_train, y_train, batch_size=batch_size)
    train_model(model, dataloader, epochs=10, learning_rate=learning_rate, weight_decay=weight_decay)

    val_loss = evaluate_model(model, X_val, y_val)
    return val_loss

def enhanced_anomaly_detection(model, X, df_subset, eps=0.5, min_samples=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    x_cat, x_num = X
    x_cat = torch.tensor(x_cat, dtype=torch.long).to(device)
    x_num = torch.tensor(x_num, dtype=torch.float32).to(device)
    with torch.no_grad():
        embedded = [emb(x_cat[:, i]) for i, emb in enumerate(model.embeddings)]
        embeddings = torch.cat(embedded, dim=1).cpu().numpy()
    scaler = StandardScaler()
    embeddings = scaler.fit_transform(embeddings)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(embeddings)

    df_subset['anomaly'] = labels
    anomalies = df_subset[df_subset['anomaly'] == -1]
    total_anomalies = len(anomalies)
    anomalies_per_year = anomalies['tourney_date'].dt.year.value_counts()
    player_anomalies = pd.concat([anomalies['winner_name'], anomalies['loser_name']])
    players_most_anomalies = player_anomalies.value_counts()
    tournaments_most_anomalies = anomalies['tourney_name'].value_counts()

    anomalies.to_csv('anomalies.csv', index=False)
    anomalies_per_year.to_csv('anomalies_per_year.csv', header=['Number of Anomalies'])
    players_most_anomalies.to_csv('players_most_anomalies.csv', header=['Number of Anomalies'])
    tournaments_most_anomalies.to_csv('tournaments_most_anomalies.csv', header=['Number of Anomalies'])

    plt.figure(figsize=(10, 6))
    reduced_embeddings = TSNE(n_components=2).fit_transform(embeddings)
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.colorbar(label='Cluster Labels (Anomalies in -1)')
    plt.title('DBSCAN Clustering of Embeddings for Anomaly Detection')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.savefig('anomaly_detection_plot.png')
    plt.close()

    logging.info(f"Total Anomalies: {total_anomalies}")
    logging.info(f"Anomalies Per Year: \n{anomalies_per_year}")
    logging.info(f"Players with Most Anomalies: \n{players_most_anomalies.head(10)}")
    logging.info(f"Tournaments with Most Anomalies: \n{tournaments_most_anomalies.head(10)}")

    return labels

if __name__ == "__main__":
    try:
        df = load_data()
        df, label_encoders = preprocess_data(df)
        df, numeric_columns = engineer_features(df)

        categorical_columns = ['tourney_id', 'surface', 'winner_id', 'loser_id']
        X_cat = df[categorical_columns].values.astype(np.int64)
        X_num = df[numeric_columns].values.astype(np.float32)
        y = df['winner_rank'].values.astype(np.float32)

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

        model = JointEmbeddedModel(categorical_dims, numerical_dim, best_params['embedding_dim'], best_params['hidden_dim'])
        dataloader = create_dataloader(X_train, y_train, batch_size=best_params['batch_size'])
        train_model(model, dataloader, epochs=20, learning_rate=best_params['learning_rate'], weight_decay=best_params['weight_decay'])

        test_loss = evaluate_model(model, (X_cat_test, X_num_test), y_test)
        logging.info(f"Final Test Loss: {test_loss}")

        anomaly_labels = enhanced_anomaly_detection(model, (X_cat_test, X_num_test), df.loc[test_indices])

        torch.save(model.state_dict(), 'model.pt')
        pd.DataFrame(anomaly_labels, columns=['anomaly']).to_csv('anomaly_labels.csv', index=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        with torch.no_grad():
            outputs = model(torch.tensor(X_cat_test, dtype=torch.long).to(device), 
                            torch.tensor(X_num_test, dtype=torch.float32).to(device))
            outputs = outputs.cpu().numpy()

        pd.DataFrame(outputs, columns=['model_output']).to_csv('model_outputs.csv', index=False)
        model_params = {name: param.data.cpu().numpy() for name, param in model.named_parameters()}
        pd.DataFrame([model_params]).to_csv('model_params.csv', index=False)

        logging.info("Script execution completed successfully.")
    except Exception as e:
        logging.error(f"An error occurred during script execution: {str(e)}")