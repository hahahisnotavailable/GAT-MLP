import os
import pandas as pd
import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, classification_report
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import random
from collections import defaultdict
import networkx as nx
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
CLQ_DIR = os.path.join(BASE_DIR, 'clq_files')
EXCEL_DIR = os.path.join(BASE_DIR, 'dataset')
MODEL_DIR = os.path.join(BASE_DIR, 'GNN_models')
CACHED_DIR = os.path.join(BASE_DIR, 'cached_graphs')
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CACHED_DIR, exist_ok=True)
LOG_FILE = "training.log"


global_feature_scaler = StandardScaler()
def parse_clq_file(file_name):
    cache_path = os.path.join(CACHED_DIR, f"{file_name}.pt")
    if os.path.exists(cache_path):
        try:
            data = torch.load(cache_path, weights_only=False)
            if data.x.shape[1] == 2:
                return data
            else:
                print(f" {file_name} feature dimension discrepancy（{data.x.shape[1]}）")
        except Exception as e:
            print(f"cache read failure {file_name}: {e}")

    file_path = os.path.join(CLQ_DIR, f"{file_name}.clq")
    try:
        edges_raw = []
        node_ids = set()
        with open(file_path, 'r') as f:
            for line in f:
                if line.startswith('e'):
                    _, u, v = line.strip().split()
                    u, v = int(u), int(v)
                    edges_raw.append((u, v))
                    node_ids.add(u)
                    node_ids.add(v)

        if not edges_raw:
            return None


        sorted_nodes = sorted(node_ids)
        node_id_map = {old_id: new_id for new_id, old_id in enumerate(sorted_nodes)}
        num_nodes = len(node_id_map)


        edges = [(node_id_map[u], node_id_map[v]) for u, v in edges_raw]
        edge_index = torch.tensor(edges + [(v, u) for u, v in edges], dtype=torch.long).t().contiguous()


        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        G.add_edges_from(edges)


        deg = np.array([val for _, val in G.degree()])
        core = np.array([val for _, val in sorted(nx.core_number(G).items())])


        features = np.vstack([deg, core]).T  # (num_nodes, 2)
        features = MinMaxScaler().fit_transform(features)
        node_features = torch.tensor(features, dtype=torch.float)

        data = Data(x=node_features, edge_index=edge_index)
        torch.save(data, cache_path)
        return data

    except Exception as e:
        print(f"{file_path} wrong: {e}")
        return None




def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def build_label_dataset(label_column, excel_file="method2_dataset_class.xlsx"):
    try:
        excel_path = os.path.join(EXCEL_DIR, excel_file)
        if not os.path.exists(excel_path):
            raise FileNotFoundError(f"Excelnotfound: {excel_path}")
        df = pd.read_excel(excel_path)

        feature_cols = ['V', 'E', 'dmax', 'davg', 'D', 'r', 'T', 'Tavg', 'Tmax', 'Kavg', 'k', 'K']
        required_cols = ["Graph Name", label_column] + feature_cols
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"missing: {', '.join(missing_cols)}")

        df = df.dropna(subset=required_cols)
        clq_files = {f.replace('.clq', '') for f in os.listdir(CLQ_DIR) if f.endswith('.clq')}
        valid_graphs = set(df["Graph Name"]).intersection(clq_files)
        if not valid_graphs:
            raise ValueError("No matching graph")
        df = df[df["Graph Name"].isin(valid_graphs)]


        global global_feature_scaler
        df[feature_cols] = global_feature_scaler.fit_transform(df[feature_cols])

        label_set = sorted(set(df[label_column]))
        label_map = {label: idx for idx, label in enumerate(label_set)}

        dataset = []
        for _, row in df.iterrows():
            name = row["Graph Name"]
            graph_data = parse_clq_file(name)
            if graph_data:
                graph_data.y = torch.tensor([label_map[row[label_column]]], dtype=torch.long)
                graph_data.global_feats = torch.tensor(row[feature_cols].values.astype(np.float32))
                dataset.append(graph_data)

        return dataset, label_map
    except Exception as e:
        print(f"Dataset construction failed: {e}")
        return [], {}
'''
def build_multilabel_dataset(label_cols, excel_file="method3_dataset_class.xlsx"):
    try:
        excel_path = os.path.join(EXCEL_DIR, excel_file)
        if not os.path.exists(excel_path):
            raise FileNotFoundError(f"Excel: {excel_path}")

        all_sheets = pd.read_excel(excel_path, sheet_name=None)
        combined_df = None

        for sheet_name, sheet_df in all_sheets.items():
            
            cols_in_sheet = [col for col in sheet_df.columns if col in ["Graph Name"] + label_cols]
            sheet_df = sheet_df[cols_in_sheet]
            if combined_df is None:
                combined_df = sheet_df
            else:
                combined_df = pd.merge(combined_df, sheet_df, on="Graph Name", how="outer")

        feature_cols = ['V', 'E', 'dmax', 'davg', 'D', 'r', 'T', 'Tavg', 'Tmax', 'Kavg', 'k', 'K']
        df = pd.read_excel(excel_path, sheet_name=0)  
        df = pd.merge(df[["Graph Name"] + feature_cols], combined_df, on="Graph Name", how="inner")

        df = df.dropna(subset=["Graph Name"])
        clq_files = {f.replace('.clq', '') for f in os.listdir(CLQ_DIR) if f.endswith('.clq')}
        valid_graphs = set(df["Graph Name"]).intersection(clq_files)
        if not valid_graphs:
            raise ValueError("无匹配的图数据")
        df = df[df["Graph Name"].isin(valid_graphs)]
        global global_feature_scaler
        df[feature_cols] = global_feature_scaler.fit_transform(df[feature_cols])

        dataset = []
        for _, row in df.iterrows():
            name = row["Graph Name"]
            graph_data = parse_clq_file(name)  
            if graph_data:
                labels = torch.tensor(row[label_cols].values.astype(np.float32))
                graph_data.y = labels
                graph_data.global_feats = torch.tensor(row[feature_cols].values.astype(np.float32))
                dataset.append(graph_data)

        
        return dataset
    except Exception as e:

        return []
'''
class GATWithGlobal(torch.nn.Module):
    def __init__(self, node_in_dim, hidden_dim, global_in_dim, out_dim, dropout=0.5, heads=4):
        super(GATWithGlobal, self).__init__()

        self.gat1 = GATConv(node_in_dim, hidden_dim, heads=heads, dropout=dropout)
        self.gat2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=False, dropout=dropout)


        self.global_mlp = torch.nn.Sequential(
            torch.nn.Linear(global_in_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU()
        )


        self.final_fc = torch.nn.Linear(hidden_dim * 2, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index, batch, global_feats):
        if global_feats.dim() == 1:
            global_feats = global_feats.unsqueeze(0)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.gat1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.gat2(x, edge_index))
        x = global_mean_pool(x, batch)  # [batch_size, hidden_dim]


        global_emb = self.global_mlp(global_feats)

        # embedding concat
        combined = torch.cat([x, global_emb], dim=1)
        return self.final_fc(combined)
class TrainingLogger:
    def __init__(self, is_multilabel):
        self.is_multilabel = is_multilabel
        self.epochs = []
        self.losses = []
        self.metrics = []

    def log_epoch(self, epoch, loss, metric):
        self.epochs.append(epoch)
        self.losses.append(loss)
        self.metrics.append(metric)
'''
    def plot_curves(self, title_suffix=""):
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.epochs, self.losses, '-o', color='blue')
        plt.title(f'Training Loss {title_suffix}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.subplot(1, 2, 2)
        plt.plot(self.epochs, self.metrics, '-o', color='red')
        plt.title(f'Training {"F1 Score" if self.is_multilabel else "Accuracy"} {title_suffix}')
        plt.xlabel('Epoch')
        plt.ylabel('Metric Value')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(BASE_DIR, f'training_curves_{"multi" if self.is_multilabel else "single"}.png'))
        plt.show()
'''
def train_model(model, dataset, is_multilabel, label_info, params):
    logger = TrainingLogger(is_multilabel)
    epochs, batch_size, lr, hidden_dim, dropout, patience, seed, test_size = params.values()
    if not is_multilabel:
        train_data, val_test_data = train_test_split(dataset, test_size=test_size, random_state=seed,
                                                     stratify=[data.y.item() for data in dataset])
    else:
        train_data, val_test_data = train_test_split(dataset, test_size=test_size, random_state=seed)
    val_data, test_data = train_test_split(val_test_data, test_size=0.5, random_state=seed)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    if is_multilabel:
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        all_labels = [data.y.item() for data in dataset]
        class_weights = compute_class_weight('balanced', classes=np.unique(all_labels), y=all_labels)
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    best_metric = 0.0
    early_stop_counter = 0
    model_name = "best_model.pt"
    model_path = os.path.join(MODEL_DIR, model_name)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            global_feats = torch.stack([d.global_feats for d in batch.to_data_list()]).to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch, global_feats)
            loss = criterion(out, batch.y.float() if is_multilabel else batch.y.squeeze())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        model.eval()
        with torch.no_grad():
            if is_multilabel:
                val_preds, val_labels = [], []
                for batch in val_loader:
                    batch = batch.to(device)
                    global_feats = torch.stack([d.global_feats for d in batch.to_data_list()]).to(device)
                    out = model(batch.x, batch.edge_index, batch.batch, global_feats)
                    pred = torch.sigmoid(out) > 0.5
                    val_preds.append(pred.cpu())
                    val_labels.append(batch.y.cpu())
                y_true = torch.cat(val_labels).numpy()
                y_pred = torch.cat(val_preds).numpy()
                current_metric = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            else:
                val_preds, val_labels = [], []
                for batch in val_loader:
                    batch = batch.to(device)
                    global_feats = torch.stack([d.global_feats for d in batch.to_data_list()]).to(device)
                    out = model(batch.x, batch.edge_index, batch.batch, global_feats)
                    preds = out.argmax(dim=1)
                    val_preds.extend(preds.cpu().numpy())
                    val_labels.extend(batch.y.cpu().numpy().squeeze())
                current_metric = accuracy_score(val_labels, val_preds)

        logger.log_epoch(epoch, avg_loss, current_metric)
        print(f"Epoch {epoch:03d} | Loss: {avg_loss:.4f} | {'Val F1 Score' if is_multilabel else 'Val Accuracy'}: {current_metric:.4f}")

        if current_metric > best_metric:
            best_metric = current_metric
            torch.save(model.state_dict(), model_path)
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"stop training")
                break

    print(f"Training completed")
    return model_path, test_loader, is_multilabel, label_info, logger
def evaluate_model(model_path, test_loader, is_multilabel, label_map=None, hidden_dim=32, dropout=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_channels = 4 if is_multilabel else len(label_map)
    model = GATWithGlobal(
        node_in_dim=2,
        hidden_dim=hidden_dim,
        global_in_dim=12,
        out_dim=out_channels,
        dropout=dropout
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    if is_multilabel:
        y_true_list, y_pred_list = [], []
        for batch in test_loader:
            batch = batch.to(device)
            global_feats = torch.stack([d.global_feats for d in batch.to_data_list()]).to(device)
            out = model(batch.x, batch.edge_index, batch.batch, global_feats)
            pred = torch.sigmoid(out) > 0.5
            y_pred_list.append(pred.cpu().numpy())
            y_true_list.append(batch.y.cpu().numpy())
        y_true = np.concatenate(y_true_list)
        y_pred = np.concatenate(y_pred_list)

        print("\n=== Multi-label evaluation results ===")
        print(f"Subset accuracy: {np.mean(np.all(y_true == y_pred, axis=1)):.4f}")
        print(f"Hanming Accuracy rate: {1 - hamming_loss(y_true, y_pred):.4f}")
        print(f"Macro F1 score: {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
        print(f"Micro F1 score: {f1_score(y_true, y_pred, average='micro', zero_division=0):.4f}")
    else:
        test_preds, test_labels = [], []
        for batch in test_loader:
            batch = batch.to(device)
            global_feats = torch.stack([d.global_feats for d in batch.to_data_list()]).to(device)
            out = model(batch.x, batch.edge_index, batch.batch, global_feats)
            preds = out.argmax(dim=1)
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(batch.y.cpu().numpy().squeeze())

        target_names = [k for k, _ in sorted(label_map.items(), key=lambda x: x[1])]

        acc = accuracy_score(test_labels, test_preds)
        weighted_f1 = f1_score(test_labels, test_preds, average='weighted', zero_division=0)
        macro_f1 = f1_score(test_labels, test_preds, average='macro', zero_division=0)
        print(f"acc: {acc:.4f}")
        print(f"weighted-F1: {weighted_f1:.4f}")
        print(f"macro-F1: {macro_f1:.4f}")
        labels = list(range(len(target_names)))
        print("Classification report:\n", classification_report(
            test_labels, test_preds,
            labels=labels,
            target_names=target_names,
            zero_division=0
        ))


def test_model(is_multilabel, label_info=None, model_name=None, params=None):
    model_name = model_name or ("best_model.pt")
    model_path = os.path.join(MODEL_DIR, model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件未找到: {model_path}")

    if is_multilabel:
        label_cols = ["is_Clisat", "is_LMC", "is_MoMC", "is_dOmega"]
        dataset = build_multilabel_dataset(label_cols)
        if not dataset:
            print("Loading failed")
            return
        test_loader = DataLoader(dataset, batch_size=params["batch_size"], shuffle=False)
    else:
        dataset, label_map = build_label_dataset("class")
        if not dataset:
            print("Loading failed")
            return
        _, test_data = train_test_split(
            dataset, test_size=0.2, random_state=params["seed"],
            stratify=[data.y.item() for data in dataset]
        )
        if not test_data:
            print("Test datasets empty")
            return
        test_loader = DataLoader(test_data, batch_size=params["batch_size"], shuffle=False)
        label_info = label_map

    evaluate_model(model_path, test_loader, is_multilabel, label_info, params["hidden_dim"], params["dropout"])
if __name__ == "__main__":
    params = {
        "epochs": 50,
        "batch_size": 16,
        "lr": 0.001,
        "hidden_dim": 32,
        "dropout": 0.5,
        "patience": 10,
        "seed": 100,
        "test_size": 0.2
    }

    set_seed(params["seed"])
    print("Select the run mode:")
    print("1. Training")
    print("2. Test only (use existing trained model)")
    while True:
        mode = input("number 1/2: ").strip()
        if mode in {"1", "2"}:
            break

    if mode == "1":
        print("Loading dataset and training")
        dataset, label_map = build_label_dataset("class")
        if not dataset:
            print("Dataset failed to load")
            exit(1)

        model = GATWithGlobal(
            node_in_dim=2,
            hidden_dim=params["hidden_dim"],
            global_in_dim=12,
            out_dim=len(label_map),
            dropout=params["dropout"]
        )

        model_path, test_loader, _, label_info, logger = train_model(
            model, dataset, is_multilabel=False, label_info=label_map, params=params
        )

    else:
        print("Loading test data")
        dataset, label_map = build_label_dataset("class")
        if not dataset:
            print("Dataset failed to load")
            exit(1)

        _, test_data = train_test_split(
            dataset, test_size=params["test_size"],
            random_state=params["seed"],
            stratify=[data.y.item() for data in dataset]
        )
        test_loader = DataLoader(test_data, batch_size=params["batch_size"], shuffle=False)

        model_path = os.path.join(MODEL_DIR, "best_model.pt")
        if not os.path.exists(model_path):
            print(f"can't find {model_path}, Train the model first。")
            exit(1)

        evaluate_model(model_path, test_loader, is_multilabel=False,
                       label_map=label_map, hidden_dim=params["hidden_dim"],
                       dropout=params["dropout"])
#The full version will be uploaded soon