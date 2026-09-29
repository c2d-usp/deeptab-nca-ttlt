import os
import json
import copy
import torch
import argparse
import numpy as np
import torch.nn as nn
from model_mlp import MLP
from tqdm.auto import tqdm
from data_val import get_dataset
from ttlt import ttlt_prediction
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, f1_score, classification_report
)

def train_model(model, X_train, y_train, X_val, y_val,
                lr=1e-3, weight_decay=1e-5, batch_size=50,
                patience=100, max_epochs=500):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=50
    )
    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.MSELoss() if y_train.dim() == 2 else nn.CrossEntropyLoss()

    best_loss = float('inf')
    patience_counter = 0
    best_state = None

    pbar = tqdm(range(max_epochs), desc="Training", unit="epoch")
    for epoch in pbar:
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val), y_val)

        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1

        pbar.set_postfix({
            "val_loss": f"{val_loss:.4f}",
            "best":     f"{best_loss:.4f}",
            "patience": f"{patience_counter}/{patience}",
        })

        if patience_counter > patience:
            break

    model.load_state_dict(best_state)
    return model

def compute_metrics(preds_np, truth_np, task):
    if task == "regression":
        return {
            "rmse": float(np.sqrt(mean_squared_error(truth_np, preds_np))),
            "mae":  float(mean_absolute_error(truth_np, preds_np)),
            "r2":   float(r2_score(truth_np, preds_np)),
        }
    else:  # classification
        preds_cls = preds_np.astype(int)
        truth_cls = truth_np.astype(int)
        report = classification_report(truth_cls, preds_cls, output_dict=True)
        return {
            "accuracy": float(accuracy_score(truth_cls, preds_cls)),
            "f1_macro": float(f1_score(truth_cls, preds_cls, average="macro")),
            "f1_weighted": float(f1_score(truth_cls, preds_cls, average="weighted")),
            "per_class_report": report,
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,
                        default='predict_students_dropout_and_academic_success')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    """
    Load dataset
    """
    dataset = args.dataset
    train_val_data, test_data, info = get_dataset(dataset, r"data")

    task_type = str(info.get("task_type", "")).lower() if isinstance(info, dict) else ""
    task = "classification" if task_type in {"binclass", "multiclass"} else "regression"

    X_train = train_val_data[0]["train"]
    X_val   = train_val_data[0]["val"]
    y_train = train_val_data[2]["train"]
    y_val   = train_val_data[2]["val"]
    X_test  = test_data[0]["test"]
    y_test  = test_data[2]["test"]

    """
    Preprocessing
    """
    x_scaler = StandardScaler()
    X_train = x_scaler.fit_transform(X_train)
    X_val   = x_scaler.transform(X_val)
    X_test  = x_scaler.transform(X_test)

    if task == "regression":
        y_scaler = StandardScaler()
        y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
        y_val   = y_scaler.transform(y_val.reshape(-1, 1)).ravel()
        y_test  = y_scaler.transform(y_test.reshape(-1, 1)).ravel()
    else:
        y_scaler = LabelEncoder()
        y_train = y_scaler.fit_transform(y_train)
        y_val   = y_scaler.transform(y_val)
        y_test  = y_scaler.transform(y_test)

    print(f"{dataset} train: {X_train.shape}  val: {X_val.shape}  test: {X_test.shape}")

    """
    Convert to torch tensors and move to device
    """
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_val_t   = torch.tensor(X_val,   dtype=torch.float32).to(device)
    X_test_t  = torch.tensor(X_test,  dtype=torch.float32).to(device)

    if task == "classification":
        y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
        y_val_t   = torch.tensor(y_val,   dtype=torch.long).to(device)
        y_test_t  = torch.tensor(y_test,  dtype=torch.long).to(device)
        output_dim = int(y_train_t.max().item() + 1)
    else:
        y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
        y_val_t   = torch.tensor(y_val,   dtype=torch.float32).unsqueeze(1).to(device)
        y_test_t  = torch.tensor(y_test,  dtype=torch.float32).unsqueeze(1).to(device)
        output_dim = 1

    """
    Model
    """
    model = MLP(
        input_dim=X_train_t.shape[1],
        output_dim=output_dim,
        num_layers=2,
        hidden_units=100,
        task=task
    ).to(device)
    model = train_model(model, X_train_t, y_train_t, X_val_t, y_val_t)

    """
    Standard inference
    """
    model.eval()
    with torch.no_grad():
        if task == "regression":
            preds_standard = model(X_test_t).squeeze(-1)   # [N]
        else:
            logits_standard = model(X_test_t)              # [N, C]
            preds_standard = torch.argmax(logits_standard, dim=1)  # [N]

    if task == "regression":
        preds_np  = y_scaler.inverse_transform(preds_standard.view(-1, 1).cpu().numpy()).squeeze()
        truth_np  = y_scaler.inverse_transform(y_test_t.view(-1, 1).cpu().numpy()).squeeze()
    else:
        preds_np  = preds_standard.cpu().numpy()
        truth_np  = y_test_t.cpu().numpy()

    standard = compute_metrics(preds_np, truth_np, task)

    """
    TTLT inference
    """
    pbar = tqdm(range(len(X_test_t)), desc="TTLT inference", unit="sample")
    preds = []

    nca = NeighborhoodComponentsAnalysis(init='identity', random_state=42)
    nca.fit(X_train_t.cpu().numpy(), y_train_t.cpu().numpy())

    for i in pbar:
        pred = ttlt_prediction(
            X_test_t[i], X_train_t, y_train_t,
            model=model,
            projection=nca,
            task=task,
            k=(100 if task == "regression" else 10),
            T=100
        )
        preds.append(pred)

    preds = torch.stack(preds).squeeze()

    """
    Inverse transform predictions and compute metrics
    """
    if task == "regression":
        preds_np  = y_scaler.inverse_transform(preds.view(-1, 1).cpu().numpy()).squeeze()
        truth_np  = y_scaler.inverse_transform(y_test_t.view(-1, 1).cpu().numpy()).squeeze()
    else:
        preds_np  = preds.cpu().numpy()
        truth_np  = y_test_t.cpu().numpy()

    metrics = compute_metrics(preds_np, truth_np, task)

    """
    Save JSON
    """
    out_dir = os.path.join("output", f'{dataset}')
    os.makedirs(out_dir, exist_ok=True)

    model_path = os.path.join(out_dir, "model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    json_path = os.path.join(out_dir, "metrics.json")
    os.makedirs("output", exist_ok=True)

    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Metrics saved to {json_path}")
    print(json.dumps({k: v for k, v in metrics.items() if k != "per_class_report"}, indent=2))

    json_path = os.path.join(out_dir, "standard.json")
    os.makedirs("output", exist_ok=True)

    with open(json_path, "w") as f:
        json.dump(standard, f, indent=2)