import torch
import copy
import torch.nn as nn

def get_k_nearest_neighbors(x_query, X, y, k, projection=None):
    if projection is None:
        distances = torch.norm(X - x_query.unsqueeze(0), dim=1)
        knn_idx = torch.topk(distances, k, largest=False).indices
        return X[knn_idx], y[knn_idx], distances[knn_idx]
    
    X_latent = projection.transform(X.cpu().numpy())
    x_query_latent = projection.transform(x_query.cpu().numpy().reshape(1, -1)).squeeze()
    X_latent = torch.from_numpy(X_latent).float().to(x_query.device)
    x_query_latent = torch.from_numpy(x_query_latent).float().to(x_query.device)

    return get_k_nearest_neighbors(x_query_latent, X_latent, y, k)

def ttlt_prediction(
    x, X, y,
    *,
    model=None,
    projection=None,
    task="classification",
    k=None, T=100,
    lr=1e-5,
    weight_decay=1e-5
):
    assert model is not None, "Model must be provided for TTLT."
    assert task in ["regression", "classification"], "Task must be either 'regression' or 'classification'."

    if k is None:
        k = 100 if task == "regression" else 10

    if projection is not None:
        x_star = projection.transform(x.cpu().numpy().reshape(1, -1)).squeeze()
        x_star = torch.from_numpy(x_star).float().to(x.device)
    else:
        x_star = x

    X_neighbors, y_neighbors, dists = get_k_nearest_neighbors(x_star, X, y, k, projection=projection)

    eps = 1e-8
    w = 1.0 / (dists + eps)
    w = w / w.sum()

    f_local = copy.deepcopy(model)
    f_local.train()
    optimizer = torch.optim.Adam(f_local.parameters(), lr=lr, weight_decay=weight_decay)

    if task == "regression":
        criterion = nn.MSELoss(reduction='none')
    else:
        criterion = nn.CrossEntropyLoss(reduction='none')

    for _ in range(T):
        optimizer.zero_grad()
        preds = f_local(X_neighbors)
        loss_vec = criterion(preds, y_neighbors)
        loss = (w * loss_vec).sum()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        out = f_local(x.unsqueeze(0))

    if task == "classification":
        return out.argmax(dim=1)
    return out