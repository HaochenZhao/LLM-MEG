import torch

# Example tensor
N, d = 5142, 10  # N is the number of samples, d is the number of features
data = torch.randn(N, d)

# Shuffle the indices
indices = torch.randperm(N)

# Split into 5 folds
n_splits = 5
fold_size = N // n_splits

for fold in range(n_splits):
    # Calculate test indices for this fold
    test_indices = indices[fold * fold_size:(fold + 1) * fold_size]
    
    # Calculate train indices (all indices except the current test indices)
    train_indices = torch.cat([indices[:fold * fold_size], indices[(fold + 1) * fold_size:]])

    # Split the data into train and test sets for this fold
    train_data = data[train_indices]
    test_data = data[test_indices]

    print(f"Fold {fold + 1}")
    print(f"Train data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}\n")