import itertools
from torch.utils.data import Dataset, DataLoader

from utils import d_types_methods

# Torch Dataset
class PotteryDataset(Dataset):
    def __init__(self, X_list, y):
        """
        X_list: list of tensors, each [N, d] (can be 1 or more feature sets)
        y: tensor of targets [N] or [N, t] (t = number of targets)
        """
        self.X_list = X_list
        self.y = y

    def __len__(self):
        # Return number of samples in dataset
        return self.y.shape[0]

    def __getitem__(self, idx):
        # Return one sample (features and target) at position idx
        return [X[idx] for X in self.X_list], self.y[idx]

    def __dim__(self):
        X_dim = tuple(X.shape[1] for X in self.X_list)
        y_dim = self.y.shape[1] if len(self.y.shape) > 1 else 1
        return X_dim, y_dim


feature_types = d_types_methods["text"] + d_types_methods["image"]
feature_type_combos = tuple(itertools.product(d_types_methods["text"], d_types_methods["image"]))

# Create All Pottery Datasets
def create_pottery_datasets(X, y):
    datasets = {}

    for subset in X.keys():
        datasets[subset] = {}
        for ft in feature_types:
            datasets[subset][ft] = PotteryDataset([X[subset][ft]], y[subset])

        for ft_txt, ft_img in feature_type_combos:
            ft = f"{ft_txt} + {ft_img}"
            datasets[subset][ft] = PotteryDataset([X[subset][ft_txt], X[subset][ft_img]], y[subset])

    return datasets

# Create All DataLoaders
def create_pottery_dataloaders(datasets, batch_size=64):
    loaders = {}
    for subset in datasets.keys():
        shuffle = True if subset == "train" else False
        loaders[subset] = {}
        for ft, dataset in datasets[subset].items():
            loaders[subset][ft] = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loaders