from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import torch
import torchvision.transforms as transforms
from torch.nn.functional import normalize


class DataTransform(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        xi = self.transform(sample)
        xj = self.transform(sample)
        return xi, xj


def augmentation_transformers():
    return transforms.Compose([transforms.RandomResizedCrop([8, 14]),
                               transforms.RandomHorizontalFlip()])


class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """

    def __init__(self, tensors, transform=None, normalise_data=True):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform
        self.normalise_data = normalise_data

    def __getitem__(self, index):
        x = self.tensors[0][index]
        if self.normalise_data:
            try:
                x = normalize(x.view(24, -1), dim=0).view(x.size())
            except RuntimeError:
                x = normalize(x.view(-1, ), dim=0).view(x.size())

        if self.transform:
            x = self.transform(x)

        y = self.tensors[1][index]

        return x, y

    def __len__(self):
        return self.tensors[0].size(0)


def torch_loader(X, y, batch_size=10, normalise_data=True, shuffle=True, seed=0, split=True, augmentation=False):
    transformers = DataTransform(augmentation_transformers()) if augmentation else None
    if split:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=seed, stratify=y)
        train_dataset = CustomTensorDataset([torch.Tensor(X_train), torch.tensor(y_train)], transformers, normalise_data)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        test_dataset = CustomTensorDataset([torch.Tensor(X_test), torch.tensor(y_test)], transformers, normalise_data)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
        return train_dataloader, test_dataloader
    else:
        train_dataset = CustomTensorDataset([torch.Tensor(X), torch.tensor(y)], transformers, normalise_data)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        return train_dataloader


def torch_unlabelled_loader(X, batch_size=10, shuffle=True, augmentation=False):
    transformers = DataTransform(augmentation_transformers()) if augmentation else None
    train_dataset = CustomTensorDataset([torch.Tensor(X), torch.ones(X.shape[0])], transformers)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
    return train_dataloader


__all__ = ['torch_loader', 'torch_unlabelled_loader']
