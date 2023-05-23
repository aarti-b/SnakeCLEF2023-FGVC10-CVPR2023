from torch.utils.data import WeightedRandomSampler
from torch import where, tensor
from torch.utils.data import Dataset


__all__ = ['ClassUniformSampler', 'oversample_dataset', 'augment_dataset']


class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]
        return data, label

def ClassUniformSampler(dataset):

    label_col = dataset.label_col

    # create sampling weights
    class_counts = dataset.df[label_col].value_counts()
    desired_imgs_per_class = len(dataset) // len(class_counts)
    class_weight = (desired_imgs_per_class / class_counts).reset_index()
    class_weight.columns = [label_col, 'weight']
    sampling_weights = dataset.df.merge(
        class_weight, 'left', on=[label_col])['weight'].values
    assert len(sampling_weights) == len(dataset)

    # create weighted random sampler
    sampler = WeightedRandomSampler(
        sampling_weights, num_samples=len(dataset), replacement=True)

    return sampler


def oversample_dataset(dataset, label_col, desired_samples):
    # Count the number of samples in each class
    class_counts = tensor(dataset.df[label_col].value_counts())

    # Find the minority class labels
    minority_classes = where(class_counts < desired_samples)[0]

    # Oversample the minority classes
    oversampled_data = []
    oversampled_labels = []

    for i in range(len(dataset)):
        data, label = dataset[i]
        oversampled_data.append(data)
        oversampled_labels.append(label)

        if label in minority_classes:
            oversampled_data.append(data)
            oversampled_labels.append(label)

    # Create a new dataset with the oversampled data
    oversampled_dataset = CustomDataset(oversampled_data, oversampled_labels)
    return oversampled_dataset


def augment_dataset(dataset, transforms):

    augmented_data = []
    augmented_labels = []
    for i in range(len(dataset)):
        data, label = dataset[i]
        for transform in transforms:
            augmented_data.append(transform(data))
            augmented_labels.append(label)

    # Create a new dataset with the augmented data
    augmented_dataset = CustomDataset(augmented_data, augmented_labels)
    return augmented_dataset
