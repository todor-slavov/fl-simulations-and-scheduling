import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Normalize, Compose
from torch.utils.data import random_split, DataLoader, Subset
from time import time
from logging import DEBUG, INFO, WARNING
from flwr.common.logger import log

import numpy as np

import matplotlib.pyplot as plt

def get_mnist(data_path: str = './data'):
    tr = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    trainset = MNIST(data_path, train=True, download=True, transform=tr)
    testset = MNIST(data_path, train=False, download=True, transform=tr)
    return trainset, testset


def prepare_dataset(num_partitions: int, batch_size: int, val_ratio: float = 0.1):
    trainset, testset = get_mnist()
    num_images = len(trainset) // num_partitions
    partition_len = [num_images] * num_partitions
    trainsets = random_split(trainset, partition_len, torch.Generator().manual_seed(2024))

    trainloaders = []
    valloaders = []
    for tr_set in trainsets:
        num_total = len(tr_set)
        num_val = int(val_ratio * num_total)
        num_train = num_total - num_val

        for_train, for_val = random_split(tr_set, [num_train, num_val], torch.Generator().manual_seed(2024))

        trainloaders.append(DataLoader(for_train, batch_size=batch_size, shuffle=True, num_workers=2))
        valloaders.append(DataLoader(for_val, batch_size=batch_size, shuffle=True, num_workers=2))

    testloader = DataLoader(testset, batch_size=120)

    return trainloaders, valloaders, testloader

def prepare_dataset_nonIID(num_partitions: int, batch_size: int, val_ratio: float = 0.1):
    trainset, testset = get_mnist()

    targets = np.array(trainset.targets)
    sorted_indices = np.argsort(targets)
    sorted_targets = targets[sorted_indices]

    class_indices = []
    for class_label in range(10):
        class_indices.append(sorted_indices[sorted_targets == class_label])

    shards = []
    num_shards_per_class = 2
    for indices in class_indices:
        shard_size = len(indices) // num_shards_per_class
        for i in range(num_shards_per_class):
            shards.append(indices[i * shard_size: (i + 1) * shard_size])
    
    np.random.seed(2024)
    np.random.shuffle(shards)
    
    client_indices = []
    used_labels = set()
    shard_label_map = {i: set(targets[shard]) for i, shard in enumerate(shards)}

    for i in range(0, num_partitions, 2):
        shard_pair = []
        for shard_idx, labels in shard_label_map.items():
            if not labels & used_labels:
                shard_pair.append(shard_idx)
                used_labels.update(labels)
            if len(shard_pair) == 2:
                break
        
        if len(shard_pair) != 2:
            raise ValueError("Not enough unique shards to assign to each client pair.")

        client_shards = [shards[shard_pair[0]], shards[shard_pair[1]]]
        client_data = np.concatenate(client_shards)
        client_indices.append(client_data)
        client_indices.append(client_data)

    trainloaders = []
    valloaders = []
    for indices in client_indices:
        client_subset = Subset(trainset, indices)
        
        num_total = len(client_subset)
        num_val = int(val_ratio * num_total)
        num_train = num_total - num_val

        for_train, for_val = random_split(client_subset, [num_train, num_val], torch.Generator().manual_seed(2024))
        train_labels = [label for _, label in for_train]
        print(set(train_labels))

        trainloaders.append(DataLoader(for_train, batch_size=batch_size, shuffle=True, num_workers=2))
        valloaders.append(DataLoader(for_val, batch_size=batch_size, shuffle=True, num_workers=2))

    testloader = DataLoader(testset, batch_size=120)

    return trainloaders, valloaders, testloader


def prepare_dataset_nonIID_varying_sizes(num_partitions: int, batch_size: int, val_ratio: float = 0.1):
    trainset, testset = get_mnist()
    
    total_images = len(trainset)
    mean_images = total_images // num_partitions
    stddev_images = mean_images // 2
    
    dataset_seed = 123456
    permutation_seed = int(time())
    
    log(INFO, f"Using seed {permutation_seed} for client dataset permutation")
    log(INFO, f"Using seed {dataset_seed} for dataset partition")
    
    # Set random seed for partition sizes
    np.random.seed(dataset_seed)
    sizes = np.random.normal(mean_images, stddev_images, num_partitions).astype(int)
    sizes = np.clip(sizes, a_min=100, a_max=None)  # Ensure at least 100 images per client
    sizes = sizes * total_images // sizes.sum()  # Normalize to ensure the total sum is the dataset size

    # Set random seed for permutation of dataset
    np.random.seed(permutation_seed)
    indices = np.random.permutation(total_images)
    
    # Split indices by class
    targets = np.array(trainset.targets)
    class_indices = [indices[targets[indices] == i] for i in range(10)]
    
    for i in range(10):
        class_indices[i] = np.random.permutation(class_indices[i])
    
    client_indices = []
    np.random.seed(dataset_seed)
    for size in sizes:
        client_subset = []
        remaining_size = size
        
        while remaining_size > 0:
            # Select a random class
            class_label = np.random.randint(0, 10)
            # Ensure there are samples left in this class
            if len(class_indices[class_label]) == 0:
                continue
            # Randomly decide how many samples to take from this class (up to remaining size)
            num_samples = np.random.randint(1, min(len(class_indices[class_label]), remaining_size) + 1)
            # Take the samples
            client_subset.extend(class_indices[class_label][:num_samples])
            # Remove these samples from the class indices
            class_indices[class_label] = class_indices[class_label][num_samples:]
            remaining_size -= num_samples
            
        client_indices.append(np.array(client_subset))
    
    trainloaders = []
    valloaders = []
    for indices in client_indices:
        client_subset = Subset(trainset, indices)
        
        num_total = len(client_subset)
        num_val = int(val_ratio * num_total)
        num_train = num_total - num_val

        # Set seed for splitting into training and validation
        np.random.seed(dataset_seed)
        for_train, for_val = random_split(client_subset, [num_train, num_val], torch.Generator().manual_seed(dataset_seed))
        
        trainloaders.append(DataLoader(for_train, batch_size=batch_size, shuffle=True, num_workers=2))
        valloaders.append(DataLoader(for_val, batch_size=batch_size, shuffle=True, num_workers=2))

    # print(client_indices)
    # print("huj")
    # exit()

    # for client in range(num_clients):
    #     pass 

    data_amounts = [0 for _ in range(num_partitions)]

    log(INFO, "Client dataset sizes:")
    for i, trainloader in enumerate(trainloaders):
        labels = [label for _, label in trainloader.dataset]
        unique, counts = np.unique(labels, return_counts=True)
        log(INFO, f"Client {i} has {len(trainloader.dataset)} training samples with label distribution: {dict(zip(unique, counts))}")
        for (u, c) in zip(unique, counts):
            data_amounts[u] += c

    test_indices = []
    test_targets = np.array(testset.targets)
    for _ in range(total_images):
        class_label = np.random.randint(0, 10)
        index = np.random.choice(np.where(test_targets == class_label)[0])
        test_indices.append(index)
    testloader = DataLoader(Subset(testset, test_indices), batch_size=120, shuffle=True, num_workers=2)

    np.random.seed(dataset_seed)
    remove_probabilities = [1 for _ in range(num_partitions)]
    np.random.shuffle(remove_probabilities)
    np.random.seed(permutation_seed)
    
    indices_to_keep = []

    for i in range(10):
        class_indices = np.where(test_targets == i)[0]
        num_remove = int(len(class_indices) * (1 - remove_probabilities[i]))
        indices_to_keep_now = np.random.choice(class_indices, size=len(class_indices) - num_remove, replace=False)
        indices_to_keep.extend(indices_to_keep_now)

    testloader.dataset.indices = indices_to_keep

    labels = [label for _, label in testloader.dataset]
    unique, counts = np.unique(labels, return_counts=True)
    log(INFO, f"Testloader has {len(testloader)} test samples with label distribution: {dict(zip(unique, counts))}")

    return trainloaders, valloaders, testloader

def extract_labels(trainloaders):
    all_labels = []
    for loader in trainloaders:
        labels = []
        for batch in loader:
            _, targets = batch
            labels.extend(targets.tolist())
        all_labels.append(labels)
    return all_labels

def plot_class_distributions(trainloaders):
    all_labels = extract_labels(trainloaders)
    num_clients = len(all_labels)
    target_labels = sorted(set([label for labels in all_labels for label in labels]))

    print(all_labels)

    fig, ax = plt.subplots(figsize=(15, 8))
    width = 0.4  # Width of each bar

    colors = ['deepskyblue', 'orange', 'greenyellow', 'olivedrab', 'plum', 'blueviolet', 'pink', 'lavender', 'navy', 'cyan']

    bottom = np.zeros(num_clients)  # Bottom position for each bar segment
    for label in target_labels:
        label_counts = [client_labels.count(label) for client_labels in all_labels]
        
        ax.bar(range(num_clients), label_counts, width, bottom=bottom, label=f'Label {label}', color=colors[label])
        bottom += label_counts  # Update the bottom position for the next label segment
    
    # ax.set_xlabel('Clients', fontsize=18)
    # ax.set_ylabel('Number of Samples', fontsize=18)
    # ax.set_title('Class Distribution per Client on the CIFAR-10 Dataset', fontsize=18)
    # ax.legend(loc='upper right', fontsize=14)
    # ax.set_xticks(range(num_clients))
    # ax.set_xticklabels([f'Client {i}' for i in range(num_clients)], fontsize=15)
    # ax.yaxis.set_tick_params(labelsize=14)
    # plt.show()
    ax.set_xlabel('Clients', fontsize=24)
    ax.set_ylabel('Number of Samples', fontsize=24)
    # ax.set_title('Class Distribution per Client', fontsize=18)
    
    # Create legend and set its properties
    legend = ax.legend(loc='upper right', fontsize=18, title_fontsize='13')
    plt.setp(legend.get_texts())  # Set font weight for legend texts
    
    ax.set_xticks(range(num_clients))
    ax.set_xticklabels([f'Client {i}' for i in range(num_clients)], fontsize=13)
    ax.yaxis.set_tick_params(labelsize=14)  # Increase y-axis tick label font size and set semibold font

    plt.tick_params(axis='y', labelsize=20)  # Change 14 to your desired size
    plt.tick_params(axis='x', labelsize=20)  # Change 14 to your desired size

    plt.savefig('Class Distribution per Client.pdf', format='pdf')

    plt.show()


num_clients = 10
batch_size = 64
trainloaders, valloaders, testloader = prepare_dataset_nonIID_varying_sizes(num_clients, batch_size)

plot_class_distributions(trainloaders)
# plot_class_distributions(testloader)