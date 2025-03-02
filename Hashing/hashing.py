import os
import numpy as np
import torch
import random
from sklearn.decomposition import PCA

# ----------------------------------------------------------------------------------------------------------------
def flatten_and_stack_descriptors(descriptor_folder, save_path):
    descriptor_files = sorted([f for f in os.listdir(descriptor_folder) if f.endswith('.npy')])
    flattened_descriptors = []

    for file in descriptor_files:
        file_path = os.path.join(descriptor_folder, file)
        descriptor = np.load(file_path)

        descriptor_vector = descriptor.flatten()
        descriptor_tensor = torch.FloatTensor(descriptor_vector)

        flattened_descriptors.append(descriptor_tensor)

    combined_tensor = torch.stack(flattened_descriptors, dim=0)
    print(f"Combined Descriptor Tensor Shape: {combined_tensor.shape}")
    combined_file = os.path.join(save_path, "combined_tensor.npy")
    np.save(combined_file, combined_tensor)
    return combined_tensor


# ----------------------------------------------------------------------------------------------------------------
def reduce_dimensionality(tensor, n_components, save_path):
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(tensor.numpy())
    reduced_file = os.path.join(save_path, f"reduced_tensor_{n_components}.npy")
    np.save(reduced_file, reduced)
    return torch.FloatTensor(reduced)


# ----------------------------------------------------------------------------------------------------------------
def hashed_values(data, no_of_hash, feature_size, random_seed=42):
    Wl = torch.FloatTensor(no_of_hash, feature_size).normal_(0, 1)  # Random projection matrix
    Bin_values = torch.matmul(data, Wl.T)  # Project data into hash space
    return Bin_values


def partition(list_bin_width, Bin_values, no_of_hash, random_seed=42):
    """
    Partition hashed values into clusters/buckets based on bin widths.
    """
    summary_dict = {}
    for bin_width in list_bin_width:
        bias = torch.tensor([random.uniform(-bin_width, bin_width) for _ in range(no_of_hash)])
        temp = torch.floor((1 / bin_width) * (Bin_values + bias))
        cluster, _ = torch.max(temp, dim=1)
        dict_hash_indices = {}
        no_nodes = Bin_values.shape[0]
        for i in range(no_nodes):
            dict_hash_indices[i] = int(cluster[i])
        summary_dict[bin_width] = dict_hash_indices
    return summary_dict


# ----------------------------------------------------------------------------------------------------------------
def process_flattened_descriptors_with_hashing(combined_tensor, bin_width, output_folder):
    """
    Normalize data, compute hashed values, partition into bins, and save the result.
    """
    no_of_hash = 100
    list_bin_width = [bin_width]

    # Normalize the tensor
    combined_tensor = torch.nn.functional.normalize(combined_tensor, p=2, dim=1)

    # Generate hashed values
    bin_values = hashed_values(combined_tensor, no_of_hash, feature_size=combined_tensor.shape[1])

    # Partition into bins
    hash_clusters = partition(list_bin_width, bin_values, no_of_hash)

    # Extract unique cluster values
    unique_clusters = set()
    for bin_width, clusters in hash_clusters.items():
        unique_clusters.update(clusters.values())
    print("Unique Cluster Values:", len(unique_clusters))

    # Save bucket data
    output_file = os.path.join(output_folder, f"hash_buckets_flattened_{bin_width}.txt")
    with open(output_file, 'w') as f:
        for bin_width, clusters in hash_clusters.items():
            for node, cluster in clusters.items():
                f.write(f"{node}: {cluster}\n")

    print(f"Hashing and bucketing complete. Results saved to {output_file}")
    return hash_clusters


# ----------------------------------------------------------------------------------------------------------------
def load_tensor(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    loaded_array = np.load(file_path)
    combined_tensor = torch.from_numpy(loaded_array)
    print(f"Loaded Tensor Shape: {combined_tensor.shape}")
    return combined_tensor


# ----------------------------------------------------------------------------------------------------------------
# MAIN
if __name__ == "__main__":
    DESCRIPTOR_FOLDER = "./descriptors_output" #Folder Containing SOAP Descriptors
    OUTPUT_FOLDER = "./hash_buckets_output"
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    n_components = 50 #Number of PCA Components
    bin_width = 0.0005 #Width of HASH Bucket

    # Step 1: Flatten and stack descriptors (if not already done)
    combined_tensor_file = os.path.join(OUTPUT_FOLDER, "combined_tensor.npy")
    if not os.path.exists(combined_tensor_file):
        print("Flattening and stacking descriptors...")
        combined_tensor = flatten_and_stack_descriptors(DESCRIPTOR_FOLDER, OUTPUT_FOLDER)
    else:
        print("Loading combined tensor from file...")
        combined_tensor = load_tensor(combined_tensor_file)

    # Step 2: Reduce dimensionality (if not already done)
    reduced_tensor_file = os.path.join(OUTPUT_FOLDER, f"reduced_tensor_{n_components}.npy")
    if not os.path.exists(reduced_tensor_file):
        print("Reducing dimensionality with PCA...")
        reduced_tensor = reduce_dimensionality(combined_tensor, n_components, OUTPUT_FOLDER)
    else:
        print("Loading reduced tensor from file...")
        reduced_tensor = load_tensor(reduced_tensor_file)

    # Step 3: Process with hashing and bucketing
    hash_buckets_file = os.path.join(OUTPUT_FOLDER, f"hash_buckets_flattened_{bin_width}.txt")
    if not os.path.exists(hash_buckets_file):
        print("Performing hashing and bucketing...")
        hash_clusters = process_flattened_descriptors_with_hashing(reduced_tensor, bin_width, OUTPUT_FOLDER)
    else:
        print("Hashing and bucketing already completed. Skipping this step.")
