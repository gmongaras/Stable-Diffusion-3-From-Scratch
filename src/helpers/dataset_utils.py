import numpy as np
import torch
import random
import os
from tqdm import tqdm
from torch.utils.data import DataLoader, Sampler
from datasets import load_dataset
import math
from collections import defaultdict
from multiprocessing import Pool, cpu_count
import concurrent.futures



# def process_chunk(chunk):
#     """Processes a chunk of dataset indices and groups them by bucket_size."""
#     local_bucket_dict = defaultdict(list)
#     for idx, sample in chunk:
#         local_bucket_dict[sample['bucket_size']].append(idx)
#     return dict(local_bucket_dict)

# def merge_dicts(dicts):
#     """Merges multiple defaultdicts into one."""
#     merged = defaultdict(list)
#     for d in dicts:
#         for key, values in d.items():
#             merged[key].extend(values)
#     return merged

# def load_indices(path, dataset, chunk_size=500000):
#     """Parallelized bucket indexing function."""
#     if not os.path.exists(path):
#         print("Generating bucket indices...")

#         # Split dataset into chunks
#         chunks = [(i, sample) for i, sample in enumerate(dataset)]
#         chunked_data = [chunks[i:i + chunk_size] for i in range(0, len(chunks), chunk_size)]

#         # Use multiprocessing Pool
#         with Pool(processes=cpu_count()) as pool:
#             results = list(tqdm(pool.imap(process_chunk, chunked_data), total=len(chunked_data)))

#         # Merge results
#         bucket_dict = merge_dicts(results)

#         # Save as NumPy file
#         np.save(path, dict(bucket_dict))

#         print(f"Saved bucket indices to {path}")

def process_part(dataset, indices):
    bucket_dict_ = defaultdict(list)
    dataset = dataset[indices]["bucket_size"]
    for bucket_size, idx in zip(dataset, indices):
        bucket_dict_[bucket_size].append(idx)
    return bucket_dict_


def load_indices(path, dataset, n_proc=32):
    # If precomputed indices are not available, generate them
    if not os.path.exists(path):
        from collections import defaultdict

        print("Generating bucket indices...")


        
        """
        from collections import defaultdict
        from concurrent.futures import ThreadPoolExecutor
        from tqdm import tqdm

        bucket_dict = defaultdict(list)

        # Function to process each sample
        def process_sample(args):
            idx, sample = args
            return sample['bucket_size'], idx

        # Threaded execution
        with ThreadPoolExecutor() as executor:
            results = list(tqdm(executor.map(process_sample, enumerate(dataset)), total=len(dataset)))

        # Populate bucket_dict from threaded results
        for bucket_size, idx in results:
            bucket_dict[bucket_size].append(idx)
        """


        # Group indices
        bucket_dict = defaultdict(list)
        step = 10_000
        indices = [range(i, min(i+step, len(dataset))) for i in range(0, len(dataset), step)]

        # Multiproc
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_proc) as executor:
            futures = {executor.submit(process_part, dataset, idxs): idxs for idxs in indices}
            
            # Accumulate outputs
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(indices)):
                r = future.result()
                for k,v in r.items():
                    bucket_dict[k] += v



        # # Group indices by bucket_size
        # bucket_dict = defaultdict(list)
        # for idx, sample in tqdm(enumerate(dataset), total=len(dataset)):
        #     bucket_dict[sample['bucket_size']].append(idx)

        # # Save indices to JSON
        # with open(path, "w") as f:
        #     json.dump(bucket_dict, f)

        np.save(path, dict(bucket_dict))  # Save as a NumPy file

        print(f"Saved bucket indices to {path}")


# Random bucket sampler
class RandomBucketSampler(Sampler):
    def __init__(self, bucket_path, dataset, batch_size):
        # Load precomputed indices
        self.bucket_list = np.load(bucket_path, allow_pickle=True).item()

        # Remove buckets smaller than the batch size
        removed = [i for i,j in self.bucket_list.items() if len(j) <= batch_size]
        print(f"Removing the following buckets as they are too small: {removed}")
        self.bucket_list = {i:j for i,j in self.bucket_list.items() if len(j) > batch_size}
        
        self.bucket_list = [(bucket_size, indices) for bucket_size, indices in self.bucket_list.items()]
        self.batch_size = batch_size
        self.bucket_order = list(range(len(self.bucket_list)))

        # Probability distribution for bucket sizes
        total = sum(len(i[1]) for i in self.bucket_list)
        self.probs = np.array([len(i[1])/total for i in self.bucket_list])

        # First few buckets will always be the largest to allocate proper GPU memory on first pass
        self.first_n = 0
        self.first_size = "x".join([str(i) for i in np.array([i[0].split("x") for i in self.bucket_list]).astype(int).max(0)])
        self.first_idx = [i[0] for i in self.bucket_list].index(self.first_size)

    def __iter__(self):
        while True:
            # Get a random bucket based on the probability distribution
            if self.first_n <= 0:
                bucket_idx = np.random.choice(self.bucket_order, p=self.probs)
            else:
                bucket_idx = self.first_idx
                self.first_n -= 1


            # Get a random batch from the selected bucket
            _, indices = self.bucket_list[bucket_idx]
            # # Not enough samples for the batch
            # if self.batch_size > len(indices):
            #     return indices
            batch_indices = random.sample(indices, self.batch_size)
            yield batch_indices

    def __len__(self):
        return sum(len(indices) // self.batch_size for _, indices in self.bucket_list)

# Dataset Wrapper
class HuggingFaceDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, idx):
        return self.dataset[idx]  # Preprocessing here if needed

    def __len__(self):
        return len(self.dataset)
