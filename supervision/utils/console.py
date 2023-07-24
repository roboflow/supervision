from tqdm import tqdm


def get_dataset_loading_tqdm_iterable(iterable):
    return tqdm(iterable, desc="Loading Dataset: ")
