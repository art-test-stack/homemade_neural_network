import numpy as np

def save_data_in_folder(address, data_subset, X, Y, labels):
    np.save(address / f"X_{data_subset}.npy", X)
    np.save(address / f"Y_{data_subset}.npy", Y)
    np.save(address / f"labels_{data_subset}.npy", labels)


def load_subsets(address, data_subset):
    X = np.load(address / f"X_{data_subset}.npy")
    Y = np.load(address / f"Y_{data_subset}.npy")
    labels = np.load(address / f"labels_{data_subset}.npy")

    return (X, Y, labels)