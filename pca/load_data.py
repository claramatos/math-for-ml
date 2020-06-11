def load_mnist():
    mnist_path = "mnist-original.mat"
    from scipy.io import loadmat
    mnist_raw = loadmat(mnist_path)
    mnist = {
        "data": mnist_raw["data"].T,
        "target": mnist_raw["label"][0],
        "COL_NAMES": ["label", "data"],
        "DESCR": "mldata.org dataset: mnist-original",
    }
    return mnist


if __name__ == '__main__':
    import numpy as np

    x = np.array([3, 1, 2])
    print(np.argsort(x))
    print(x[np.argsort(x)])
