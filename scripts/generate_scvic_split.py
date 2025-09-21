from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parents[1]))
import logging
import os

from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np

from cids.data import SCVICCIDSDataset
from cids.util import misc_funcs as misc
import util


if __name__ == "__main__":
    os.makedirs(os.path.join(misc.root(), "logs/dataset/scvic/"), exist_ok=True)
    logger = util.setup_logger(os.path.join(misc.root(), "logs/dataset/scvic/split.log"), level=logging.DEBUG)

    ds = SCVICCIDSDataset(misc.data_raw(scvic=True))
    logger.info("Loaded data, generate split")
    labels = [ds[i][3] for i in range(len(ds))]

    train, test = next(StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(np.zeros(len(labels)), labels))

    os.makedirs(os.path.join(misc.data(), "scvic"), exist_ok=True)
    np.savetxt(os.path.join(misc.data(), "scvic/train_indices.txt"), train, fmt="%d")
    np.savetxt(os.path.join(misc.data(), "scvic/test_indices.txt"), test, fmt="%d")

    logger.info("DONE")
    