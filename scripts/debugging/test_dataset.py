from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parents[2]))
import os
import logging
import time

import numpy as np
import torch

from cids.data import OpTCDataset, worker_init
from cids.util import misc_funcs as misc
from util import setup_logger

def manuel_iterator(iterable):
    iterator = iter(iterable)
    while True:
        try:
            input("Press Enter to view the next item (or Ctrl+C to exit)...")
            out = next(iterator)
            logger.info(out.dtype)
            logger.info(out.shape)
            logger.info(out.sum())
        except StopIteration:
            logger.info("End of the iterable reached.")
            break
        except KeyboardInterrupt:
            logger.info("\nIteration stopped manually.")
            break

def automatic_process_analyis(iterable):
    steps = 0    
    for w in iterable:
        steps += 1
         #process = w["new_process"]
        #if process.sum() > 1:
        #    raise Exception("Something went terribly wrong")
        #if process.sum() == 1 and process.iloc[0] != 1:
         #   logger.error(process.to_json(indent=2))
         #   raise Exception("Something went terribly wrong")
    logger.info(f"Iterated over dataset with {steps} step")
        
def timing(iterable):
    start = time.time()
    iteration_mean = 0
    iteration_start = time.time_ns()
    i = 0
    logger.debug(f"Load Dataloader")
    # dl = torch.utils.data.DataLoader(dataset=iterable, num_workers=2, worker_init_fn=worker_init, batch_size=1)
    for w in iterable:
        i += 1
        iteration_mean += (time.time_ns() - iteration_start - iteration_mean) / i
        iteration_start = time.time_ns()

    logger.info(f"Mean Timer per Iteration: {iteration_mean / 1e6}ms")
    logger.info(f"Total Time: {(time.time() - start)*1e3}ms")

def multi_worker():
    for b in [8, 16, 64, 256, 512]:
        for workers in [0, 2, 4, 8, 16]:
            dataset = OpTCDataset("hids-v5_201_train", parts=16, window_size=10, shuffle=True)
            if workers == 0:
                dl = torch.utils.data.DataLoader(dataset=dataset, batch_size=b)
            else:
                dl = torch.utils.data.DataLoader(dataset, num_workers=workers, worker_init_fn=worker_init, batch_size=b)
            # manuel_iterator(dl)
            logger.info(f"Start timing for n_workers = {workers} and batch = {b}")
            try:
                timing(dl)
            except Exception as e:
                logger.exception("An unexpected exception occured in timing")
                raise e

if __name__ == "__main__":
    logger = setup_logger(os.path.join(misc.root(), "logs/dataset/test.log"), level=logging.INFO)
    logger.info("Start iterationg through dataset")
    
    dataset_gen = OpTCDataset("cids-v5_201_train-ff", parts=16, window_size=30, shuffle=True)
    dl = torch.utils.data.DataLoader(dataset_gen, batch_size=128)
    automatic_process_analyis(dl)
    # manuel_iterator(dataset_gen)
    # timing(dataset_gen)
    # multi_worker()
