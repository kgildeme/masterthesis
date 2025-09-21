from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parents[1]))

import os
import glob
import torch

from cids.util import misc_funcs as misc


if __name__ == "__main__":

    directories = glob.glob(os.path.join(misc.root(), "models/03_semisupervision/finetuned/MLPAE-201/SSL-FULL-ITER-201-2/deepsad--*"))

    print(len(glob.glob(os.path.join(misc.root(), "models/03_semisupervision/finetuned/MLPAE-201/SSL-FULL-ITER-201-2/deepsad--*/*/ckpt.pt"))))
    cs = {}

    print("Collect c")
    for seed in range(1, 11):
        dir_seeds = glob.glob(os.path.join(misc.root(), f"models/03_semisupervision/finetuned/MLPAE-201/SSL-FULL-ITER-201-2/deepsad--*/seed{seed}/ckpt.pt"))
        ckpt = torch.load(dir_seeds[0])
        cs[f"seed{seed}"] = ckpt["c"]
        del ckpt
    
    for dir in directories:
        print(f"Copy c in {dir}")
        for seed in range(1, 11):
            ckpts = glob.glob(os.path.join(dir, f"seed{seed}", "[!c]*"))  
            for ckpt in ckpts:
                model = torch.load(ckpt)
                model["c"] = cs[f"seed{seed}"]
                torch.save(model, ckpt)
    

