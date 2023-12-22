import os
import numpy as np
import torch as tc
import random
import torch

# def set_random_seed(seed):		
# 	random .seed(seed)
# 	np     .random.seed(seed)
# 	tc     .manual_seed(seed)
# 	tc     .cuda.manual_seed(seed)
# 	tc     .cuda.manual_seed_all(seed)
# 	tc     .backends.cudnn.deterministic = True
# 	tc     .backends.cudnn.benchmark     = False


def set_random_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)