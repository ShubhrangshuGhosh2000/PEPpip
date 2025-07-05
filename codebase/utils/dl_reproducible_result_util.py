# Seed value
seed_value = 456

# Set 'PYTHONHASHSEED' environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED'] = str(seed_value)

# Set 'python' built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)

# Set 'numpy' pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)

# Set 'pytorch' related seed values (Refer: https://pytorch.org/docs/stable/notes/randomness.html)
import torch
torch.manual_seed(seed_value)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

# Disabling the benchmarking feature with torch.backends.cudnn.benchmark = False causes
# cuDNN to deterministically select an algorithm, possibly at the cost of reduced performance.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Avoiding nondeterministic algorithms
# torch.use_deterministic_algorithms(True)
