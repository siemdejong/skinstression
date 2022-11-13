import os
import time
print(os.environ["SLURM_NODEID"])
time.sleep(60)