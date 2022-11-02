import os

if __name__ == "__main__":
    print(os.environ.get("SLURM_JOB_CPUS_PER_NODE"))
