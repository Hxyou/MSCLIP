import argparse
import os
import subprocess

cfg_files_dataset = {
    "imagenet": "experiments/dataset/imagenet.yaml",
    }

def parse_args():
    """Parse arguments: a list of dataset names and a model"""
    parser = argparse.ArgumentParser(
        description='Zeroshot Eval')

    parser.add_argument('--ds',
                        help='Evaluation dataset configure file name.',
                        type=str)

    parser.add_argument('--model',
                        required=True,
                        help='Evaluation model configure file name',
                        type=str)

    parser.add_argument('--save-feature',
                        help='Flag to save feature or not',
                        default=False,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args_batch = parser.parse_args()

    return args_batch


def run_jobs():
    """
    Run a list of zeroshot evaluation jobs.
    """
    args = parse_args()
    
    if args.ds is None:
        datasets = list(cfg_files_dataset.keys())
    else:
        datasets = args.ds.split(",")
    # Check dataset availability
    for dataset_name in datasets:
        if not os.path.exists(dataset_name) and (not os.path.exists(cfg_files_dataset[dataset_name])):
            raise Exception(f"Dataset {dataset_name} does not exist.")

    for dataset_name in datasets:
        if os.path.exists(dataset_name):
            cfg_file_ds = dataset_name
        else:
            cfg_file_ds = cfg_files_dataset[dataset_name]
        cfg_file_model = args.model
        # pdb.set_trace()
        cmd = ["python", "tools/zero_shot.py", "--ds", cfg_file_ds, "--model", cfg_file_model, ]

        subprocess.run(cmd)
      

if __name__ == "__main__":
    run_jobs()

