import warnings
import os
import sys
import json
import signal
import subprocess

log_dir = "/path/to/logs"  # Change this to your desired log directory
os.makedirs(log_dir, exist_ok=True)

config_dir = "/path/to/configs"  # Change this to your configuration directory
if not os.path.exists(config_dir):
    print(f"❌ Configuration directory '{config_dir}' does not exist. Please create it and add the necessary JSON files.")
    sys.exit(1)

def run_with_group(args, log_handle):
    # Start subprocess in a new process group
    return subprocess.run(
        args,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        check=True,
        preexec_fn=os.setsid
    )

def main():
    try:
        # Load configurations from JSON files
        with open(f'{config_dir}/parameters.json', 'r') as f:
            parameters_config = json.load(f)

        with open(f'{config_dir}/datasets.json', 'r') as f:
            datasets_config = json.load(f)

        with open(f'{config_dir}/models.json', 'r') as f:
            models_config = json.load(f)
        
        mlruns_directory = parameters_config['mlruns_directory']
        epochs = parameters_config['epochs']
        learning_rate = parameters_config['learning_rate']
        num_workers = parameters_config['num_workers']
        train_shuffle = parameters_config['train_shuffle']
        val_shuffle = parameters_config['validation_shuffle']
        test_shuffle = parameters_config['test_shuffle']
        pretrained_option = parameters_config['pretrained']
        pretrained_name = parameters_config['pretrained_name']
        
        for pretrained in pretrained_option:
            for task_name in datasets_config:
                for dataset_name in datasets_config[task_name]:
                    dataset_class = datasets_config[task_name][dataset_name]['dataloader_class']
                    train_conf = datasets_config[task_name][dataset_name]['train_config']
                    validation_conf = datasets_config[task_name][dataset_name]['validation_config']
                    test_conf = datasets_config[task_name][dataset_name]['test_config']
                    model_conf = datasets_config[task_name][dataset_name]['model_config']
                    batch_size = datasets_config[task_name][dataset_name]['batch_size']
                    
                    for model_class in models_config[task_name]:
                        log_file = f"{log_dir}/{dataset_name}_{model_class}_{pretrained}.log"

                        if os.path.exists(log_file):
                            print(f"⏩ Skipping run for dataset '{dataset_name}' with pretrained {pretrained} model '{model_class}' (log file exists).")
                            continue

                        print(f"Starting run for dataset '{dataset_name}' with pretrained {pretrained} model '{model_class}'")
                        sys.stdout.flush()

                        args = [
                            sys.executable,
                            "/path/to/mlflow_run.py",  # Change this to the actual path of your mlflow_run.py script
                            "--dataset_class", dataset_class,
                            "--model_class", model_class,
                            "--batch_size", str(batch_size),
                            "--epochs", str(epochs),
                            "--mlruns_directory", mlruns_directory,
                            "--train_shuffle", str(train_shuffle),
                            "--val_shuffle", str(val_shuffle),
                            "--test_shuffle", str(test_shuffle),
                            "--num_workers", str(num_workers),
                            "--train_conf", json.dumps(train_conf),
                            "--validation_conf", json.dumps(validation_conf),
                            "--test_conf", json.dumps(test_conf),
                            "--model_conf", json.dumps(model_conf),
                            "--learning_rate", str(learning_rate),
                            "--pretrained", str(pretrained),
                            "--dataset_name", dataset_name,
                            "--task_name", task_name,
                            "--pretrained_name", json.dumps(pretrained_name)
                        ]
                        try:
                            with open(log_file, "w") as lf:
                                run_with_group(args, lf)
                        except subprocess.CalledProcessError as e:
                            print(f"❌ Error in run for dataset '{dataset_name}' with pretrained {pretrained} with model '{model_class}': {e}")
    
    except KeyboardInterrupt:
        # If you hit Ctrl+C, kill the whole process group
        print("🔴 Terminating all subprocesses...")
        os.killpg(0, signal.SIGTERM)
            
if __name__ == "__main__":
    main()