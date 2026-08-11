import mlflow
import torch
import shutil
import glob
from tensorboard.backend.event_processing import event_accumulator
from fvcore.nn import FlopCountAnalysis
from datetime import datetime
import importlib
import logging

def flatten_test_metrics(metrics):
    flat = {}
    for metric_dict in metrics:
        for name, value in metric_dict.items():
            if isinstance(value, (int, float)):
                flat[f"{name}/test"] = value
            elif isinstance(value, (list, torch.Tensor)):
                values = value if isinstance(value, list) else value.cpu().numpy()
                if "per Class" in name:
                    for i, v in enumerate(values):
                        flat[f"{name}_class_{i}/test"] = float(v)
                else:
                    flat[f"{name}/test"] = float(torch.tensor(values).float().mean().item())
    return flat

def get_file_list_from_pattern(pattern):
    file_list = glob.glob(pattern)

    if not file_list:
        print(f"No files found matching pattern: {pattern}")
        return []

    return file_list

def log_tensorboard_metrics_to_mlflow(event_file):
    # Load and parse the event file
    ea = event_accumulator.EventAccumulator(event_file)
    ea.Reload()

    # Log all scalar metrics to MLflow
    for tag in ea.Tags()["scalars"]:
        for event in ea.Scalars(tag):
            mlflow.log_metric(tag, event.value, step=event.step)
    
    # Find the epoch with the lowest validation loss
    if "Loss/val" in ea.Tags()["scalars"]:
        val_losses = ea.Scalars("Loss/val")
        best_event = min(val_losses, key=lambda e: e.value)
        best_epoch = best_event.step
        mlflow.set_tag("best_epoch", best_epoch)

        # Collect all scalar metrics for that epoch
        best_metrics = {}
        for tag in ea.Tags()["scalars"]:
            if "/val" in tag:
                # Find the event at this epoch (if it exists)
                events = [e for e in ea.Scalars(tag) if e.step == best_epoch]
                if events:
                    best_metrics[f"best_{tag}"] = events[0].value

        # Log them to MLflow
        mlflow.log_metrics(best_metrics)
    else:
        print("Warning: No 'Loss/val' tag found in TensorBoard logs. Cannot determine best epoch.")

def model_run(mlruns_directory, train_dataset, validation_dataset, test_dataset, dataset_config, epochs, temp_directory, model_config, run_name, dataset_name, model_class, task_name, model):
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(mlruns_directory)
    
    # Assign runs to a named experiment
    mlflow.set_experiment(dataset_name)

    # Start MLflow run
    with mlflow.start_run(run_name=run_name) as run:
        # Print experiment and run IDs right away
        print(f"MLflow experiment ID: {run.info.experiment_id}")
        print(f"MLflow run ID: {run.info.run_id}")

        # Log tags
        mlflow.set_tag("model_type", model_class)
        mlflow.set_tag("dataset", dataset_name)
        mlflow.set_tag("task", task_name)
        mlflow.set_tag("added", "pin_memory=True, evaluate_train_every_n_epochs=epochs+1")

        # Log parameters
        mlflow.log_params({
            "epochs": epochs,
            "batch_size": dataset_config["batch_size"],
            "learning_rate": model_config["learning_rate"],
            "pretrained": model_config["pretrained"],
            "num_workers": dataset_config["num_workers"]
        })

        model.prepare()

        # Calculate FLOPS for a single forward pass
        temp_model = model.model
        temp_model.eval()
        device = next(temp_model.parameters()).device

        # Create a synthetic input tensor with the same shape as the training data
        sample_tensor, _ = train_dataset[0]
        if task_name == "change_detection":
            c, h, w = sample_tensor[0].shape
            synthetic_input1 = torch.randn(1, c, h, w).to(device)
            synthetic_input2 = torch.randn(1, c, h, w).to(device)

            with torch.no_grad():
                flops = FlopCountAnalysis(temp_model, (synthetic_input1, synthetic_input2))
                total_gflops = flops.total() / 1e9
        else:
            c, h, w = sample_tensor.shape
            synthetic_input = torch.randn(1, c, h, w).to(device)

            with torch.no_grad():
                flops = FlopCountAnalysis(temp_model, synthetic_input)
                total_gflops = flops.total() / 1e9
        
        # Log the metric to MLflow
        mlflow.log_metric("GFLOPS_forward_pass", total_gflops)
        mlflow.set_tag("flops_input_shape", f"{c}x{h}x{w}")
        
        # Set model back to train mode
        temp_model.train()

        # Train model
        model.train_and_evaluate_model(
            train_dataset=train_dataset,
            epochs=epochs,
            model_directory=temp_directory,
            val_dataset=validation_dataset,
            run_id=run_name,
        )

        # Log TensorBoard metrics to MLflow
        metrics_path = get_file_list_from_pattern(f"{temp_directory}/{run_name}/events.out.tfevents*")[0]
        log_tensorboard_metrics_to_mlflow(metrics_path)

        # Optional: reload specific weights before logging
        best_model_path = get_file_list_from_pattern(f"{temp_directory}/{run_name}/*best*.pth.tar")[0]

        # Evaluate on test dataset
        model.load_model(best_model_path)
        model.running_metrics.reset()
        model.evaluate(dataset=test_dataset, model_path=best_model_path)
        test_metrics = model.running_metrics.get_scores(model.metrics)
        mlflow.log_metrics(flatten_test_metrics(test_metrics))

        # Log artifacts
        mlflow.log_artifact(best_model_path, artifact_path="checkpoints")
        mlflow.log_artifact(metrics_path, artifact_path="metrics")

    
    # Remove the directory and all its contents
    shutil.rmtree(temp_directory)

def prepare_run(dataset_class, model_class, batch_size, epochs, mlruns_directory, train_shuffle, val_shuffle, test_shuffle, num_workers, train_conf, validation_conf, test_conf, model_conf, learning_rate, pretrained, dataset_name, task_name, pretrained_name):
    dataset_module = importlib.import_module("aitlas.datasets")
    dataset_cls = getattr(dataset_module, dataset_class)
    model_module = importlib.import_module("aitlas.models")
    model_cls = getattr(model_module, model_class)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    temp_directory = f"./temp_{timestamp}"

    train_dataset_config = {
        "batch_size": batch_size,
        "shuffle": train_shuffle,
        "num_workers": num_workers,
        "pin_memory" : True,
        **train_conf
    }
    train_dataset = dataset_cls(train_dataset_config)

    validation_dataset_config = {
        "batch_size": batch_size,
        "shuffle": val_shuffle,
        "num_workers": num_workers,
        "pin_memory" : True,
        **validation_conf
    }
    validation_dataset = dataset_cls(validation_dataset_config)

    test_dataset_config = {
        "batch_size": batch_size,
        "shuffle": test_shuffle,
        "num_workers": num_workers,
        "pin_memory" : True,
        **test_conf
    }
    test_dataset = dataset_cls(test_dataset_config)

    model_config = {
        "learning_rate": learning_rate,
        "pretrained": pretrained,
        "evaluate_train_every_n_epochs": epochs+1,   # Don't evaluate on training set to speed up the runs
        "automatic_mixed_precision": True,
        **model_conf
    }
    model = model_cls(model_config)

    if pretrained:
        training_type = pretrained_name[1]
    else:
        training_type = pretrained_name[0]
    run_name = f"{task_name}_{dataset_name}_{training_type}_{model_class}"
    
    model_run(mlruns_directory, train_dataset, validation_dataset, test_dataset, train_dataset_config, epochs, temp_directory, model_config, run_name, dataset_name, model_class, task_name, model)
    logging.getLogger().handlers.clear()

if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_class")
    parser.add_argument("--model_class")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--mlruns_directory")
    parser.add_argument("--train_shuffle", type=lambda x: x.lower() == "true")
    parser.add_argument("--val_shuffle", type=lambda x: x.lower() == "true")
    parser.add_argument("--test_shuffle", type=lambda x: x.lower() == "true")
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--train_conf")
    parser.add_argument("--validation_conf")
    parser.add_argument("--test_conf")
    parser.add_argument("--model_conf")
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--pretrained", type=lambda x: x.lower() == "true")
    parser.add_argument("--dataset_name")
    parser.add_argument("--task_name")
    parser.add_argument("--pretrained_name")

    args = parser.parse_args()

    # Convert JSON strings back to dicts/lists
    train_conf = json.loads(args.train_conf)
    validation_conf = json.loads(args.validation_conf)
    test_conf = json.loads(args.test_conf)
    model_conf = json.loads(args.model_conf)
    pretrained_name = json.loads(args.pretrained_name)

    prepare_run(
        args.dataset_class,
        args.model_class,
        args.batch_size,
        args.epochs,
        args.mlruns_directory,
        args.train_shuffle,
        args.val_shuffle,
        args.test_shuffle,
        args.num_workers,
        train_conf,
        validation_conf,
        test_conf,
        model_conf,
        args.learning_rate,
        args.pretrained,
        args.dataset_name,
        args.task_name,
        pretrained_name,
    )
