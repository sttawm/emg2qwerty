#!/usr/bin/env python3
"""
Submit training jobs to Google Cloud Vertex AI

This script handles:
1. Building the Docker image
2. Pushing to Artifact Registry
3. Creating and submitting the Vertex AI training job

Usage:
    python train_remote.py
    python train_remote.py --experiment baseline_v1 user=generic trainer.max_epochs=100
    python train_remote.py --gpu NVIDIA_TESLA_V100
    python train_remote.py --spot  # Preemptible/spot (60-70% cheaper)
    python train_remote.py trainer.accelerator=cpu  # CPU only (no GPU quota)
    python train_remote.py --build-only  # Only build, don't submit
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def load_env_file(filepath: Path):
    """Load environment variables from a shell export file"""
    if not filepath.exists():
        return

    with open(filepath) as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue
            # Parse: export VAR="value" or export VAR=value
            if 'export ' in line:
                var_part = line.split('export ', 1)[1]
                if '=' in var_part:
                    key, value = var_part.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    # Expand ${VAR} references
                    while '${' in value:
                        start = value.index('${')
                        end = value.index('}', start)
                        var_name = value[start+2:end]
                        # Check if it's from oc.env (Hydra syntax)
                        if var_name.startswith('oc.env:'):
                            var_name = var_name.split(':', 1)[1]
                        var_value = os.environ.get(var_name, '')
                        value = value[:start] + var_value + value[end+1:]
                    os.environ[key] = value


# Auto-load configuration files
script_dir = Path(__file__).parent
load_env_file(script_dir / 'bootstrap' / 'shared_config.env')
load_env_file(script_dir / 'bootstrap' / '.teammate_config.env')

# Configuration from environment variables (now auto-loaded)
PROJECT_ID = os.getenv('GCP_PROJECT_ID')
REGION = os.getenv('GCP_REGION', 'us-central1')
ARTIFACT_REGISTRY = os.getenv('ARTIFACT_REGISTRY')
SHARED_DATA_BUCKET = os.getenv('SHARED_DATA_BUCKET')
SHARED_LOGS_BUCKET = os.getenv('SHARED_LOGS_BUCKET')
TEAMMATE_NAME = os.getenv('TEAMMATE_NAME', 'unnamed')

# Validate configuration
if not PROJECT_ID:
    print("ERROR: GCP_PROJECT_ID not set")
    print("Did you run: cd bootstrap && ./setup_teammate.sh")
    sys.exit(1)

if not SHARED_DATA_BUCKET or not SHARED_LOGS_BUCKET:
    print("ERROR: Shared bucket environment variables not set")
    print("Make sure bootstrap/shared_config.env exists (should be in git)")
    sys.exit(1)

if not ARTIFACT_REGISTRY:
    ARTIFACT_REGISTRY = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/emg2qwerty-training"

# Docker image configuration
IMAGE_NAME = "emg2qwerty-training"
IMAGE_TAG = datetime.now().strftime("%Y%m%d-%H%M%S")
IMAGE_URI = f"{ARTIFACT_REGISTRY}/{IMAGE_NAME}:{IMAGE_TAG}"
IMAGE_URI_LATEST = f"{ARTIFACT_REGISTRY}/{IMAGE_NAME}:latest"


def run_command(cmd: list[str], description: str, capture_output: bool = False) -> tuple[bool, str]:
    """Run a shell command and handle errors"""
    print(f"\n{'='*80}")
    print(f"{description}")
    print(f"{'='*80}")
    print(f"Running: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, capture_output=capture_output, text=True)

    if result.returncode != 0:
        print(f"\n❌ ERROR: {description} failed")
        if capture_output and result.stderr:
            print(f"Error details:\n{result.stderr}")
        return False, ""

    print(f"\n✓ {description} completed successfully")
    return True, result.stdout if capture_output else ""


def build_docker_image() -> bool:
    """Build the Docker image for Vertex AI training (using Cloud Build)"""
    print("\n🔨 Building Docker image with Cloud Build...")
    print("(No local Docker required!)")

    # Build using Cloud Build instead of local Docker
    cmd = [
        "gcloud", "builds", "submit",
        "--project", PROJECT_ID,
        "--region", REGION,
        "--tag", IMAGE_URI,
        "--timeout", "20m",
        "."
    ]

    success, _ = run_command(cmd, "Cloud Build image build")

    if success:
        # Tag as latest
        print("\n📦 Tagging image as 'latest'...")
        tag_cmd = [
            "gcloud", "artifacts", "docker", "tags", "add",
            IMAGE_URI,
            IMAGE_URI_LATEST,
            "--project", PROJECT_ID
        ]
        run_command(tag_cmd, "Tag image as latest")

    return success


def push_docker_image() -> bool:
    """Push the Docker image to Artifact Registry"""
    # Cloud Build already pushed the image, so this is a no-op
    print("\n✓ Image already in Artifact Registry (Cloud Build pushed it)")
    return True


def submit_training_job(
    job_name: Optional[str] = None,
    experiment_name: Optional[str] = None,
    machine_type: str = "n1-standard-4",
    accelerator_type: str = "NVIDIA_TESLA_T4",
    accelerator_count: int = 1,
    training_args: str = "",
    use_spot: bool = False,
    no_gpu: bool = False
) -> tuple[bool, str]:
    """Submit a training job to Vertex AI"""
    print("\n🚀 Submitting training job to Vertex AI...")

    # Generate job name if not provided (include teammate name and experiment name)
    if not job_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if experiment_name:
            # Include experiment name in job name for easy identification in console
            job_name = f"{TEAMMATE_NAME}_{experiment_name}_{timestamp}"
        else:
            job_name = f"{TEAMMATE_NAME}_train_{timestamp}"

    # Create a temporary config file for the job
    import tempfile
    import json

    # Use job name as experiment name if not specified
    if not experiment_name:
        experiment_name = job_name

    # Build container spec with environment variables
    container_spec = {
        "imageUri": IMAGE_URI,
        "env": [
            {"name": "SHARED_DATA_BUCKET", "value": SHARED_DATA_BUCKET},
            {"name": "SHARED_LOGS_BUCKET", "value": SHARED_LOGS_BUCKET},
            {"name": "EXPERIMENT_NAME", "value": experiment_name}
        ]
    }

    # Add training arguments if provided
    if training_args:
        container_spec["args"] = training_args.split()

    # Build machine spec
    machine_spec = {
        "machineType": machine_type
    }

    # Only add GPU if not CPU-only training
    if not no_gpu:
        machine_spec["acceleratorType"] = accelerator_type
        machine_spec["acceleratorCount"] = accelerator_count

    # Build worker pool spec
    worker_pool_spec = {
        "machineSpec": machine_spec,
        "replicaCount": 1,
        "containerSpec": container_spec
    }

    # Build full job spec (displayName is passed via CLI, not config)
    job_spec = {
        "workerPoolSpecs": [worker_pool_spec]
    }

    # Configure spot/preemptible instances if requested
    if use_spot:
        job_spec["scheduling"] = {
            "strategy": "SPOT"
        }

    # Write config to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(job_spec, f, indent=2)
        config_file = f.name

    try:
        # Build gcloud command with config file
        cmd = [
            "gcloud", "ai", "custom-jobs", "create",
            f"--region={REGION}",
            f"--display-name={job_name}",
            f"--config={config_file}"
        ]

        success, output = run_command(cmd, f"Submit training job '{job_name}'", capture_output=True)

        if not success:
            return False, ""

        # Print the output for user to see
        print(output)

        # Extract job ID from output (format: projects/.../locations/.../customJobs/...)
        import re
        job_id_match = re.search(r'(projects/[^/]+/locations/[^/]+/customJobs/\d+)', output)

        if job_id_match:
            job_id = job_id_match.group(1)
        else:
            # Fallback: Get the most recent job with this display name
            try:
                list_cmd = [
                    "gcloud", "ai", "custom-jobs", "list",
                    f"--region={REGION}",
                    f"--filter=displayName:{job_name}",
                    "--format=value(name)",
                    "--limit=1"
                ]
                list_result = subprocess.run(list_cmd, capture_output=True, text=True)
                if list_result.returncode == 0 and list_result.stdout.strip():
                    job_id = list_result.stdout.strip()
                else:
                    job_id = ""
            except:
                job_id = ""

        return True, job_id
    finally:
        # Clean up temp file
        import os as os_module
        os_module.unlink(config_file)


def main():
    parser = argparse.ArgumentParser(
        description="Submit EMG2QWERTY training jobs to Vertex AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default training
  python train_remote.py

  # Named experiment (groups runs in TensorBoard)
  python train_remote.py --experiment baseline_v1

  # With training arguments (pass directly, just like train_local.sh)
  python train_remote.py --experiment lr_sweep user=generic trainer.max_epochs=200

  # With GPU selection
  python train_remote.py --gpu NVIDIA_TESLA_V100 user=generic

  # Spot/preemptible instances (60-70% cheaper)
  python train_remote.py --spot

  # CPU only (no GPU quota needed - auto-detected)
  python train_remote.py trainer.accelerator=cpu trainer.max_epochs=1

  # Build only (don't submit)
  python train_remote.py --build-only
        """
    )

    parser.add_argument(
        '--build-only',
        action='store_true',
        help='Build Docker image only (don\'t submit job)'
    )

    parser.add_argument(
        '--job-name',
        type=str,
        help='Custom job name (auto-generated if not provided)'
    )

    parser.add_argument(
        '--experiment',
        type=str,
        help='Experiment name for grouping runs in TensorBoard (defaults to job name)'
    )

    parser.add_argument(
        '--machine',
        type=str,
        default='n1-standard-4',
        help='Machine type (default: n1-standard-4)'
    )

    parser.add_argument(
        '--gpu',
        type=str,
        default='NVIDIA_TESLA_T4',
        choices=['NVIDIA_TESLA_K80', 'NVIDIA_TESLA_T4', 'NVIDIA_TESLA_V100', 'NVIDIA_TESLA_P4', 'NVIDIA_TESLA_P100'],
        help='GPU type (default: NVIDIA_TESLA_T4)'
    )

    parser.add_argument(
        '--no-gpu',
        action='store_true',
        help='Train without GPU (auto-detected if trainer.accelerator=cpu is passed)'
    )

    parser.add_argument(
        '--gpu-count',
        type=int,
        default=1,
        help='Number of GPUs (default: 1)'
    )

    parser.add_argument(
        '--spot',
        action='store_true',
        help='Use spot (preemptible) instances (60-70% cheaper but can be interrupted)'
    )

    # Use parse_known_args to capture training args
    args, training_args = parser.parse_known_args()
    training_args_str = ' '.join(training_args)

    # Auto-detect if CPU training is requested
    if 'trainer.accelerator=cpu' in training_args_str:
        args.no_gpu = True
        print("ℹ️  Detected trainer.accelerator=cpu, automatically disabling GPU allocation")

    # Generate job name preview
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    preview_job_name = args.job_name or f"{TEAMMATE_NAME}_train_{timestamp}"
    preview_experiment = args.experiment or preview_job_name

    print(f"\n{'='*80}")
    print("Vertex AI Training Job Submission")
    print(f"{'='*80}")
    print(f"Teammate: {TEAMMATE_NAME}")
    print(f"Job Name: {preview_job_name}")
    print(f"Experiment: {preview_experiment}")
    print(f"Personal Project: {PROJECT_ID}")
    print(f"Region: {REGION}")
    print(f"GPU: {args.gpu if not args.no_gpu else 'None (CPU only)'}")
    print(f"Shared Data: gs://{SHARED_DATA_BUCKET}/data/")
    print(f"Shared Logs: gs://{SHARED_LOGS_BUCKET}/logs/{preview_experiment}/")
    print(f"\nCommand that will run in container:")
    print(f"  python -m emg2qwerty.train cluster=vertex {training_args_str}")
    print(f"{'='*80}\n")

    # Always build Docker image
    if not build_docker_image():
        sys.exit(1)

    if not push_docker_image():
        sys.exit(1)

    # Submit training job (unless --build-only)
    if not args.build_only:
        success, job_id = submit_training_job(
            job_name=args.job_name,
            experiment_name=args.experiment,
            machine_type=args.machine,
            accelerator_type=args.gpu,
            accelerator_count=args.gpu_count,
            training_args=training_args_str,
            use_spot=args.spot,
            no_gpu=args.no_gpu
        )
        if not success:
            sys.exit(1)

        print(f"\n{'='*80}")
        print("✅ Job submitted successfully!")
        print(f"{'='*80}")
        print("\nMonitor your job at:")
        print(f"https://console.cloud.google.com/vertex-ai/training/custom-jobs?project={PROJECT_ID}")
        print("\nView logs with:")
        if job_id:
            print(f"gcloud ai custom-jobs stream-logs {job_id} --region={REGION}")
        else:
            print(f"gcloud ai custom-jobs stream-logs {args.job_name or 'JOB_ID'} --region={REGION}")
        print("\nView TensorBoard (real-time metrics):")
        print("tensorboard --logdir=gs://emg2qwerty-team-logs/logs/")
        print("# Then open http://localhost:6006")
        print(f"\n{'='*80}\n")
    else:
        print(f"\n✅ Docker image built and pushed successfully!")
        print(f"To submit a job, run: python train_remote.py\n")


if __name__ == "__main__":
    main()
