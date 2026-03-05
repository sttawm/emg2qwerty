#!/usr/bin/env python3
"""
Submit training jobs to Google Cloud Vertex AI

This script handles:
1. Building the Docker image
2. Pushing to Artifact Registry
3. Creating and submitting the Vertex AI training job

Usage:
    python vertex_submit.py --build --submit
    python vertex_submit.py --submit  # Use existing image
    python vertex_submit.py --build   # Only build, don't submit
    python vertex_submit.py --submit --args "trainer.max_epochs=100 user=generic"
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


def run_command(cmd: list[str], description: str) -> bool:
    """Run a shell command and handle errors"""
    print(f"\n{'='*80}")
    print(f"{description}")
    print(f"{'='*80}")
    print(f"Running: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        print(f"\n❌ ERROR: {description} failed")
        return False

    print(f"\n✓ {description} completed successfully")
    return True


def build_docker_image() -> bool:
    """Build the Docker image for Vertex AI training"""
    print("\n🔨 Building Docker image...")

    # Build the image
    cmd = [
        "docker", "build",
        "-t", IMAGE_URI,
        "-t", IMAGE_URI_LATEST,
        "-f", "Dockerfile",
        "."
    ]

    return run_command(cmd, "Docker image build")


def push_docker_image() -> bool:
    """Push the Docker image to Artifact Registry"""
    print("\n📤 Pushing Docker image to Artifact Registry...")

    # Push timestamped image
    if not run_command(
        ["docker", "push", IMAGE_URI],
        f"Push image {IMAGE_URI}"
    ):
        return False

    # Push latest tag
    if not run_command(
        ["docker", "push", IMAGE_URI_LATEST],
        f"Push image {IMAGE_URI_LATEST}"
    ):
        return False

    return True


def submit_training_job(
    job_name: Optional[str] = None,
    machine_type: str = "n1-standard-4",
    accelerator_type: str = "NVIDIA_TESLA_T4",
    accelerator_count: int = 1,
    training_args: str = "",
    use_spot: bool = False
) -> bool:
    """Submit a training job to Vertex AI"""
    print("\n🚀 Submitting training job to Vertex AI...")

    # Generate job name if not provided (include teammate name)
    if not job_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        job_name = f"{TEAMMATE_NAME}_train_{timestamp}"

    # Build gcloud command
    cmd = [
        "gcloud", "ai", "custom-jobs", "create",
        f"--region={REGION}",
        f"--display-name={job_name}",
        f"--worker-pool-spec=machine-type={machine_type},replica-count=1,accelerator-type={accelerator_type},accelerator-count={accelerator_count},container-image-uri={IMAGE_URI}",
    ]

    # Add training arguments if provided
    if training_args:
        # Parse args and add as container args
        args_list = training_args.split()
        container_args = ",".join(args_list)
        cmd[4] += f",container-args={container_args}"

    # Add environment variables for shared buckets
    cmd[4] += f",env=SHARED_DATA_BUCKET={SHARED_DATA_BUCKET}"
    cmd[4] += f",env=SHARED_LOGS_BUCKET={SHARED_LOGS_BUCKET}"

    # Use spot (preemptible) instances if requested
    if use_spot:
        cmd.append("--enable-web-access")
        # Note: Spot instances save ~70% but can be preempted

    return run_command(cmd, f"Submit training job '{job_name}'")


def main():
    parser = argparse.ArgumentParser(
        description="Submit EMG2QWERTY training jobs to Vertex AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default training
  python vertex_submit.py

  # With training arguments (pass directly, just like train_local.sh)
  python vertex_submit.py user=generic trainer.max_epochs=200

  # With GPU selection
  python vertex_submit.py --gpu NVIDIA_TESLA_V100 user=generic

  # Spot instances (cheaper)
  python vertex_submit.py --spot

  # Build only (don't submit)
  python vertex_submit.py --build-only
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
        '--gpu-count',
        type=int,
        default=1,
        help='Number of GPUs (default: 1)'
    )

    parser.add_argument(
        '--spot',
        action='store_true',
        help='Use spot (preemptible) instances (cheaper but can be interrupted)'
    )

    # Use parse_known_args to capture training args
    args, training_args = parser.parse_known_args()
    training_args_str = ' '.join(training_args)

    # Generate job name preview
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    preview_job_name = args.job_name or f"{TEAMMATE_NAME}_train_{timestamp}"

    print(f"\n{'='*80}")
    print("Vertex AI Training Job Submission")
    print(f"{'='*80}")
    print(f"Teammate: {TEAMMATE_NAME}")
    print(f"Job Name: {preview_job_name}")
    print(f"Personal Project: {PROJECT_ID}")
    print(f"Region: {REGION}")
    print(f"GPU: {args.gpu}")
    print(f"Shared Data: gs://{SHARED_DATA_BUCKET}/data/")
    print(f"Shared Logs: gs://{SHARED_LOGS_BUCKET}/logs/")
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
        if not submit_training_job(
            job_name=args.job_name,
            machine_type=args.machine,
            accelerator_type=args.gpu,
            accelerator_count=args.gpu_count,
            training_args=training_args_str,
            use_spot=args.spot
        ):
            sys.exit(1)

        print(f"\n{'='*80}")
        print("✅ Job submitted successfully!")
        print(f"{'='*80}")
        print("\nMonitor your job at:")
        print(f"https://console.cloud.google.com/vertex-ai/training/custom-jobs?project={PROJECT_ID}")
        print("\nView logs with:")
        print(f"gcloud ai custom-jobs stream-logs {args.job_name or 'JOB_ID'} --region={REGION}")
        print(f"\n{'='*80}\n")
    else:
        print(f"\n✅ Docker image built and pushed successfully!")
        print(f"To submit a job, run: python vertex_submit.py\n")


if __name__ == "__main__":
    main()
