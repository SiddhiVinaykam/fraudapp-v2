# fraudappv2 Application

A Application created with Physarum platform

## Project Structure

```bash
AppPackage/
└── physarum_code
    ├── README.md
    ├── ci
    │   ├── inference-ksvc-create-run.yaml
    │   ├── inference-ksvc-create-task.yaml
    │   ├── inference-ksvc-delete-run.yaml
    │   ├── inference-ksvc-delete-task.yaml
    │   ├── inference-ksvc-promote-canary-run.yaml
    │   ├── inference-ksvc-promote-canary-task.yaml
    │   ├── inference-ksvc-rollout-canary-run.yaml
    │   ├── inference-ksvc-rollout-canary-task.yaml
    │   ├── inference-ksvc-update-run.yaml
    │   ├── inference-ksvc-update-task.yaml
    │   ├── inference_graph-deploy-all-run.yaml
    │   ├── inference_graph-deploy-all-task.yaml
    │   ├── tekton-secret.yaml
    │   └── tekton-service-account.yaml
    ├── deploy
    │   ├── fraudappv2-dev-tekton.Dockerfile
    │   ├── fraudappv2-dev.Dockerfile
    │   ├── fraudappv2-prod.Dockerfile
    │   ├── physarum_code-base-sklearn.Dockerfile
    │   └── physarum_codetekton-base.Dockerfile
    ├── notes.txt
    ├── src
    │   ├── fraudappv2
    │   │   ├── pipeline.py
    │   │   └── start.py
    │   └── setup.py
    └── tmp
        └── adult.csv

6 directories, 25 files
```

## Set up
```
virtualenv -p python3 venv
source venv/bin/activate
pip install -e physarum_code/src/.
```

## Running pipelines local
```bash
fraudappv2 pipelines pipeline_name run_all
```

## Deploy pipeline dev
```bash
fraudappv2 pipelines pipeline_name deploy-dev
```


## Deploy pipeline prod
```bash
fraudappv2 pipelines pipeline_name deploy-prod
```

## Deploy Model to create infrenceservice
```bash
fraudappv2 inference ksvc-create
```

## Update infrenceservice
```bash
fraudappv2 inference ksvc-update
```

## Delete inferenceservice
```bash
fraudappv2 inference ksvc-delete
```

## Docker
1. Build image
```bash
docker build -t fraudappv2:latest -f physarum_code/deploy/fraudappv2-prod.Dockerfile .
```
2. Run container
```bash
docker run fraudappv2:latest
```