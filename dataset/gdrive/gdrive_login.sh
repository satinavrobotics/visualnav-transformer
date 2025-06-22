#!/bin/bash
# Activate conda environment
source /opt/conda/etc/profile.d/conda.sh
conda activate training_server
# GCP ADC login with scopes
echo "Please open the URL if prompted..."
gcloud auth application-default login --scopes=https://www.googleapis.com/auth/drive,https://www.googleapis.com/auth/cloud-platform
# Set the quota project
gcloud auth application-default set-quota-project satinav-61179

# Execute with: . drive_login.sh