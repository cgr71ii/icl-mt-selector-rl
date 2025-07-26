#!/bin/bash

template_file="$1" # e.g., template_1.txt
pretrained_model="$2" # e.g., meta-llama/Llama-2-7b-hf

if [[ -z "$template_file" ]] || [[ -z "$pretrained_model" ]]; then
    echo "Usage: $0 <template_file> <pretrained_model>"
    exit 1
fi

if [[ ! -f "$template_file" ]]; then
    echo "Template file '$template_file' does not exist."
    exit 1
fi

envvars_before=$(export | sort)

set -a # Export all variables to child processes
source "$template_file" # Load the template file
set +a # Stop exporting future imported variables to child processes

envvars_after=$(export | sort)

gunicorn --timeout 0 -w 1 --threads 10 --worker-class gthread "flask_server_wrapper:init(8, 0.2, '$pretrained_model')"
