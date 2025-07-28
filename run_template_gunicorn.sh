#!/bin/bash

containsElement() {
    # Check if an element exists in an array (https://stackoverflow.com/a/8574392)
    local e match="$1"
    shift
    for e; do [[ "$e" == "$match" ]] && return 0; done
    return 1
}

cleanup() {
    # kill all descendants of this script
    echo "$(date): $0: caught SIGINT or SIGTERM: cleaning up..."

    to_check=($$) # start from this script's PID
    seen_check=($$)
    descendants=()

    while [ ${#to_check[@]} -gt 0 ]; do
        current_pid=${to_check[0]}
        to_check=("${to_check[@]:1}") # pop first pid from the queue
        children=$(ps -o pid= --ppid "$current_pid")

        for pid in $children; do
            descendants+=("$pid")
            containsElement "$pid" "${seen_check[@]}"
            add=$?

            if [[ "$add" == "1" ]]; then
                to_check+=("$pid") # queue this child to find its children later
                seen_check+=("$pid")
            fi
        done
    done

    if [ ${#descendants[@]} -gt 0 ]; then
        # kill all descendants in reverse order to avoid killing parents first
        echo "$(date): $0: killing: ${descendants[@]}"

        for (( idx=${#descendants[@]}-1 ; idx>=0 ; idx-- )) ; do
            kill -SIGTERM "${descendants[$idx]}" 2>/dev/null
        done
    fi

    exit 0
}

trap cleanup SIGINT SIGTERM

echo "$(date) start server script: pid $$"

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

echo "$(date) starting gunicorn server with template: $template_file and model: $pretrained_model"

gunicorn --timeout 0 -w 1 --threads 10 --worker-class gthread "flask_server_wrapper:init(8, 0.2, '$pretrained_model')" &
pid=$!

echo "$(date) start server: pid $pid"

wait "$pid" # signal-aware wait to ensure cleanup on exit
