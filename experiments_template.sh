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

pid=$$
pgid=$(ps -o pgid= -p "$pid" | tr -d ' ')
echo "$(date) starting script: $0 (pid: $pid; pgid: $pgid)"

#for template in $(echo "template_1 template_2 template_3"); do
for template in $(echo "template_1 template_2"); do
for m in $(echo "Llama-2-7b-hf Llama-2-7b-chat-hf"); do

./run_template_gunicorn.sh "${template}.txt" "meta-llama/${m}" &>> ./preliminar_experiments_template/experiments_template.gunicorn.${m}.${template}.log &
pid=$!

echo "$(date) start server: pid $pid"

sleep 30 # wait for the server to start

#for ldata in $(echo "eng_Latn-fra_Latn:English:French eng_Latn-deu_Latn:English:German eng_Latn-swh_Latn:English:Swahili eng_Latn-wol_Latn:English:Wolof fra_Latn-eng_Latn:French:English deu_Latn-eng_Latn:German:English swh_Latn-eng_Latn:Swahili:English wol_Latn-eng_Latn:Wolof:English"); do
for ldata in $(echo "eng_Latn-deu_Latn:English:German eng_Latn-swh_Latn:English:Swahili eng_Latn-wol_Latn:English:Wolof"); do
    l=$(echo $ldata | cut -d':' -f1)
    l1=$(echo $ldata | cut -d':' -f2)
    l2=$(echo $ldata | cut -d':' -f3)
    f="flores_200.dev.${l}.out"
    for n_icl in $(echo "0 5"); do
    for seed in $(if [[ "$n_icl" == "0" ]]; then echo "42"; else echo "42 43 44"; fi); do
        f2="./preliminar_experiments_template/${f}.mt.preliminar_experiments.${m}.icl_random_${n_icl}.random_pool_flores_dev.seed_${seed}"
        if [[ -f "${f2}.out" ]]; then
            echo "$(date) skipping: ${f2} already exists"
            continue
        fi

        echo "$(date) $template $f2"
        cat "$f" | cut -f1 | python3 baseline_icl_random.py "$l1" "$l2" "$f" "${n_icl}" 50 "${seed}" > "${f2}.out" 2> "${f2}.log"
    done
    done
done

echo "$(date) end: $template $m"
echo "$(date) kill server: pid $pid"

# Kill the child script (and all its children)
#pgid=$(ps -o pgid= -p "$pid" | tr -d ' ')
#echo "$(date) kill server: $pid pgid $pgid"
#pkill -SIGINT -g $pgid # it triggers the cleanup function of this script
kill -SIGTERM "$pid" 2>/dev/null

sleep 30 # wait for the server to stop

done
done
