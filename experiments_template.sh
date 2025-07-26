#!/bin/bash

for template in $(echo "template_1 template_2 template_3"); do
for m in $(echo "Llama-2-7b-chat-hf Llama-2-7b-hf"); do

./run_template_gunicorn.sh "${template}.txt" "meta-llama/${m}" &> ./preliminar_experiments_template/experiments_template.gunicorn.${m}.${template}.log &
pid=$!

sleep 30 # wait for the server to start

for ldata in $(echo "eng_Latn-fra_Latn:English:French eng_Latn-deu_Latn:English:German eng_Latn-swh_Latn:English:Swahili eng_Latn-wol_Latn:English:Wolof fra_Latn-eng_Latn:French:English deu_Latn-eng_Latn:German:English swh_Latn-eng_Latn:Swahili:English wol_Latn-eng_Latn:Wolof:English"); do
    l=$(echo $ldata | cut -d':' -f1)
    l1=$(echo $ldata | cut -d':' -f2)
    l2=$(echo $ldata | cut -d':' -f3)
    f="flores_200.dev.${l}.out"
    for n_icl in $(echo "0 5"); do
    for seed in $(if [[ "$n_icl" == "0" ]]; then echo "42"; else echo "42 43 44"; fi); do
        f2="./preliminar_experiments_template/${f}.mt.preliminar_experiments.${m}.icl_random_${n_icl}.random_pool_flores_dev.seed_${seed}"
        echo "$(date) $template $f2"
        cat "$f" | cut -f1 | python3 baseline_icl_random.py "$l1" "$l2" "$f" "${n_icl}" 50 "${seed}" > "${f2}.out" 2> "${f2}.log"
    done
    done
    echo "$(date) end"
done

# Kill the child script (and all its children)
pgid=$(ps -o pgid= -p "$pid" | tr -d ' ')
echo "$pid pgid $pgid"
pkill -SIGINT -g $pgid

done
done
