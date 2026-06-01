# ICL examples selector with RL for MT

In this repository we relase the code and datasets used for a method developed for selecting in-context learning (ICL) examples with reinforcement learning (RL) for machine translation (MT).

# Usage

First, you will need to deploy the flask-based server in order to send requests to the large language model (LLM):

```bash
# In the case of 8 GPUs:
for p in $(seq 8000 1 8007); do
    MT_ICL_IS_CAUSAL_OR_CHAT="causal" ./run_template_gunicorn.sh template_2_chat_v2.txt "Qwen/Qwen2.5-1.5B" "$p" 4 1 10 &> ./flask_server.${p}.log &
done
```

Then you will need to deploy the flask-based server for computing COMET scores:

```bash
gunicorn --bind "0.0.0.0:8000" --timeout 0 -w 1 --threads 64 --worker-class gthread "evaluate_comet_22_flask_server_wrapper:init(64, 0.2, False)" &> ./eval_comet.log &
```

Then you can train a model:

```bash
train="flores_200.dev.eng_Latn-spa_Latn.out"
dev="flores_200.devtest.eng_Latn-spa_Latn.out.shuf.out.dev.out"
pool="./flores_200.dev.eng_Latn-spa_Latn.out"
server="127.0.0.1"

python3 ./gym_env_run_experiments_v2_discrete_ppo.py \
  English Spanish "$train":"$dev" "$pool" seed="42" \
  translate_model_api="http://${server}:8007/translate|http://${server}:8006/translate|http://${server}:8005/translate|http://${server}:8004/translate|http://${server}:8003/translate|http://${server}:8002/translate|http://${server}:8001/translate|http://${server}:8000/translate" \
  embedding_single_token_model_api="http://${server}:8007/get_embedding_from_model_embedding_matrix" \
  embedding_pooling_model_api="http://${server}:8007/get_embedding_pooling|http://${server}:8006/get_embedding_pooling|http://${server}:8005/get_embedding_pooling|http://${server}:8004/get_embedding_pooling|http://${server}:8003/get_embedding_pooling|http://${server}:8002/get_embedding_pooling|http://${server}:8001/get_embedding_pooling|http://${server}:8000/get_embedding_pooling" \
  embedding_external_system="http://${server}:8007/get_embedding_from_given_model|http://${server}:8006/get_embedding_from_given_model|http://${server}:8005/get_embedding_from_given_model|http://${server}:8004/get_embedding_from_given_model|http://${server}:8003/get_embedding_from_given_model|http://${server}:8002/get_embedding_from_given_model|http://${server}:8001/get_embedding_from_given_model|http://${server}:8000/get_embedding_from_given_model" \
  max_icl_examples="5" store_model_on_eval="1" state_representation="representation_per_token_with_features_v3" eval_strategy_training="api-eval" \
  eval_strategy_eval="api-eval" eval_model_api="http://${server}:8000/evaluate_comet_22" embedding_pooling_model_layer="75%" use_transformer="1" \
  state_window_length="512" use_vec_normalize="0" reward_division="100" \
  &> ./result
```

Then you can evaluate your model:

```bash
server="127.0.0.1"

for data in $(echo "flores_200.devtest:flores_devtest_test_split.:.shuf.out.test.out"); do
    fdata=$(echo "$data" | cut -d: -f1)
    ndata=$(echo "$data" | cut -d: -f2)
    sdata=$(echo "$data" | cut -d: -f3)
    dev="./${fdata}.eng_Latn-spa_Latn.out${sdata}"
    pool="./flores_200.dev.eng_Latn-spa_Latn.out"
    model="/path/to/your/model" # TODO modify this value (check training log file)
    out="./inference_eval.api_eval.${ndata}1.log"
    if [[ -f "$out" ]]; then continue; fi
    echo "$(date) $data : $out"

    python3 ./gym_env_run_experiments_inference_v7_td_new_arch.discrete_ppo.parallel.py \
      "$model" English Spanish "$dev" "$pool" seed="42" \
      translate_model_api="http://${server}:8000/translate|http://${server}:8001/translate|http://${server}:8002/translate|http://${server}:8003/translate|http://${server}:8004/translate|http://${server}:8005/translate|http://${server}:8006/translate|http://${server}:8007/translate" \
      embedding_single_token_model_api="http://${server}:8000/get_embedding_from_model_embedding_matrix" \
      embedding_pooling_model_api="http://${server}:8007/get_embedding_pooling|http://${server}:8006/get_embedding_pooling|http://${server}:8005/get_embedding_pooling|http://${server}:8004/get_embedding_pooling|http://${server}:8003/get_embedding_pooling|http://${server}:8002/get_embedding_pooling|http://${server}:8001/get_embedding_pooling|http://${server}:8000/get_embedding_pooling" \
      embedding_external_system="http://${server}:8007/get_embedding_from_given_model|http://${server}:8006/get_embedding_from_given_model|http://${server}:8005/get_embedding_from_given_model|http://${server}:8004/get_embedding_from_given_model|http://${server}:8003/get_embedding_from_given_model|http://${server}:8002/get_embedding_from_given_model|http://${server}:8001/get_embedding_from_given_model|http://${server}:8000/get_embedding_from_given_model" \
      max_icl_examples="5" state_representation="representation_per_token_with_features_v3" linear_bottleneck="0" \
      eval_strategy_eval="api-eval" eval_model_api="http://${server}:8000/evaluate_comet_22" store_rewards_fn="${out}.rewards" \
      embedding_pooling_model_layer="75%" use_transformer="1" state_window_length="512" use_vec_normalize="0" \
    &> "$out"
done

date
```
