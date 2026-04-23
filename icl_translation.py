
import os
import sys
import logging

import utils

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers import StoppingCriteria, StoppingCriteriaList

logger = utils.set_up_logging_logger(logging.getLogger("MT_ICL.icl_translation"), level=logging.DEBUG)

class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_token_ids):
        assert isinstance(stop_token_ids, list), "stop_token_ids must be a list"

        for token_id in stop_token_ids:
            assert isinstance(token_id, int), "stop_token_ids must be a list of integers (token IDs)"

        super().__init__()
        self.stop_token_ids = set(stop_token_ids)

    def __call__(self, input_ids, scores, **kwargs):
        # input_ids is a tensor of shape (batch_size, sequence_length), including the prompt

        for idx in range(len(input_ids)):
            last_token_id = input_ids[idx, -1].item()

            if last_token_id in self.stop_token_ids:
                return True

        return False

class StopOnTokensSeq(StoppingCriteria):
    def __init__(self, stop_token_ids, tokenizer):
        assert isinstance(stop_token_ids, list), "stop_token_ids must be a list"

        for l in stop_token_ids:
            assert isinstance(l, list), "stop_token_ids must be a list of lists"

            for token_id in l:
                assert isinstance(token_id, int), "stop_token_ids must be a list of lists of integers (token IDs)"

        super().__init__()
        self.stop_token_ids = set([tuple(l) for l in stop_token_ids])
        self.stop_token_token = [(l, tuple([tokenizer.decode(l2) for l2 in l]), tokenizer.decode(l)) for l in stop_token_ids]

        logger.debug("early stopping tokens (format: tuple(ids, tuple(decode(token)), decode)): %s", self.stop_token_token)

    def __call__(self, input_ids, scores, **kwargs):
        # input_ids is a tensor of shape (batch_size * beam_size, sequence_length), including the prompt

        assert len(input_ids.shape) == 2

        #logger.debug("input_ids.shape: %s", input_ids.shape)

        stop = [False] * input_ids.shape[0]

        for idx in range(len(input_ids)):
            for stop_token_id in self.stop_token_ids:
                #logger.debug("stop_token_id: %s; tuple(input_ids[idx, -len(stop_token_id):].tolist()): %s", stop_token_id, tuple(input_ids[idx, -len(stop_token_id):].tolist()))

                if tuple(input_ids[idx, -len(stop_token_id):].tolist()) == stop_token_id:
                    stop[idx] = True

                    if len(set(stop)) == 1: # all values must be True
                        return True

        return False

def translate(model, tokenizer, prompts, max_new_tokens=1024, stopping_criteria=None, normalize=True, lock=None, num_beams=4):
    all_outputs, all_original_outputs = [], []

    # Tokenize
    inputs = tokenize_prompts(prompts, tokenizer, lock=lock)
    inputs = inputs.to(model.device)

    # Generate with beam search
    # Decoding: https://aclanthology.org/2024.emnlp-main.489/
    output = model.generate(
        **inputs,
        #max_new_tokens=1024,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        early_stopping=True,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False,
        top_p=None,
        top_k=None,
        temperature=None,
        stopping_criteria=stopping_criteria,
    )

    assert inputs.input_ids.shape[0] == len(prompts)
    assert output.shape[0] == len(prompts)

    # Decode
    decoded_outputs = [tokenizer.decode(output[idx][inputs.input_ids[idx].shape[-1]:], skip_special_tokens=True) for idx in range(output.shape[0])]

    #print("\n=== TRANSLATION ===")

    for idx in range(len(decoded_outputs)):
        decoded_output = decoded_outputs[idx]
        decoded_output = decoded_output.strip().split('\n')[0]
        decoded_output = decoded_output
        original_decoded_output = str(decoded_outputs[idx])

        if normalize:
            decoded_output = decoded_output.replace('\t', ' ').replace('\n', ' ').replace('\r', '').strip()
            original_decoded_output = original_decoded_output.replace('\t', r' \t ').replace('\n', r' \n ').replace('\r', '').strip()

        all_outputs.append(decoded_output)
        all_original_outputs.append(original_decoded_output)

    return all_outputs, all_original_outputs

def tokenize_prompts(prompts, tokenizer, lock=None):
    if lock is not None:
        lock.acquire()
    inputs = tokenizer(prompts, padding="longest", return_tensors="pt", padding_side="left")
    if lock is not None:
        lock.release()

    assert inputs.input_ids.shape[0] == len(prompts), f"Batch size mismatch between tokenized inputs and prompts: {inputs.input_ids.shape[0]} vs {len(prompts)}"
    assert len(inputs.input_ids.shape) == 2, inputs.input_ids.shape # batch_size, seq_len

    return inputs

_gamma_factor_values = None

def get_embedding_pooling(model, tokenizer, prompts, pooling="mean", layer=-1, lock=None, _inputs=None, _masks=None, target_sentence_n_tokens=None, gamma=0.8, log_instead_of_ppl=True):
    global _gamma_factor_values

    # Tokenize
    if lock is not None:
        lock.acquire()
    inputs = tokenize_prompts(prompts, tokenizer, lock=None).to(model.device) if _inputs is None and _masks is None else None
    if lock is not None:
        lock.release()

    input_ids = inputs["input_ids"].to(model.device) if inputs is not None else _inputs.to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device) if inputs is not None else _masks.to(model.device)

    assert len(attention_mask.shape) == 2, f"attention_mask expected shape: (batch_size, seq_len); got: {attention_mask.shape}"

    for idx in range(attention_mask.shape[0]):
        assert attention_mask[idx].sum().item() > 0, f"All tokens are padding for input idx {idx}: input_ids: {input_ids[idx]}, attention_mask: {attention_mask[idx]}"
        assert torch.all((attention_mask[idx] == 0) | (attention_mask[idx] == 1)).item(), f"Attention mask must be binary (0 or 1) for input idx {idx}: attention_mask: {attention_mask[idx]}"
        assert torch.all(attention_mask[idx][:-1] <= attention_mask[idx][1:]).item(), f"Attention mask must be left-to-right (non-decreasing; LLMs padding) for input idx {idx}: attention_mask: {attention_mask[idx]}"

    assert torch.all(attention_mask[:,-1] == 1).item(), attention_mask
    assert isinstance(layer, (str, int)), f"layer must be either str or int; got: {type(layer)}"

    _layer = None

    # Forward pass with hidden states
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

        if _layer is None:
            if isinstance(layer, str):
                if layer[-1] == '%':
                    assert float(layer[:-1]) >= 0 and float(layer[:-1]) <= 100, f"Percentage layer must be between 0 and 100: got {layer}"

                    _layer = int(float(layer[:-1]) / 100 * (len(outputs.hidden_states) - 1) + 0.5)
                else:
                    _layer = int(layer)
            else:
                _layer = layer

            assert isinstance(_layer, int), f"_layer must be int; got: {type(_layer)}"

            logger.debug("Using layer %s (original: %s) of %d total layers", _layer, layer, len(outputs.hidden_states))

        hidden_states = outputs.hidden_states[_layer] # shape: (batch_size, seq_len, hidden_dim)

        assert len(hidden_states.shape) == 3, f"hidden_states expected shape: (batch_size, seq_len, hidden_dim); got: {hidden_states.shape}"
        assert hidden_states.shape[0] == attention_mask.shape[0], f"hidden_states and attention_mask batch size mismatch: {hidden_states.shape[0]} vs {attention_mask.shape[0]}"
        assert hidden_states.shape[1] == attention_mask.shape[1], f"hidden_states and attention_mask sequence length mismatch: {hidden_states.shape[1]} vs {attention_mask.shape[1]}"

        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()

    assert attention_mask_expanded.shape == hidden_states.shape, f"attention_mask_expanded and hidden_states shape mismatch: {attention_mask_expanded.shape} vs {hidden_states.shape}"

    if pooling == "mean":
        # Mean pooling

        sum_embeddings = torch.sum(hidden_states * attention_mask_expanded, dim=1)
        sum_mask = attention_mask_expanded.sum(dim=1)
        pooled_embeddings = sum_embeddings / sum_mask
    elif pooling == "max":
        # Max pooling

        masked_embeddings = hidden_states * attention_mask_expanded
        pooled_embeddings, _ = torch.max(masked_embeddings, dim=1)
    elif pooling == "last":
        # Last token pooling
        pooled_embeddings = hidden_states[:, -1, :]
    elif pooling == "none":
        # No pooling, return all token embeddings (after removing padding with attention mask)
        pooled_embeddings = hidden_states * attention_mask_expanded

        assert pooled_embeddings.shape == hidden_states.shape, f"pooled_embeddings and hidden_states shape mismatch: {pooled_embeddings.shape} vs {hidden_states.shape}"
    elif pooling == "features":
        logits = outputs.logits
        target_ids = input_ids.clone()
        attention_mask = attention_mask.clone()

        # Shift logits, targets, mask and hidden states (code adapted from https://github.com/jogonba2/llmixtic/blob/911bd990e84060ea25d18a783436b621bbb6e954/src/vectorizer.py#L197)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_targets = target_ids[..., 1:].contiguous()
        mask = attention_mask[..., 1:].contiguous()

        # Get probabilities
        probs = shift_logits.softmax(dim=-1)
        smallest_normal = torch.finfo(
            type=probs.dtype
        ).smallest_normal
        probs[probs == 0] = smallest_normal

        assert len(probs.shape) == 3, f"probs expected shape: (batch_size, seq_len - 1, vocab_size); got: {probs.shape}"

        # Compute features
        features = ["constant", "observed", "most_likely", "entropy"]
        len_features = len(features)
        model_features = {feature: torch.tensor([]) for feature in features}
        batch_features = compute_features(
                            probs,
                            shift_targets,
                            mask,
                            eps=smallest_normal,
                            features=features,
                        )
        for feature_name, feature in batch_features.items():
            idx = features.index(feature_name)

            assert idx != -1, f"Unknown feature name: {feature_name}"

            del features[idx]

            model_features[feature_name] = torch.cat(
                (
                    model_features[feature_name],
                    feature.cpu(),
                ),
                dim=0,
            )

        assert len(features) == 0, f"Some features were not computed: {features}"

        pooled_embeddings = torch.cat(list(model_features.values()), dim=-1)

        assert pooled_embeddings.shape == (hidden_states.shape[0], hidden_states.shape[1] - 1, len_features), f"pooled_embeddings shape mismatch: {pooled_embeddings.shape} vs {(hidden_states.shape[0], hidden_states.shape[1] - 1, len_features)}"
    elif pooling == "mean+last":
        sum_embeddings = torch.sum(hidden_states * attention_mask_expanded, dim=1)
        sum_mask = attention_mask_expanded.sum(dim=1)
        mean_pooled_embeddings = sum_embeddings / sum_mask
        last_token_embeddings = hidden_states[:, -1, :]
        pooled_embeddings = torch.cat([mean_pooled_embeddings, last_token_embeddings], dim=-1)
    elif pooling in ("target_sentence_probs_mean_reward", "target_sentence_neg_ppl_reward"):
        # Get the probabilities of the target sentence tokens (after removing padding with attention mask)

        assert target_sentence_n_tokens is not None
        assert isinstance(target_sentence_n_tokens, list), f"target_sentence_n_tokens must be a list; got: {type(target_sentence_n_tokens)}"
        assert len(target_sentence_n_tokens) == hidden_states.shape[0], f"target_sentence_n_tokens length mismatch with batch size: {len(target_sentence_n_tokens)} vs {hidden_states.shape[0]}"
        assert all(isinstance(t, int) for t in target_sentence_n_tokens), f"All target_sentence_n_tokens must be integers: {target_sentence_n_tokens}"
        assert all(t > 0 for t in target_sentence_n_tokens), f"All target_sentence_n_tokens must be positive: {target_sentence_n_tokens}"

        logits = outputs.logits
        target_ids = input_ids.clone()
        attention_mask = attention_mask.clone()

        # Shift logits, targets, mask and hidden states (code adapted from https://github.com/jogonba2/llmixtic/blob/911bd990e84060ea25d18a783436b621bbb6e954/src/vectorizer.py#L197)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_targets = target_ids[..., 1:].contiguous()
        mask = attention_mask[..., 1:].contiguous()

        # Get probabilities
        probs = shift_logits.softmax(dim=-1)
        smallest_normal = torch.finfo(
            type=probs.dtype
        ).smallest_normal
        probs[probs == 0] = smallest_normal

        assert len(probs.shape) == 3, f"probs expected shape: (batch_size, seq_len - 1, vocab_size); got: {probs.shape}"

        batch_features = compute_features(
                            probs,
                            shift_targets,
                            mask,
                            eps=smallest_normal,
                            features=["observed"],
                        )
        token_log_probs = batch_features["observed"]

        assert len(token_log_probs.shape) == 3, f"token_log_probs expected shape: (batch_size, seq_len - 1, 1); got: {token_log_probs.shape}"
        assert token_log_probs.shape == (hidden_states.shape[0], hidden_states.shape[1] - 1, 1), f"token_log_probs shape mismatch: {token_log_probs.shape} vs {(hidden_states.shape[0], hidden_states.shape[1] - 1, 1)}"
        assert token_log_probs.shape[1] == attention_mask_expanded.shape[1] - 1, f"token_log_probs and attention_mask_expanded sequence length mismatch: {token_log_probs.shape[1]} vs {attention_mask_expanded.shape[1] - 1}"
        assert len(target_sentence_n_tokens) == token_log_probs.shape[0], f"target_sentence_n_tokens length mismatch with batch size: {len(target_sentence_n_tokens)} vs {token_log_probs.shape[0]}"

        token_log_probs = token_log_probs.squeeze(-1)
        rewards = []

        assert len(target_sentence_n_tokens) == len(prompts)

        max_n_tokens = max(target_sentence_n_tokens) + 1

        if _gamma_factor_values is None or _gamma_factor_values.shape[0] < max_n_tokens:
            _gamma_factor_values = torch.tensor([gamma ** i for i in range(max_n_tokens)]).to(token_log_probs.device)

        # Get the probs only for the target sentence tokens for the valid tokens
        for i, n_tokens in enumerate(target_sentence_n_tokens):
            valid_positions = token_log_probs[i][attention_mask[i, 1:].bool()]

            assert len(valid_positions) >= n_tokens, f"Number of valid tokens must be greater than or equal to target_sentence_n_tokens for input idx {i}: {len(valid_positions)} vs {n_tokens}"

            target_only = valid_positions[-n_tokens:]

            assert len(target_only) == len(_gamma_factor_values[:n_tokens]), f"Length of target_only and _gamma_factor_values must match for input idx {i}: {len(target_only)} vs {len(_gamma_factor_values[:n_tokens])}"

            #target_only = target_only * _gamma_factor_values[:n_tokens]

            if pooling == "target_sentence_probs_mean_reward":
                # exp(log_prob) -> probabilities
                rewards.append((target_only.exp() * _gamma_factor_values[:n_tokens]).sum() / _gamma_factor_values[:n_tokens].sum())
            elif pooling == "target_sentence_neg_ppl_reward":
                value = (target_only * _gamma_factor_values[:n_tokens]).sum() / _gamma_factor_values[:n_tokens].sum()

                if not log_instead_of_ppl:
                    value = (value * -1).exp() * -1

                rewards.append(value)
            else:
                raise ValueError(f"Unknown pooling method: {pooling}")

        pooled_embeddings = torch.stack(rewards, dim=0).unsqueeze(-1)

        assert len(pooled_embeddings.shape) == 2, f"pooled_embeddings expected shape: (batch_size, 1); got: {pooled_embeddings.shape}"
        assert pooled_embeddings.shape == (hidden_states.shape[0], 1), f"pooled_embeddings shape mismatch: {pooled_embeddings.shape} vs {(hidden_states.shape[0], 1)}"
    else:
        raise ValueError(f"Unknown pooling method: {pooling}")

    if pooling in ("none", "features"):
        assert len(pooled_embeddings.shape) == 3, f"pooled_embeddings expected shape: (batch_size, seq_len, dim); got: {pooled_embeddings.shape}"

        if pooling == "none":
            _attention_mask = attention_mask
            expected_shape = hidden_states.shape[1:]
        else:
            _attention_mask = attention_mask[..., 1:]
            expected_shape = (hidden_states.shape[1] - 1, len_features)

        assert pooled_embeddings.shape == (hidden_states.shape[0], *expected_shape), f"pooled_embeddings expected shape (except batch_size): {(hidden_states.shape[0], *expected_shape)}; got: {pooled_embeddings.shape}"

        # Remove padding tokens with attention mask (also flip sequence to be right-to-left, so the first non-padding tokens are the last ones from the prompt, which are more likely to be relevant, and the padding tokens are at the end, which makes it easier for removing them)
        for idx in range(hidden_states.shape[0]):
            n_non_padded_tokens = _attention_mask[idx].sum().item()
            n_padded_tokens = _attention_mask.shape[1] - n_non_padded_tokens

            assert torch.all(_attention_mask[idx, n_padded_tokens:] == 1).item(), f"Non-padded tokens must have attention mask 1 for input idx {idx}: attention_mask: {attention_mask[idx]}"
            assert torch.all(_attention_mask[idx, :n_padded_tokens] == 0).item(), f"Padded tokens must have attention mask 0 for input idx {idx}: attention_mask: {attention_mask[idx]}"

            tmp = torch.zeros(expected_shape, device=pooled_embeddings.device, dtype=pooled_embeddings.dtype)
            tmp[:n_non_padded_tokens, :] = torch.flip(pooled_embeddings[idx, n_padded_tokens:, :], dims=(0,)) # shift left to remove padding and flip sequence to be right-to-left

            assert torch.allclose(torch.flip(pooled_embeddings[idx], dims=(0,)), tmp), f"Flipped pooled_embeddings does not match expected for input idx {idx}: {torch.flip(pooled_embeddings[idx])} vs {tmp}"

            pooled_embeddings[idx] = tmp
    else:
        assert len(pooled_embeddings.shape) == 2, f"pooled_embeddings expected shape: (batch_size, dim); got: {pooled_embeddings.shape}"

    pooled_embeddings = pooled_embeddings.cpu()

    return pooled_embeddings

def compute_features(
    probs,
    shift_targets,
    mask,
    eps=1e-14,
    features=["constant", "observed", "most_likely", "entropy"],
    ):
    features = set(features)
    features_result = {}

    if "constant" in features:
        features.remove("constant")

        # Feature 0: Constant feature (all zeros)
        constant = torch.zeros_like(shift_targets).float()
        constant = constant * mask
        features_result["constant"] = constant.unsqueeze(dim=-1)

    if "observed" in features:
        features.remove("observed")

        # Feature 1: Log probability of the observed token
        observed = torch.log(
                torch.gather(
                    probs, dim=-1, index=shift_targets.unsqueeze(dim=-1)
                ).squeeze(dim=-1)
                + eps
            )
        observed = observed * mask
        features_result["observed"] = observed.unsqueeze(dim=-1)

    if "most_likely" in features:
        features.remove("most_likely")

        # Feature 2: Log probability of the most likely token (according to the model)
        most_likely = torch.log(torch.max(probs, dim=-1).values + eps)
        most_likely = most_likely * mask
        features_result["most_likely"] = most_likely.unsqueeze(dim=-1)

    if "entropy" in features:
        features.remove("entropy")

        # Feature 3: Entropy of the distribution at each position
        entropy = -torch.sum(probs * torch.log2(probs + eps), dim=-1)
        entropy = entropy * mask
        features_result["entropy"] = entropy.unsqueeze(dim=-1)

    if len(features) > 0:
        logger.error("Some features were not computed: %s", features)

    return features_result

def get_token_embedding(token: str, tokenizer, model):
    token_id = tokenizer.convert_tokens_to_ids(token)

    assert token_id is not None, f"Token '{token}' not found in tokenizer vocabulary."

    embedding_matrix = model.model.embed_tokens.weight # embedding matrix (shape: vocab_size, hidden_dim)
    token_embedding = embedding_matrix[token_id].cpu()

    return token_embedding, token_id

def build_prompt(src_sentences, src_lang, trg_lang, tokenizer, icl_examples, _bsz, is_causal_or_chat=None, teacher_forcing=False, add_eos_token=True,
                 lock=None,
                 icl_template="[src_lang]: [source_text]\n[trg_lang]: [translation_text]\n",
                 zs_causal_template="[src_lang]: [source_text]\n[trg_lang]: ",
                 zs_chat_user_template="[src_lang]: [source_text]",
                 zs_chat_response_prefix_template="[trg_lang]: ",
                 zswr_causal_template="[src_lang]: [source_text]\n[trg_lang]: [translation_text]",
                 zswr_chat_user_template="[src_lang]: [source_text]",
                 zswr_chat_response_prefix_template="[trg_lang]: [translation_text]",
                 chat_system_prompt_template="You are a machine translation system that translates sentences from [src_lang] to [trg_lang]. You just respond with the translation, without any additional comments.",
                 user_prefix_template='',
                 compute_src_sentence_n_tokens=False,
):
    if teacher_forcing and icl_examples is None:
        icl_examples = [[] for _ in range(len(src_sentences))]

    assert len(src_sentences) <= _bsz
    assert len(src_sentences) == len(icl_examples), f"{len(src_sentences)} vs {len(icl_examples)}: {src_sentences} vs {icl_examples}"
    assert isinstance(icl_examples, list)

    if is_causal_or_chat is None:
        if "MT_ICL_IS_CAUSAL_OR_CHAT" in os.environ:
            is_causal_or_chat = os.environ["MT_ICL_IS_CAUSAL_OR_CHAT"].strip().lower()

            assert is_causal_or_chat in ("causal", "chat"), "MT_ICL_IS_CAUSAL_OR_CHAT must be either 'causal' or 'chat'"
        else:
            try:
                tokenizer.apply_chat_template([{"role": "system", "content": "foo"}, {"role": "user", "content": "foo"}])
            except:
                is_causal_or_chat = "causal"
            else:
                is_causal_or_chat = "chat"

            logger.warning("is_causal_or_chat is None: using inferred value: %s", is_causal_or_chat)

    for icl_example in icl_examples:
        assert isinstance(icl_example, list)

        if len(icl_example) > 0: # ZS is possible
            for el in icl_example:
                assert len(el) == 2, f"Each icl_example must have exactly two elements: source and target: {len(el)}: {el}"

    prompts = []
    src_sentence_n_tokens = -1
    src_sentence_idx = 0

    while len(prompts) < _bsz and len(prompts) < len(src_sentences):
        _src_lang = src_lang[src_sentence_idx]
        _trg_lang = trg_lang[src_sentence_idx]
        src_sentence = src_sentences[src_sentence_idx]

        if teacher_forcing:
            assert isinstance(src_sentence, list) or isinstance(src_sentence, tuple)
            assert len(src_sentence) == 2, f"Each sentence must have exactly two elements: source and target: {len(src_sentence)}: {src_sentence}"

            _src_sentence = src_sentence[0].strip()
            _trg_sentence = src_sentence[1].strip()
        else:
            _src_sentence = src_sentence.strip()

        if is_causal_or_chat == "chat":
            prompt = []
            system_prompt = str(chat_system_prompt_template)
            system_prompt = system_prompt.replace("[src_lang]", _src_lang)
            system_prompt = system_prompt.replace("[trg_lang]", _trg_lang)
            system_prompt = system_prompt.replace("[source_text]", _src_sentence)
            user_prefix = str(user_prefix_template)
            user_prefix = user_prefix.replace("[src_lang]", _src_lang)
            user_prefix = user_prefix.replace("[trg_lang]", _trg_lang)
            user_prefix = user_prefix.replace("[source_text]", _src_sentence)

            if teacher_forcing:
                system_prompt = system_prompt.replace("[translation_text]", _trg_sentence)
                user_prefix = user_prefix.replace("[translation_text]", _trg_sentence)

            #for icl_src, icl_trg in icl_examples[src_sentence_idx]:
            #    system_prompt += f"\n\nExample instruction: {icl_src}"
            #    system_prompt += f"\n\nTranslate to {_trg_lang}"
            #    system_prompt += f"\n\nExample response:\n\nSure, here's the translation:\n{icl_trg}"

            _prompt = ''
            _prompt += user_prefix

            for icl_src, icl_trg in icl_examples[src_sentence_idx]:
                _prompt2 = str(icl_template)
                _prompt2 = _prompt2.replace("[src_lang]", _src_lang)
                _prompt2 = _prompt2.replace("[trg_lang]", _trg_lang)
                _prompt2 = _prompt2.replace("[source_text]", icl_src)
                _prompt2 = _prompt2.replace("[translation_text]", icl_trg)

                _prompt += _prompt2 #+ '\n' # The user decides the format of the prompt

            if teacher_forcing:
                _prompt2 = str(zswr_chat_user_template)
                _prompt2 = _prompt2.replace("[src_lang]", _src_lang)
                _prompt2 = _prompt2.replace("[trg_lang]", _trg_lang)
                _prompt2 = _prompt2.replace("[source_text]", _src_sentence)
                _prompt2 = _prompt2.replace("[translation_text]", _trg_sentence)
                # response prefix
                _prompt3 = str(zswr_chat_response_prefix_template)
                _prompt3 = _prompt3.replace("[src_lang]", _src_lang)
                _prompt3 = _prompt3.replace("[trg_lang]", _trg_lang)
                _prompt3 = _prompt3.replace("[source_text]", _src_sentence)
                _prompt3 = _prompt3.replace("[translation_text]", _trg_sentence)
            else:
                _prompt2 = str(zs_chat_user_template)
                _prompt2 = _prompt2.replace("[src_lang]", _src_lang)
                _prompt2 = _prompt2.replace("[trg_lang]", _trg_lang)
                _prompt2 = _prompt2.replace("[source_text]", _src_sentence)
                # response prefix
                _prompt3 = str(zs_chat_response_prefix_template)
                _prompt3 = _prompt3.replace("[src_lang]", _src_lang)
                _prompt3 = _prompt3.replace("[trg_lang]", _trg_lang)
                _prompt3 = _prompt3.replace("[source_text]", _src_sentence)

            _prompt += _prompt2

            #prompt.append({"role": "system", "content": system_prompt})
            #prompt.append({"role": "user", "content": f"{_src_sentence}\n\nTranslate to {_trg_lang}"})
            #prompt.append({"role": "assistant", "content": "PLACEHOLDER_PLACEHOLDER"})

            if system_prompt:
                prompt.append({"role": "system", "content": system_prompt})

            prompt.append({"role": "user", "content": _prompt})

            if _prompt3:
                prompt.append({"role": "assistant", "content": "PLACEHOLDER_PLACEHOLDER"})

            prompt = tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True
            )

            if _prompt3:
                placeholder_idx = prompt.find("PLACEHOLDER_PLACEHOLDER")

                assert placeholder_idx != -1
                assert prompt.find("PLACEHOLDER_PLACEHOLDER", placeholder_idx + 1) == -1 # only 1 placeholder

                # Force the initial response of the model
                #if teacher_forcing:
                #    prompt = f"{prompt[:placeholder_idx]}Sure, here's the response: {_trg_sentence}"
                #else:
                #    prompt = f"{prompt[:placeholder_idx]}Sure, here's the response: "

                prompt = f"{prompt[:placeholder_idx]}{_prompt3}"
        elif is_causal_or_chat == "causal":
            user_prefix = str(user_prefix_template)
            user_prefix = user_prefix.replace("[src_lang]", _src_lang)
            user_prefix = user_prefix.replace("[trg_lang]", _trg_lang)
            user_prefix = user_prefix.replace("[source_text]", _src_sentence)

            if teacher_forcing:
                user_prefix = user_prefix.replace("[translation_text]", _trg_sentence)

            _prompt = ''
            _prompt += user_prefix

            for icl_src, icl_trg in icl_examples[src_sentence_idx]:
                _prompt2 = str(icl_template)
                _prompt2 = _prompt2.replace("[src_lang]", _src_lang)
                _prompt2 = _prompt2.replace("[trg_lang]", _trg_lang)
                _prompt2 = _prompt2.replace("[source_text]", icl_src)
                _prompt2 = _prompt2.replace("[translation_text]", icl_trg)

                _prompt += _prompt2 #+ '\n' # The user decides the format of the prompt

            if teacher_forcing:
                _prompt2 = str(zswr_causal_template)
                _prompt2 = _prompt2.replace("[src_lang]", _src_lang)
                _prompt2 = _prompt2.replace("[trg_lang]", _trg_lang)
                _prompt2 = _prompt2.replace("[source_text]", _src_sentence)
                _prompt2 = _prompt2.replace("[translation_text]", _trg_sentence)
            else:
                _prompt2 = str(zs_causal_template)
                _prompt2 = _prompt2.replace("[src_lang]", _src_lang)
                _prompt2 = _prompt2.replace("[trg_lang]", _trg_lang)
                _prompt2 = _prompt2.replace("[source_text]", _src_sentence)

            _prompt += _prompt2
            prompt = _prompt
        else:
            raise Exception(f"Unknown: {is_causal_or_chat}")

        if teacher_forcing and add_eos_token:
            prompt += tokenizer.eos_token

        prompts.append(prompt)

        if compute_src_sentence_n_tokens:
            if lock is not None:
                lock.acquire()

            if teacher_forcing:
                _src_sentence_n_tokens = tokenizer(_src_sentence, add_special_tokens=False, return_tensors="pt").input_ids.shape[-1]
                _trg_sentence_n_tokens = tokenizer(_trg_sentence, add_special_tokens=False, return_tensors="pt").input_ids.shape[-1]
                src_sentence_n_tokens = max(src_sentence_n_tokens, _src_sentence_n_tokens + _trg_sentence_n_tokens)
            else:
                _src_sentence_n_tokens = tokenizer(src_sentence, add_special_tokens=False, return_tensors="pt").input_ids.shape[-1]
                src_sentence_n_tokens = max(src_sentence_n_tokens, _src_sentence_n_tokens)

            if lock is not None:
                lock.release()

        src_sentence_idx += 1

    assert len(prompts) == len(src_sentences[:_bsz])

    return prompts, src_sentence_n_tokens

def main():
    src_lang = sys.argv[1]
    trg_lang = sys.argv[2]
    src_sentences_fn = sys.argv[3]
    model_name = sys.argv[4] if len(sys.argv) > 4 else "meta-llama/Llama-2-7b-hf"  # Default model if not provided
    is_causal_or_chat = sys.argv[5] if len(sys.argv) > 5 else "causal"  # Default to causal if not provided
    bsz = int(sys.argv[6]) if len(sys.argv) > 6 else 8

    assert is_causal_or_chat in ("causal", "chat"), "is_causal_or_chat must be either 'causal' or 'chat'"
    assert bsz > 0, "Batch size must be a positive integer"

    src_sentences = []
    n_original_sentences = 0

    with open(src_sentences_fn, "rt") as fd:
        for l in fd:
            l = l.replace('\t', ' ').replace('\n', ' ').replace('\r', '').strip()

            if len(l) > 0:
                src_sentences.append(l)

            n_original_sentences += 1

    discarded_n_sentences = n_original_sentences - len(src_sentences)
    discarded_n_sentences_perc = discarded_n_sentences * 100 / n_original_sentences

    logger.info("Loaded sentences: %d (discarded: %d, %.2f%%)", len(src_sentences), discarded_n_sentences, discarded_n_sentences_perc)

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = "cuda"
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map=device)

    if tokenizer.pad_token is None:
        # https://github.com/meta-llama/llama3/issues/114#issuecomment-2127131096
        tokenizer.pad_token = tokenizer.eos_token

    # Early stopping criteria
    stop_seqs = ['\n']
    stop_token_seq_ids = [tokenizer.encode(seq, add_special_tokens=False, return_tensors=None) for seq in stop_seqs]

    if model_name in ("meta-llama/Llama-2-7b-hf", "meta-llama/Llama-2-7b-chat-hf"):
        # https://github.com/huggingface/transformers/issues/26273

        for idx in range(len(stop_token_seq_ids)):
            while True:
                try:
                    stop_token_seq_ids[idx].remove(29871)
                except ValueError:
                    break

    stopping_criteria = StoppingCriteriaList([StopOnTokensSeq(stop_token_seq_ids, tokenizer)])

    # Read stdin
    icl_examples = list(map(lambda s: s.split('\t'), sys.stdin.read().splitlines()))

    if len(icl_examples) == 1 and len(icl_examples[0]) == 1 and icl_examples[0][0] == "ZS":
        # zero-shot
        icl_examples = []

        logger.info("Few-shots: 0 (zero-shot)")
    else:
        logger.info("Few-shots: %d", len(icl_examples))

    for idx, l in enumerate(icl_examples):
        assert len(l) == 2, f"Line {idx + 1} should have exactly two columns: source and target: {l}"

        icl_examples[idx] = (l[0].strip(), l[1].strip())

    # Build prompt
    _device = device
    _bsz = bsz

    while True:
        try:
            if model.device != _device:
                model = model.to(_device)

            _src_sentences = src_sentences[:_bsz]
            _icl_examples = [icl_examples for _ in range(len(_src_sentences))]
            _src_lang = [src_lang] * len(_src_sentences)
            _trg_lang = [trg_lang] * len(_src_sentences)

            assert len(_icl_examples) == len(_src_sentences)

            prompts, src_sentence_n_tokens = build_prompt(_src_sentences, _src_lang, _trg_lang, tokenizer, _icl_examples, _bsz, is_causal_or_chat)
            #max_new_tokens = min(1024, src_sentence_n_tokens * 10)
            max_new_tokens = 256

            logger.debug("src_sentence_n_tokens: %d", src_sentence_n_tokens)
            logger.debug("max_new_tokens: %d", max_new_tokens)
            logger.info("Prompts: %s", str(prompts))

            # Translate
            all_outputs, all_original_outputs = translate(model, tokenizer, prompts, max_new_tokens=max_new_tokens, stopping_criteria=stopping_criteria)

            assert len(all_outputs) == len(all_original_outputs) == len(prompts) == len(src_sentences[:_bsz])

            for src_sentence, decoded_output, original_decoded_output in zip(src_sentences[:_bsz], all_outputs, all_original_outputs):
                logger.info("Original output: %s", original_decoded_output)
                print(f"{src_sentence}\t{decoded_output}")

            _device = device
            _bsz = bsz
            src_sentences = src_sentences[len(prompts):]
        except torch.OutOfMemoryError as e:
            # Handle OOM

            if _bsz == 1:
                _device = "cpu"
                _bsz = bsz

                logger.error("torch.OutOfMemoryError error: current batch size is 1: using CPU device and using original batch size: %d", bsz)
            else:
                logger.error("torch.OutOfMemoryError error: current batch size is %d: using smaller batch size: %d", _bsz, _bsz // 2)

                _bsz = _bsz // 2

        if len(src_sentences) == 0:
            break

if __name__ == "__main__":
    main()
