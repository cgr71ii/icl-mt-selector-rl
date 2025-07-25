
import sys
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers import StoppingCriteria, StoppingCriteriaList

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

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

        logging.debug("early stopping tokens: %s", self.stop_token_token)

    def __call__(self, input_ids, scores, **kwargs):
        # input_ids is a tensor of shape (batch_size * beam_size, sequence_length), including the prompt

        assert len(input_ids.shape) == 2

        #logging.debug("input_ids.shape: %s", input_ids.shape)

        stop = [False] * input_ids.shape[0]

        for idx in range(len(input_ids)):
            for stop_token_id in self.stop_token_ids:
                #logging.debug("stop_token_id: %s; tuple(input_ids[idx, -len(stop_token_id):].tolist()): %s", stop_token_id, tuple(input_ids[idx, -len(stop_token_id):].tolist()))

                if len(input_ids[idx]) < len(stop_token_id):
                    continue

                if tuple(input_ids[idx, -len(stop_token_id):].tolist()) == stop_token_id:
                    stop[idx] = True

                    if len(set(stop)) == 1: # all values must be True
                        return True

        return False

def translate(model, tokenizer, prompts, src_lang, trg_lang, is_causal_or_chat, max_new_tokens=1024, stopping_criteria=None, normalize=True):
    all_outputs, all_original_outputs = [], []

    # Tokenize
    inputs = tokenizer(prompts, padding="longest", return_tensors="pt", padding_side="left").to(model.device)

    # Generate with beam search
    # Decoding: https://aclanthology.org/2024.emnlp-main.489/
    output = model.generate(
        **inputs,
        #max_new_tokens=1024,
        max_new_tokens=max_new_tokens,
        num_beams=4,
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
    decoded_outputs = []
    _decoded_outputs = [tokenizer.decode(output[idx][inputs.input_ids[idx].shape[-1]:], skip_special_tokens=True) for idx in range(output.shape[0])]

    #print("\n=== TRANSLATION ===")

    for idx in range(len(_decoded_outputs)):
        decoded_output = _decoded_outputs[idx]
        decoded_output = decoded_output.strip().split('\n')[0]
        decoded_output = decoded_output
        original_decoded_output = str(_decoded_outputs[idx])

        if normalize:
            decoded_output = decoded_output.replace('\t', ' ').replace('\n', ' ').replace('\r', '').strip()
            original_decoded_output = original_decoded_output.replace('\t', r' \t ').replace('\n', r' \n ').replace('\r', '').strip()

        all_outputs.append(decoded_output)
        all_original_outputs.append(original_decoded_output)

    return all_outputs, all_original_outputs

def get_embedding_mean_pooling(model, tokenizer, prompts):
    # Tokenize
    inputs = tokenizer(prompts, padding="longest", return_tensors="pt", padding_side="left").to(model.device)
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)

    # Forward pass with hidden states
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1] # shape: (batch_size, seq_len, hidden_dim)

        assert len(hidden_states.shape) == 3, f"hidden_states expected shape: (batch_size, seq_len, hidden_dim); got: {hidden_states.shape}"

    # Mean pooling
    attention_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    sum_embeddings = torch.sum(hidden_states * attention_mask_expanded, dim=1)
    sum_mask = attention_mask_expanded.sum(dim=1)
    mean_pooled_embeddings = sum_embeddings / sum_mask
    mean_pooled_embeddings = mean_pooled_embeddings.cpu()

    return mean_pooled_embeddings

def get_token_embedding(token: str, tokenizer, model):
    token_id = tokenizer.convert_tokens_to_ids(token)

    assert token_id is not None, f"Token '{token}' not found in tokenizer vocabulary."

    embedding_matrix = model.model.embed_tokens.weight # embedding matrix (shape: vocab_size, hidden_dim)
    token_embedding = embedding_matrix[token_id].cpu()

    return token_embedding, token_id

def build_prompt(src_sentences, src_lang, trg_lang, tokenizer, icl_examples, _bsz, is_causal_or_chat, teacher_forcing=False, add_eos_token=True,
                 icl_template="[src_lang]: [source_text]\n[trg_lang]: [translation_text]",
                 zs_template="[src_lang]: [source_text]\n[trg_lang]: ",
                 zswr_template="[src_lang]: [source_text]\n[trg_lang]: [translation_text]"):
    assert len(src_sentences) <= _bsz
    assert len(src_sentences) == len(icl_examples), f"{len(src_sentences)} vs {len(icl_examples)}: {src_sentences} vs {icl_examples}"
    assert isinstance(icl_examples, list)

    if teacher_forcing and icl_examples is None:
        icl_examples = [[] for _ in range(len(src_sentences))]

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
            #system_prompt = f"You are a machine translation system that translates sentences from {_src_lang} to {_trg_lang}. You just respond with the translation, without any additional comments."

            #for icl_src, icl_trg in icl_examples[src_sentence_idx]:
            #    system_prompt += f"\n\nExample instruction: {icl_src}"
            #    system_prompt += f"\n\nTranslate to {_trg_lang}"
            #    system_prompt += f"\n\nExample response:\n\nSure, here's the translation:\n{icl_trg}"

            _prompt = ''

            for icl_src, icl_trg in icl_examples[src_sentence_idx]:
                _prompt2 = str(icl_template)
                _prompt2 = _prompt2.replace("[src_lang]", _src_lang)
                _prompt2 = _prompt2.replace("[trg_lang]", _trg_lang)
                _prompt2 = _prompt2.replace("[source_text]", _src_sentence)
                _prompt2 = _prompt2.replace("[translation_text]", _trg_sentence)

                _prompt += _prompt2 + '\n'

            if teacher_forcing:
                _prompt2 = str(zswr_template)
                _prompt2 = _prompt2.replace("[src_lang]", _src_lang)
                _prompt2 = _prompt2.replace("[trg_lang]", _trg_lang)
                _prompt2 = _prompt2.replace("[source_text]", _src_sentence)
                _prompt2 = _prompt2.replace("[translation_text]", _trg_sentence)
            else:
                _prompt2 = str(zs_template)
                _prompt2 = _prompt2.replace("[src_lang]", _src_lang)
                _prompt2 = _prompt2.replace("[trg_lang]", _trg_lang)
                _prompt2 = _prompt2.replace("[source_text]", _src_sentence)

            _prompt += _prompt2

            #prompt.append({"role": "system", "content": system_prompt})
            #prompt.append({"role": "user", "content": f"{_src_sentence}\n\nTranslate to {_trg_lang}"})
            #prompt.append({"role": "assistant", "content": "PLACEHOLDER_PLACEHOLDER"})

            prompt.append({"role": "user", "content": _prompt})

            prompt = tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True
            )

            #placeholder_idx = prompt.find("PLACEHOLDER_PLACEHOLDER")

            #assert placeholder_idx != -1
            #assert prompt.find("PLACEHOLDER_PLACEHOLDER", placeholder_idx + 1) == -1 # only 1 placeholder

            # Force the initial response of the model
            #if teacher_forcing:
            #    prompt = f"{prompt[:placeholder_idx]}Sure, here's the translation:\n{_trg_sentence}"
            #else:
            #    prompt = f"{prompt[:placeholder_idx]}Sure, here's the translation:\n"
        elif is_causal_or_chat == "causal":
            _prompt = ''

            for icl_src, icl_trg in icl_examples[src_sentence_idx]:
                _prompt2 = str(icl_template)
                _prompt2 = _prompt2.replace("[src_lang]", _src_lang)
                _prompt2 = _prompt2.replace("[trg_lang]", _trg_lang)
                _prompt2 = _prompt2.replace("[source_text]", _src_sentence)
                _prompt2 = _prompt2.replace("[translation_text]", _trg_sentence)

                _prompt += _prompt2 + '\n'

            if teacher_forcing:
                _prompt2 = str(zswr_template)
                _prompt2 = _prompt2.replace("[src_lang]", _src_lang)
                _prompt2 = _prompt2.replace("[trg_lang]", _trg_lang)
                _prompt2 = _prompt2.replace("[source_text]", _src_sentence)
                _prompt2 = _prompt2.replace("[translation_text]", _trg_sentence)
            else:
                _prompt2 = str(zs_template)
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

        if teacher_forcing:
            _src_sentence_n_tokens = tokenizer(_src_sentence, add_special_tokens=False, return_tensors="pt").input_ids.shape[-1]
            _trg_sentence_n_tokens = tokenizer(_trg_sentence, add_special_tokens=False, return_tensors="pt").input_ids.shape[-1]
            src_sentence_n_tokens = max(src_sentence_n_tokens, _src_sentence_n_tokens + _trg_sentence_n_tokens)
        else:
            _src_sentence_n_tokens = tokenizer(src_sentence, add_special_tokens=False, return_tensors="pt").input_ids.shape[-1]
            src_sentence_n_tokens = max(src_sentence_n_tokens, _src_sentence_n_tokens)

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

    logging.info("Loaded sentences: %d (discarded: %d, %.2f%%)", len(src_sentences), discarded_n_sentences, discarded_n_sentences_perc)

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

        logging.info("Few-shots: 0 (zero-shot)")
    else:
        logging.info("Few-shots: %d", len(icl_examples))

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
            max_new_tokens = 512

            logging.debug("src_sentence_n_tokens: %d", src_sentence_n_tokens)
            logging.debug("max_new_tokens: %d", max_new_tokens)
            logging.info("Prompts: %s", str(prompts))

            # Translate
            all_outputs, all_original_outputs = translate(model, tokenizer, prompts, src_lang, trg_lang, is_causal_or_chat, max_new_tokens=max_new_tokens, stopping_criteria=stopping_criteria)

            assert len(all_outputs) == len(all_original_outputs) == len(prompts) == len(src_sentences[:_bsz])

            for src_sentence, decoded_output, original_decoded_output in zip(src_sentences[:_bsz], all_outputs, all_original_outputs):
                logging.info("Original output: %s", original_decoded_output)
                print(f"{src_sentence}\t{decoded_output}")

            _device = device
            _bsz = bsz
            src_sentences = src_sentences[len(prompts):]
        except torch.OutOfMemoryError as e:
            # Handle OOM

            if _bsz == 1:
                _device = "cpu"
                _bsz = bsz

                logging.error("torch.OutOfMemoryError error: current batch size is 1: using CPU device and using original batch size: %d", bsz)
            else:
                logging.error("torch.OutOfMemoryError error: current batch size is %d: using smaller batch size: %d", _bsz, _bsz // 2)

                _bsz = _bsz // 2

        if len(src_sentences) == 0:
            break

if __name__ == "__main__":
    main()
