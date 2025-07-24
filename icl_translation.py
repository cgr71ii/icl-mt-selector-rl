
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

#prompt_example_split_char = ' '
prompt_example_split_char = '\n'

def translate(model, tokenizer, prompts, src_lang, trg_lang, is_causal_or_chat, max_new_tokens=1024, stopping_criteria=None):
    all_outputs, all_original_outputs = [], []

    # Tokenize
    inputs = tokenizer(prompts, padding="longest", return_tensors="pt", padding_side="left").to(model.device)

    # Generate with beam search
    # Decoding: https://proceedings.mlr.press/v202/garcia23a.html
    output = model.generate(
        **inputs,
        #max_new_tokens=1024,
        max_new_tokens=max_new_tokens,
        num_beams=4,
        length_penalty=0.6,
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

#        if is_causal_or_chat == "causal":
#            decoded_output_split_idx = decoded_output.find(f"{prompt_example_split_char}{src_lang}")
#
#            if prompt_example_split_char == '\n':
#                decoded_output = decoded_output.strip().split('\n')[0]
#            elif src_lang in decoded_output and decoded_output_split_idx != -1:
#                decoded_output = decoded_output[:decoded_output_split_idx]
#            else:
#                logging.warning("The output does not contain the expected language prefix: printing all the generated content")
#        elif is_causal_or_chat == "chat":
#            decoded_output = decoded_output.strip().split('\n')[0]
#        else:
#            raise Exception(f"Unexpected: {is_causal_or_chat}")
##
        decoded_output = decoded_output.strip().split('\n')[0]
##
        decoded_output = decoded_output
        original_decoded_output = str(_decoded_outputs[idx])

        all_outputs.append(decoded_output)
        all_original_outputs.append(original_decoded_output)

    return all_outputs, all_original_outputs

def main():
    global prompt_example_split_char

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
    #stop_seqs = ["<|stop|>", "###END###"]
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

        if is_causal_or_chat == "causal" and prompt_example_split_char != '\n':
            logging.warning("prompt_example_split_char=' ', but for the selected configuration is a better idea to set prompt_example_split_char='\\n' in order to split correctly the output translation. Using the latter configuration")

            prompt_example_split_char = '\n'
    else:
        logging.info("Few-shots: %d", len(icl_examples))

    for idx, l in enumerate(icl_examples):
        assert len(l) == 2, f"Line {idx + 1} should have exactly two columns: source and target: {l}"

    # Build prompt: https://proceedings.mlr.press/v202/zhang23m
    # Build prompt: https://aclanthology.org/2024.findings-naacl.176/
    _device = device
    _bsz = bsz

    while True:
        try:
            if model.device != _device:
                model = model.to(_device)

            prompts = []
            src_sentence_n_tokens = -1
            src_sentence_idx = 0

            while len(prompts) < _bsz and len(prompts) < len(src_sentences):
                src_sentence = src_sentences[src_sentence_idx].strip()

                if is_causal_or_chat == "causal":
                    #prompt = ''.join([f"{src_lang}: {icl_src}{prompt_example_split_char}{trg_lang}: {icl_trg}{prompt_example_split_char}" for icl_src, icl_trg in icl_examples])
                    #prompt += f"{src_lang}: {src_sentence}{prompt_example_split_char}{trg_lang}: "
                    prompt = ''.join([f"{icl_src}={icl_trg}{prompt_example_split_char}" for icl_src, icl_trg in icl_examples])
                    prompt += f"{src_sentence}="
                elif is_causal_or_chat == "chat":
                    # Partially combined with https://aclanthology.org/2024.eacl-short.4/
                    prompt = []
                    system_prompt = f"You are a machine translation system that translates sentences from {src_lang} to {trg_lang}. You just respond with the translation, without any additional comments."

                    if len(icl_examples) > 0:
                        #system_prompt += "\n\n"
                        system_prompt += "\n"

                    for icl_src, icl_trg in icl_examples:
                        #system_prompt += f"\nExample instruction and response: {src_lang}: {icl_src}{prompt_example_split_char}{trg_lang}: {icl_trg}{prompt_example_split_char}"
                        system_prompt += f"\nExample instruction and response: {icl_src}={icl_trg}{prompt_example_split_char}"

                    prompt.append({"role": "system", "content": system_prompt})
                    #prompt.append({"role": "user", "content": f"{src_lang}: {src_sentence}{prompt_example_split_char}{trg_lang}: "})
                    #prompt.append({"role": "user", "content": f"{src_lang}: {src_sentence}{prompt_example_split_char}"})
                    prompt.append({"role": "user", "content": f"{src_sentence}"})
                    prompt.append({"role": "assistant", "content": "PLACEHOLDER_PLACEHOLDER"})

                    prompt = tokenizer.apply_chat_template(
                        prompt,
                        tokenize=False,
                        add_generation_prompt=True
                    )

                    placeholder_idx = prompt.find("PLACEHOLDER_PLACEHOLDER")

                    assert placeholder_idx != -1
                    assert prompt.find("PLACEHOLDER_PLACEHOLDER", placeholder_idx + 1) == -1 # only 1 placeholder

                    # Force the initial response of the model
                    #prompt = f"{prompt[:placeholder_idx]}{trg_lang}: "
                    prompt = f"{prompt[:placeholder_idx]}="
                else:
                    raise Exception(f"Unexpected: {is_causal_or_chat}")

                prompts.append(prompt)
            
                _src_sentence_n_tokens = tokenizer(src_sentence, add_special_tokens=False, return_tensors="pt").input_ids.shape[-1]
                src_sentence_n_tokens = max(src_sentence_n_tokens, _src_sentence_n_tokens)
                src_sentence_idx += 1

            assert len(prompts) == len(src_sentences[:_bsz])

            max_new_tokens = min(1024, src_sentence_n_tokens * 10)

            logging.debug("src_sentence_n_tokens: %d", src_sentence_n_tokens)
            logging.debug("max_new_tokens: %d", max_new_tokens)
            logging.info("Prompts: %s", str(prompts))

            # Translate
            all_outputs, all_original_outputs = translate(model, tokenizer, prompts, src_lang, trg_lang, is_causal_or_chat, max_new_tokens=max_new_tokens, stopping_criteria=stopping_criteria)
            all_outputs = list(map(lambda s: s.replace('\n', r" \n ").replace('\t', r" \t ").strip(), all_outputs))
            all_original_outputs = list(map(lambda s: s.replace('\n', r" \n ").replace('\t', r" \t ").strip(), all_original_outputs))

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
