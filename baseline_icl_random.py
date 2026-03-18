
import sys
import json
import base64
import random
import requests

def encode_base64(s):
    return base64.b64encode(s.encode('utf-8')).decode('utf-8')

def batchify(lst, batch_size, icl_examples_pool=None, icl_num_examples=0, avoid_icl_example_equal_to_src=True):
    assert len(icl_examples_pool) > icl_num_examples, f"icl_examples_pool must contain more examples than icl_num_examples (at least, one more) to guarantee that is possible to sample a different src sentence: {len(icl_examples_pool)} vs {icl_num_examples}"

    for i in range(0, len(lst), batch_size):
        icl_examples = []
        bsz = len(lst[i:i+batch_size])

        for j in range(bsz):
            while True:
                _icl_examples = random.sample(icl_examples_pool, icl_num_examples) if icl_examples_pool is not None and icl_num_examples > 0 else []

                if avoid_icl_example_equal_to_src and len(_icl_examples) > 0:
                    src_sentence = lst[i + j].strip()
                    src_icl_examples = {icl_src_example.strip() for icl_src_example, _ in _icl_examples}

                    if src_sentence not in src_icl_examples:
                        break
                else:
                    break

            assert len(_icl_examples) == icl_num_examples, f"Each icl example must have exactly {icl_num_examples} elements, got {len(_icl_examples)}"

            icl_examples.append(_icl_examples)

        yield lst[i:i+batch_size], icl_examples

def main():
    print(f"argv: {sys.argv}", file=sys.stderr)

    src_lang_value = sys.argv[1]
    trg_lang_value = sys.argv[2]
    icl_examples_file = sys.argv[3] # tab-separated file with icl examples with format: "src_sentence\ttrg_sentence"
    icl_num_examples = int(sys.argv[4]) # if 0, zero-shot
    batch_size = int(sys.argv[5]) if len(sys.argv) > 5 else 8
    seed = sys.argv[6] if len(sys.argv) > 6 else None # default random seed
    server_port = sys.argv[7] if len(sys.argv) > 7 else "8000"
    server_name = sys.argv[8] if len(sys.argv) > 8 else "127.0.0.1"

    assert icl_num_examples >= 0, f"icl_num_examples must be non-negative, got: {icl_num_examples}"

    random.seed(seed)

    msg = f"Using {icl_num_examples} ICL examples per sentence"

    if icl_num_examples == 0:
        msg += ": zero-shot setting"

    print(msg, file=sys.stderr)

    # Read from stdin, stripping empty lines
    sentences = [line.replace("\t", " ").replace("\n", " ").replace("\r", " ").strip() for line in sys.stdin]
    icl_examples_pool = []

    with open(icl_examples_file, 'r') as f:
        for line in f:
            line = line.replace("\n", " ").replace("\r", " ").split('\t')

            assert len(line) == 2, f"Each line in the icl examples file must contain exactly two columns: {len(line)} found in line: {line}"

            src, trg = line
            src = src.strip()
            trg = trg.strip()

            icl_examples_pool.append((src, trg))

    assert len(icl_examples_pool) > icl_num_examples, f"icl_examples_pool must contain more examples than icl_num_examples: {len(icl_examples_pool)} vs {icl_num_examples}"

    random.shuffle(icl_examples_pool)

    # Encode each sentence in base64
    url = f"http://{server_name}:{server_port}/translate"
    src_sentences_idx = 0

    for batch, batch_icl_examples in batchify(sentences, batch_size, icl_examples_pool=icl_examples_pool, icl_num_examples=icl_num_examples):
        assert len(batch) == len(batch_icl_examples), f"Batch size mismatch: {len(batch)} vs {len(batch_icl_examples)}"

        payload = []

        for idx, (s, icl_examples) in enumerate(zip(batch, batch_icl_examples), 1):
            assert len(icl_examples) == icl_num_examples, f"Each icl example must have exactly {icl_num_examples} elements, got {len(icl_examples)}"

            s = encode_base64(s)

            payload.append(('src_lang', src_lang_value))
            payload.append(('trg_lang', trg_lang_value))
            payload.append(('src_sentence', s))

            for _icl_examples in icl_examples:
                assert isinstance(_icl_examples, tuple) and len(_icl_examples) == 2, f"Each icl example must be a tuple with exactly two elements: {len(_icl_examples)} found in {_icl_examples}"
                src_icl_example, trg_icl_example = _icl_examples
                src_icl_example = encode_base64(src_icl_example)
                trg_icl_example = encode_base64(trg_icl_example)

                payload.append(('icl_idx_src_sentence', str(idx)))
                payload.append(('src_example', src_icl_example))
                payload.append(('trg_example', trg_icl_example))

        response = requests.post(url, data=payload)

        if response.status_code == 200:
            response_text = json.loads(response.text)

            if response_text["err"] != "null":
                assert response_text["ok"] == "null", f"Error in response (ok): {response_text['ok']}"

                print(f"Error in response: {response_text['err']}", file=sys.stderr)

                for _ in batch:
                    print('') # Print an empty line for each sentence in the batch
            else:
                assert response_text["err"] == "null", f"Error in response (err): {response_text['err']}"
                assert isinstance(response_text["ok"], list), f"Expected 'ok' to be a list, got {type(response_text['ok'])}"
                assert len(response_text["ok"]) == len(batch), f"Length of 'ok' does not match batch size: {len(response_text['ok'])} vs {len(batch)}"

                for src, mt in zip(sentences[src_sentences_idx:src_sentences_idx+len(batch)], response_text["ok"]):
                    mt = mt.replace("\t", " ").replace("\n", " ").replace("\r", " ").strip()

                    print(f"{src}\t{mt}")
        else:
            print(f"Error: Status code {response.status_code}; text: {response.text}", file=sys.stderr)

            for _ in batch:
                print('') # Print an empty line for each sentence in the batch

        src_sentences_idx += len(batch)

if __name__ == "__main__":
    main()
