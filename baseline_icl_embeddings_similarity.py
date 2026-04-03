
import sys
import json
import base64
import random
import requests

import embeddings as embeddings_utils

import numpy as np

def encode_base64(s):
    return base64.b64encode(s.encode('utf-8')).decode('utf-8')

def batchify(sentences, batch_size, sentences_embeddings=None, icl_examples_pool=None, icl_examples_pool_embeddings=None, icl_num_examples=0, avoid_icl_example_equal_to_src=True):
    assert sentences_embeddings is not None, "sentences_embeddings must be provided"
    assert icl_examples_pool is not None, "icl_examples_pool must be provided"
    assert len(icl_examples_pool) > icl_num_examples, f"icl_examples_pool must contain more examples than icl_num_examples (at least, one more) to guarantee that is possible to sample a different src sentence: {len(icl_examples_pool)} vs {icl_num_examples}"
    assert isinstance(sentences, list), f"sentences must be a list, got {type(sentences)}"
    assert isinstance(sentences_embeddings, np.ndarray), f"sentences_embeddings must be a numpy array, got {type(sentences_embeddings)}"
    assert sentences_embeddings.shape[0] == len(sentences), f"sentences_embeddings first dimension must be equal to the number of sentences: {sentences_embeddings.shape[0]} vs {len(sentences)}"
    assert len(sentences_embeddings.shape) == 2, f"sentences_embeddings must be 2D, got {len(sentences_embeddings.shape)}D: {sentences_embeddings.shape}"
    assert isinstance(icl_examples_pool, list), f"icl_examples_pool must be a list, got {type(icl_examples_pool)}"
    assert isinstance(icl_examples_pool_embeddings, np.ndarray), f"icl_examples_pool_embeddings must be a numpy array, got {type(icl_examples_pool_embeddings)}"
    assert icl_examples_pool_embeddings.shape[0] == len(icl_examples_pool), f"icl_examples_pool_embeddings first dimension must be equal to the number of icl examples: {icl_examples_pool_embeddings.shape[0]} vs {len(icl_examples_pool)}"
    assert len(icl_examples_pool_embeddings.shape) == 2, f"icl_examples_pool_embeddings must be 2D, got {len(icl_examples_pool_embeddings.shape)}D: {icl_examples_pool_embeddings.shape}"

    similarity_matrix = embeddings_utils.get_similarity(sentences_embeddings, icl_examples_pool_embeddings, metric="cosine")

    assert similarity_matrix.shape[0] == len(sentences), f"similarity_matrix first dimension must be equal to the number of sentences: {similarity_matrix.shape[0]} vs {len(sentences)}"
    assert similarity_matrix.shape[1] == len(icl_examples_pool), f"similarity_matrix second dimension must be equal to the number of icl examples: {similarity_matrix.shape[1]} vs {len(icl_examples_pool)}"
    assert len(similarity_matrix.shape) == 2, f"similarity_matrix must be 2D, got {len(similarity_matrix.shape)}D: {similarity_matrix.shape}"

    for i in range(0, len(sentences), batch_size):
        icl_examples = []
        bsz = len(sentences[i:i+batch_size])

        for j in range(bsz):
            src_sentence = sentences[i + j].strip()
            _icl_examples = []

            if icl_examples_pool is not None and icl_num_examples > 0:
                # get top-n examples with highest similarity
                scores = similarity_matrix[i + j].tolist()

                assert len(scores) == len(icl_examples_pool), f"Scores length mismatch: {len(scores)} vs {len(icl_examples_pool)}"
                assert isinstance(scores[0], (float, np.floating)), f"Each score must be a float, got {type(scores[0])}"

                top_n_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)

                for idx in top_n_indices:
                    if len(_icl_examples) >= icl_num_examples:
                        break

                    src_icl_example = icl_examples_pool[idx][0].strip()

                    if avoid_icl_example_equal_to_src and src_icl_example == src_sentence:
                        continue

                    _icl_examples.append(icl_examples_pool[idx])

            assert len(_icl_examples) == icl_num_examples, f"Each icl example must have exactly {icl_num_examples} elements, got {len(_icl_examples)}"

            icl_examples.append(_icl_examples)

        yield sentences[i:i+batch_size], icl_examples

def main():
    print(f"argv: {sys.argv}", file=sys.stderr)

    src_lang_value = sys.argv[1] # e.g., "English" -> flores format -> eng_Latn
    trg_lang_value = sys.argv[2]
    icl_examples_file = sys.argv[3] # tab-separated file with icl examples with format: "src_sentence\ttrg_sentence"
    icl_num_examples = int(sys.argv[4]) # if 0, zero-shot
    batch_size, batch_size_embeddings = tuple(map(int, sys.argv[5].split(':'))) if len(sys.argv) > 5 else (8, 8)
    seed = int(sys.argv[6]) if len(sys.argv) > 6 else None # default random seed
    server_port = sys.argv[7] if len(sys.argv) > 7 else "8000"
    server_name = sys.argv[8] if len(sys.argv) > 8 else "127.0.0.1"
    embeddings_name = sys.argv[9] if len(sys.argv) > 9 else "SONAR"

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

    sentences_embeddings = embeddings_utils.get_embeddings(embeddings_name, sentences, src_lang_value, batch_size=batch_size_embeddings)
    icl_examples_pool_embeddings = embeddings_utils.get_embeddings(embeddings_name, [_icl_examples_pool[0] for _icl_examples_pool in icl_examples_pool], src_lang_value, batch_size=batch_size_embeddings)

    # Encode each sentence in base64
    url = f"http://{server_name}:{server_port}/translate"
    src_sentences_idx = 0

    for batch, batch_icl_examples in batchify(sentences, batch_size, sentences_embeddings=sentences_embeddings, icl_examples_pool=icl_examples_pool, icl_examples_pool_embeddings=icl_examples_pool_embeddings, icl_num_examples=icl_num_examples):
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
