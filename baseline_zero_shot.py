
import sys
import json
import base64
import requests

def encode_base64(s):
    return base64.b64encode(s.encode('utf-8')).decode('utf-8')

def batchify(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i:i+batch_size]

def main():
    print(f"argv: {sys.argv}", file=sys.stderr)

    src_lang_value = sys.argv[1]
    trg_lang_value = sys.argv[2]
    batch_size = int(sys.argv[3]) if len(sys.argv) > 3 else 8

    # Read from stdin, stripping empty lines
    sentences = [line.replace("\t", " ").replace("\n", " ").replace("\r", " ").strip() for line in sys.stdin]

    # Encode each sentence in base64
    encoded_sentences = [encode_base64(s) for s in sentences]
    url = "http://127.0.0.1:8000/translate"
    src_sentences_idx = 0

    for batch in batchify(encoded_sentences, batch_size):
        payload = []

        for s in batch:
            payload.append(('src_lang', src_lang_value))
            payload.append(('trg_lang', trg_lang_value))
            payload.append(('src_sentence', s))

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
