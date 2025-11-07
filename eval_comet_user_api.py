
import sys
import json
import base64

import requests

def encode_base64(s):
    return base64.b64encode(s.encode('utf-8')).decode('utf-8')

def batchify(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        bsz = len(lst[i:i+batch_size])

        yield lst[i:i+batch_size]

def eval(src, mt, ref, batch_size=8, server_url="http://127.0.0.1:8000"):
    assert isinstance(src, list), "Source should be a list"
    assert isinstance(mt, list), "MT should be a list"
    assert isinstance(ref, list), "Reference should be a list"
    assert len(src) == len(mt) == len(ref), "Source, MT, and Reference lists must have the same length"

    server_url = server_url.rstrip('/')
    url = f"{server_url}/evaluate_comet_22"
    scores = []
    data = [{"src": encode_base64(s), "mt": encode_base64(m), "ref": encode_base64(r)} for s, m, r in zip(src, mt, ref)]

    for idx, batch in enumerate(batchify(data, batch_size)):
        payload = []

        for sample in batch:
            payload.append(('src_sentence', sample["src"]))
            payload.append(('mt_sentence', sample["mt"]))
            payload.append(('ref_sentence', sample["ref"]))

        response = requests.post(url, data=payload)
        if response.text:
            response_text = json.loads(response.text)

            if response_text["err"] != "null":
                print(f"Response error: {response_text['err']}")
            else:
                response_result = [float(d) for d in response_text["ok"]]

                scores.extend(response_result)
        else:
            print(f"ERROR: {idx}")

    avg = sum(scores) / len(scores)

    return avg, scores

def main():
    batch_size = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    server_url = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8000"

    ##
    src = []
    mt = []
    ref = []

    for l in sys.stdin:
        l = l.rstrip("\r\n").split('\t')

        assert len(l) == 3
        assert l[0].strip() != "", "Source sentence is empty"
        assert l[1].strip() != "", "MT sentence is empty"
        assert l[2].strip() != "", "Reference sentence is empty"

        src.append(l[0].strip())
        mt.append(l[1].strip())
        ref.append(l[2].strip())

    avg, scores = eval(src, mt, ref, batch_size=batch_size, server_url=server_url)

    print(f"Average score: {avg}")

if __name__ == "__main__":
    main()
