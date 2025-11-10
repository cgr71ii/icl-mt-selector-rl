
import os
import json
import base64
import pickle
import logging
import inspect
import argparse
from threading import Lock

import utils
import icl_translation as mt_icl

import torch
import numpy as np
from flask import (
    Flask,
    request,
    jsonify,
)
import faiss

app = Flask("MT-ICL-knn-flask-server")
global_conf = {} # Empty since it will be filled once main is run
logger = utils.set_up_logging_logger(logging.getLogger("MT_ICL.knn_flask_server"), level=logging.INFO)

# Disable (less verbose) 3rd party logging
logging.getLogger("werkzeug").setLevel(logging.WARNING)

@app.route('/', methods=['GET'])
def info():
    available_routes = json.dumps(
        {
            "/hello-world": ["GET"],
            "/reset_index": ["GET", "POST"],
            "/insert": ["GET", "POST"],
            "/retrieve": ["GET", "POST"],
        },
        indent=4).replace('\n', '<br/>').replace(' ', '&nbsp;')

    return f"Available routes:<br/>{available_routes}"

@app.route('/hello-world', methods=["GET"])
def hello_world():
    return jsonify({"ok": "hello world! server is working!", "err": "null"})

@app.route('/reset_index', methods=["GET", "POST"])
def reset_index():
    if request.method not in ("GET", "POST"):
        return jsonify({"ok": "null", "err": "method is not: GET, POST"})

    if request.method == "GET":
        # GET method should be used only for testing purposes since HTML encoding is not being handled
        request_method = request.args
    elif request.method == "POST":
        request_method = request.form
    else:
        logger.error("Unknown method: %s", request.method)

        return jsonify({"ok": "null", "err": f"unknown method: {request.method}"})

    # Get parameters
    try:
        dim = utils.string2list(request_method.getlist("dimensionality"))
    except KeyError as e:
        logger.error("KeyError: %s", e)

        return jsonify({"ok": "null", "err": f"could not get some mandatory field: 'urls' are mandatory"})

    if len(dim) != 1:
        logger.error("Unexpected dimensionality: %s", dim)

        return jsonify({"ok": "null", "err": f"'dimensionality' length: {dim}"})

    dim = int(dim[0])

    if len(global_conf["representation"]) > 0 or len(global_conf["representation_item2idx"]) > 0:
        logger.error("Could not reset the embedding index as there were already data inserted")

        return jsonify({"ok": "null", "err": f"Could not reset the embedding index as there were already data inserted"})

    global_conf["dim"] = dim
    global_conf["embedding_index"] = faiss.IndexFlatL2(global_conf["dim"])

    return jsonify({
        "ok": "ok",
        "err": "null",
    })

@app.route('/insert', methods=["GET", "POST"])
def insert_embedding():
    if request.method not in ("GET", "POST"):
        return jsonify({"ok": "null", "err": "method is not: GET, POST"})

    if request.method == "GET":
        # GET method should be used only for testing purposes since HTML encoding is not being handled
        request_method = request.args
    elif request.method == "POST":
        request_method = request.form
    else:
        logger.error("Unknown method: %s", request.method)

        return jsonify({"ok": "null", "err": f"unknown method: {request.method}"})

    # Get parameters
    try:
        src_sentence = utils.string2list(request_method.getlist("src_sentence"))
        embedding = utils.string2list(request_method.getlist("embedding"))
    except KeyError as e:
        logger.error("KeyError: %s", e)

        return jsonify({"ok": "null", "err": f"could not get some mandatory field: 'urls' are mandatory"})

    # Optional parameters
    try:
        check_l2_norm = utils.string2list(request_method.getlist("check_l2_norm"))

        if len(set(check_l2_norm)) != 1:
            logger.error("Different values: %s", set(check_l2_norm))

            return jsonify({"ok": "null", "err": f"Different values: {set(check_l2_norm)}"})
        else:
            check_l2_norm = bool(int(str(check_l2_norm[0])))
    except KeyError as e:
        check_l2_norm = True

    if len(embedding) == 0:
        logger.error("No embeddings: %s", embedding)

        return jsonify({"ok": "null", "err": "'embedding' is mandatory field that cannot be empty"})

    if len(embedding) != len(src_sentence):
        logger.error("Different sizes: %d vs %d", len(embedding), len(src_sentence))

        return jsonify({"ok": "null", "err": f"Different sizes: {len(embedding)} vs {len(src_sentence)}"})

    initial_n = len(embedding)
    str2representation = global_conf["str2representation"]
    skip_idxs = {idx for idx, _str in enumerate(src_sentence) if _str in str2representation}
    embedding = [base64.b64decode(s) for s in embedding] # base64 tensor representation
    embedding = [pickle.loads(e) for e in embedding] # transform to tensor from pickle serialization

    for _str, e in zip(src_sentence, embedding):
        if not isinstance(e, np.ndarray):
            return jsonify({"ok": "null", "err": f"Not np array: {type(e)}"})

        if len(e) != global_conf["dim"]:
            return jsonify({"ok": "null", "err": f"Expected len(shape) to be {global_conf['dim']}, got {len(e)}"})

    src_sentence = [_str for idx, _str in enumerate(src_sentence) if idx not in skip_idxs]
    embedding = [e for idx, e in enumerate(embedding) if idx not in skip_idxs]

    for _str, e in zip(src_sentence, embedding):
        assert _str not in str2representation

        str2representation[_str] = e

    logger.debug("Got %d embeddings: inserting %d", initial_n, len(embedding))

    if len(embedding) > 0:
        embedding = np.array(embedding, dtype=np.float32)

        assert len(embedding.shape) == 2, embedding.shape
        assert embedding.shape[1] == global_conf["dim"], f"{embedding.shape[1]} vs {global_conf['dim']}"

        # Apply
        knn_insert_embedding(src_sentence, embedding, check_l2_norm=check_l2_norm)

    return jsonify({
        "ok": "ok",
        "err": "null",
    })

def knn_insert_embedding(urls, embeddings, check_l2_norm=True):
    assert len(embeddings.shape) == 2
    assert embeddings.shape[-1] == global_conf["dim"]
    assert isinstance(urls, list), f"Expected urls to be a list, got {type(urls)}: {urls}"
    assert len(urls) > 0, "urls must not be an empty list"
    assert isinstance(urls[0], str), f"Expected urls to be a list of strings, got {type(urls[0])}: {urls[0]}"

    #embeddings = utils.embeddings_index_sanity_check(embeddings, last_dimmension_shape=action_dim)
    dim = global_conf["dim"]
    index = global_conf["embedding_index"]
    urls_representation = global_conf["representation"]
    urls_representation_url2idx = global_conf["representation_item2idx"]
    debug = global_conf["debug"]
    total = index.ntotal

    if debug:
        logger.debug("insert.embedding (first element): %s", embeddings[0])

    utils.insert_embeddings(urls, embeddings, index, urls_representation, urls_representation_url2idx, dim, update_representation=True, check_l2_norm=check_l2_norm)

    total2 = index.ntotal

    logger.debug("Index updated: %d -> %d", total, total2)

    assert total + embeddings.shape[0] == total2, f"total + n != total2: {total} + {embeddings.shape[0]} != {total2}"

@app.route('/retrieve', methods=["GET", "POST"])
def retrieve():
    if request.method not in ("GET", "POST"):
        return jsonify({"ok": "null", "err": "method is not: GET, POST"})

    if request.method == "GET":
        # GET method should be used only for testing purposes since HTML encoding is not being handled
        request_method = request.args
    elif request.method == "POST":
        request_method = request.form
    else:
        logger.error("Unknown method: %s", request.method)

        return jsonify({"ok": "null", "err": f"unknown method: {request.method}"})

    # Get parameters
    try:
        embedding = utils.string2list(request_method.getlist("embedding"))
        k = utils.string2list(request_method.getlist("k"))
        get_representations_instead_of_embeddings = utils.string2list(request_method.getlist("get_representations_instead_of_embeddings"))
    except KeyError as e:
        logger.error("KeyError: %s", e)

        return jsonify({"ok": "null", "err": f"could not get some mandatory field: 'urls' are mandatory"})

    # Optional parameters
    try:
        check_l2_norm = utils.string2list(request_method.getlist("check_l2_norm"))

        if len(set(check_l2_norm)) != 1:
            logger.error("Different values: %s", set(check_l2_norm))

            return jsonify({"ok": "null", "err": f"Different values: {set(check_l2_norm)}"})
        else:
            check_l2_norm = bool(int(str(check_l2_norm[0])))
    except KeyError as e:
        check_l2_norm = True

    if len(embedding) != 1:
        logger.error("Different values: %s", len(embedding))

        return jsonify({"ok": "null", "err": f"Different values: {len(embedding)}"})

    if len(set(get_representations_instead_of_embeddings)) != 1:
        logger.error("Different values: %s", set(get_representations_instead_of_embeddings))

        return jsonify({"ok": "null", "err": f"Different values: {set(get_representations_instead_of_embeddings)}"})

    if len(set(k)) != 1:
        logger.error("Different values: %s", set(k))

        return jsonify({"ok": "null", "err": f"Different values: {set(k)}"})

    get_representations_instead_of_embeddings = bool(int(get_representations_instead_of_embeddings[0]))
    k = int(k[0])
    embedding = embedding[0]
    embedding = base64.b64decode(embedding) # base64 tensor representation
    embedding = pickle.loads(embedding) # transform to tensor from pickle serialization

    if embedding.shape[-1] != global_conf["dim"]:
        return jsonify({"ok": "null", "err": f"Expected shape[-1] to be {global_conf['dim']}, got {embedding.shape[-1]}"})

    if not isinstance(embedding, np.ndarray) and not isinstance(embedding, torch.Tensor):
        return jsonify({"ok": "null", "err": f"Not np array: {type(embedding)}"})

    #embedding = np.array(embedding, dtype=np.float32)

    logger.debug("Got %d embeddings", len(embedding))

    # Apply
    results, D, I = get_closest_neighbors_urls(embedding, k=k, get_representations_instead_of_embeddings=get_representations_instead_of_embeddings,
                                               check_l2_norm=check_l2_norm)
    results = {"results": results, "D": D, "I": I}
    results = pickle.dumps(results)
    results = base64.b64encode(results).decode() # base64 tensor

    return jsonify({
        "ok": results,
        "err": "null",
    })

def get_closest_neighbors_urls(proto_actions, k=1, get_representations_instead_of_embeddings=True, remove_overlapping_actions=False,
                               translation_candidate=None, check_l2_norm=False):
    """
        observations: states from which proto_actions were generated
    """

    assert not remove_overlapping_actions, "Not currently supported, but when supported, change default value remove_overlapping_actions=True"

    if translation_candidate is not None:
        assert isinstance(translation_candidate, str), type(translation_candidate)

    index = global_conf["embedding_index"]
    urls_representation = global_conf["representation"]
    str2representation = global_conf["str2representation"]
    eos_token_str = global_conf["eos_token_str"]
    action_dim = global_conf["dim"]
    debug = global_conf["debug"]
    proto_actions = utils.embeddings_index_sanity_check(proto_actions, last_dimmension_shape=action_dim, check_l2_norm=check_l2_norm)
    results = []

    assert proto_actions.shape[-1] == action_dim, f"Expected proto_actions last dimension to be {action_dim}, got {proto_actions.shape[-1]}"
    assert isinstance(k, int), k
    assert k > 0, "k must be greater than 0"

    #extra_k = remove_overlapping_actions and src_data_overlap_src_icl_examples > 0
    extra_k = False # TODO support

    if extra_k:
        k += 1

    if index.ntotal == 0:
        # Faiss index is empty
        logger.warn("Faiss index seems to be empty: %d (sentences in pool: %d)", index.ntotal, len(urls_representation))

    D, I = index.search(proto_actions, k) # [D]istance, [I]ndex
    expected_shape = (proto_actions.shape[0], k)

    assert D.shape == expected_shape, f"Expected D.shape to be {expected_shape}, got {D.shape}"
    assert I.shape == expected_shape, f"Expected I.shape to be {expected_shape}, got {I.shape}"

    _fake_representation_str = global_conf["eos_token_str"] # Default representation if no hits are found
    _fake_representation = _fake_representation_str

    # Modify D
    D[I == -1] = -100.0 # Set distance to a negative value for invalid indices
    d_modified_idxs = [(_a, _b) for _a, _b in zip(*np.where(I == -1))] if np.any(I == -1) else []

    # Obtain representations (str) from kNN idxs
    for idx1, (i, d) in enumerate(zip(I, D)):
        overlapping_hits = 0

        results.append([])

        for idx2, (value_idx, value_distance) in enumerate(zip(i, d)):
            if len(results[-1]) >= k:
                break
            if value_idx < 0:
                assert i[idx2:] == -1 * np.ones_like(i[idx2:]), f"Expected all remaining indices to be -1, got {i[idx2:]}"

                break # No more valid indices as -1 values are at the end of the list

            url = urls_representation[value_idx]

            assert isinstance(url, str), f"Expected url to be a string, got {type(url)}: {url}"

            if url != eos_token_str:
                # Check if we need to remove this hit
                src_icl_example, trg_icl_example = url.split('\t')

                if extra_k and translation_candidate == src_icl_example:
                    logger.debug("Removing overlapping action: %s", src_icl_example)

                    overlapping_hits += 1
                    d[idx2] = -200.0
                    i[idx2] = -2

                    d_modified_idxs.append((idx1, idx2))

                    continue # do not add this entry

            results[-1].append(url)

        assert len(results[-1]) <= k, f"Expected results[-1] to have at most {k} elements, got {len(results[-1])}"
        assert overlapping_hits <= 1, f"Expected at most one overlapping hit, got {overlapping_hits}: this might happen if same source is repeated in the ICL examples"

        if extra_k and overlapping_hits == 0:
            # Remove the less similar neighbor

            assert d.shape == (k,), f"Expected d to have shape ({k},), got {d.shape}"
            assert len(results[-1]) == k, f"Expected results[-1] to have length {k}, got {len(results[-1])}"

            idx2 = np.argmax(d)
            d[idx2] = -300.0
            i[idx2] = -3

            del results[-1][idx2]

        if len(results[-1]) < (k - (1 if extra_k else 0)):
            # Add items to avoid tensor errors because dimensions don't match
            logger.debug("Not enough entries close for entry %d/%d (found: %d): returning %d default representation(s) (%s)", idx1 + 1, len(I), len(results[-1]), k - len(results[-1]), _fake_representation_str)

        while len(results[-1]) < (k - (1 if extra_k else 0)):
            results[-1].append(_fake_representation)

        assert len(results[-1]) == (k - (1 if extra_k else 0))

    if debug:
        results2, results3 = [], []

        for r in results[:1]:
            results2.append(r[:5])
            results3.append(r[-5:])

        logger.error("faiss.proto_actions (k=%d): %s", k, proto_actions)
        logger.error("faiss.closest_str (first and last 5): %s ... %s", results2, results3)
        logger.error("faiss.I (first and last 5): %s ... %s", I[:,:5], I[:,-5:])
        logger.error("faiss.D (first and last 5): %s ... %s", D[:,:5], D[:,-5:])

    if not get_representations_instead_of_embeddings:
        # Get embeddings instead of strings

        assert isinstance(results, list)

        if len(results) > 0:
            assert isinstance(results[0], list)

        all_urls = [q for w in results for q in w] # Flatten list of lists

        if len(all_urls) > 0:
            assert isinstance(all_urls[0], str), f"Expected all_urls to be a list of strings, got {type(all_urls[0])}: {all_urls[0]}"

        _all_urls_subset = [url for url in all_urls if url not in str2representation]

        assert len(_all_urls_subset) == 0, f"This should not happen in this environment: {_all_urls_subset}"

        results = [torch.tensor(str2representation[url]) for url in all_urls]
        results = torch.stack(results, dim=0)

        assert len(results.shape) == 2, results.shape
        assert results.shape == (proto_actions.shape[0] * (k - (1 if extra_k else 0)), proto_actions.shape[1])

        results = results.reshape((proto_actions.shape[0], k - (1 if extra_k else 0), proto_actions.shape[1]))

        if debug:
            results2 = results[0,:5]
            results3 = results[0,-5:]

            logger.error("faiss.closest_embedding (first and last 5): %s ... %s", results2, results3)

    return results, D, I


def main(args):
    flask_port = args.flask_port
    run_flask_server = not args.do_not_run_flask_server

    # Global variables
    global_conf["dim"] = args.dim
    global_conf["eos_token_str"] = args.eos_token_str
    global_conf["embedding_index"] = faiss.IndexFlatL2(global_conf["dim"])
    global_conf["representation"] = {}
    global_conf["representation_item2idx"] = {}
    global_conf["str2representation"] = {}
    global_conf["debug"] = args.debug
    global_conf["lock"] = Lock()

#    knn_insert_embedding(representations_str, representations_emb, check_l2_norm=False)

    # Some guidance
    logger.info("Example: curl http://127.0.0.1:%d/hello-world", flask_port)
    logger.info("Examples might not work if you are not using Flask (e.g., you are using gunicorn) and you may be necesasry to adapt them to the used configuration")

    if run_flask_server:
        # Run flask server
        app.run(debug=args.flask_debug, port=flask_port)

def initialization():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="MT_ICL kNN flask server")

    parser.add_argument('--dim', type=int, required=True, help="Embedding dimensionality")
    parser.add_argument('--eos-token-str', type=str, default="</s>", help="EOS token")
    parser.add_argument('--flask-port', type=int, default=5000, help="Flask port")
    parser.add_argument('--do-not-run-flask-server', action="store_true", help="Do not run app.run")
    parser.add_argument('--debug', action="store_true", help="Debug mode")

    parser.add_argument('-v', '--verbose', action="store_true", help="Verbose logging mode")
    parser.add_argument('--flask-debug', action="store_true", help="Flask debug mode. Warning: this option might load the model multiple times")

    args = parser.parse_args()

    return args

def cli():
    global logger

    args = initialization()
    logger = utils.set_up_logging_logger(logging.getLogger("MT_ICL.knn_flask_server"), level=logging.DEBUG if args.verbose else logging.INFO)

    logger.debug("Arguments processed: {}".format(str(args)))

    main(args)

    if not args.do_not_run_flask_server:
        logger.info("Bye!")
    else:
        logger.info("Execution has finished")

if __name__ == "__main__":
    cli()
