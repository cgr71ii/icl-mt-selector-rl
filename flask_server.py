
import os
import json
import base64
import pickle
import logging
import inspect
import argparse
from threading import Lock
import shelve

import utils
import icl_translation as mt_icl
import embeddings as embeddings_import # "odd" name to avoid conflict with variables or functions named embeddings

import torch
from flask import (
    Flask,
    request,
    jsonify,
)
from service_streamer import ThreadedStreamer
from transformers import AutoModelForCausalLM, AutoTokenizer

app = Flask("MT-ICL-flask-server")
global_conf = {} # Empty since it will be filled once main is run
logger = utils.set_up_logging_logger(logging.getLogger("MT_ICL.flask_server"), level=logging.INFO)

# Disable (less verbose) 3rd party logging
logging.getLogger("werkzeug").setLevel(logging.WARNING)

sig_build_prompt = inspect.signature(mt_icl.build_prompt)
template_kwargs = {} # avoid inline declaration of template_kwargs to preserve the order of the keys
template_kwargs["icl_template"] = os.environ["MT_ICL_ICL_TEMPLATE"] if "MT_ICL_ICL_TEMPLATE" in os.environ else sig_build_prompt.parameters["icl_template"].default
template_kwargs["zs_causal_template"] = os.environ["MT_ICL_ZS_CAUSAL_TEMPLATE"] if "MT_ICL_ZS_CAUSAL_TEMPLATE" in os.environ else sig_build_prompt.parameters["zs_causal_template"].default
template_kwargs["zs_chat_user_template"] = os.environ["MT_ICL_ZS_CHAT_USER_TEMPLATE"] if "MT_ICL_ZS_CHAT_USER_TEMPLATE" in os.environ else sig_build_prompt.parameters["zs_chat_user_template"].default
template_kwargs["zs_chat_response_prefix_template"] = os.environ["MT_ICL_ZS_CHAT_RESPONSE_PREFIX_TEMPLATE"] if "MT_ICL_ZS_CHAT_RESPONSE_PREFIX_TEMPLATE" in os.environ else sig_build_prompt.parameters["zs_chat_response_prefix_template"].default
template_kwargs["zswr_causal_template"] = os.environ["MT_ICL_ZSWR_CAUSAL_TEMPLATE"] if "MT_ICL_ZSWR_CAUSAL_TEMPLATE" in os.environ else sig_build_prompt.parameters["zswr_causal_template"].default
template_kwargs["zswr_chat_user_template"] = os.environ["MT_ICL_ZSWR_CHAT_USER_TEMPLATE"] if "MT_ICL_ZSWR_CHAT_USER_TEMPLATE" in os.environ else sig_build_prompt.parameters["zswr_chat_user_template"].default
template_kwargs["zswr_chat_response_prefix_template"] = os.environ["MT_ICL_ZSWR_CHAT_RESPONSE_PREFIX_TEMPLATE"] if "MT_ICL_ZSWR_CHAT_RESPONSE_PREFIX_TEMPLATE" in os.environ else sig_build_prompt.parameters["zswr_chat_response_prefix_template"].default
template_kwargs["chat_system_prompt_template"] = os.environ["MT_ICL_CHAT_SYSTEM_PROMPT_TEMPLATE"] if "MT_ICL_CHAT_SYSTEM_PROMPT_TEMPLATE" in os.environ else sig_build_prompt.parameters["chat_system_prompt_template"].default
template_kwargs["user_prefix_template"] = os.environ["MT_ICL_USER_PREFIX_TEMPLATE"] if "MT_ICL_USER_PREFIX_TEMPLATE" in os.environ else sig_build_prompt.parameters["user_prefix_template"].default

class TranslationCache:
    def __init__(self, path):
        self.path = path
        self.lock = Lock()

        with self.lock:
            with shelve.open(self.path, 'c') as db:
                pass # create db if it does not exist

    def __contains__(self, item):
        with self.lock:
            with shelve.open(self.path, 'r') as db:
                return item in db

    def __delitem__(self, key):
        with self.lock:
            with shelve.open(self.path, 'c') as db:
                del db[key]

    def __setitem__(self, key, item):
        with self.lock:
            with shelve.open(self.path, 'c') as db:
                db[key] = item

    def __getitem__(self, key):
        with self.lock:
            with shelve.open(self.path, 'r') as db:
                return db.get(key)

    def __repr__(self):
        with self.lock:
            with shelve.open(self.path, 'r') as db:
                return list(db.keys())

    def __len__(self):
        with self.lock:
            with shelve.open(self.path, 'r') as db:
                return len(db.keys())

    def __iter__(self):
        raise Exception()

@app.route('/', methods=['GET'])
def info():
    available_routes = json.dumps(
        {
            "/hello-world": ["GET"],
            "/translate": ["GET", "POST"],
            "/get_embedding_from_model_embedding_matrix": ["GET", "POST"],
            "/get_embedding_pooling": ["GET", "POST"],
            "/get_embedding_from_given_model": ["GET", "POST"],
            "/template_info": ["GET"],
        },
        indent=4).replace('\n', '<br/>').replace(' ', '&nbsp;')

    return f"Available routes:<br/>{available_routes}"

@app.route('/template_info', methods=['GET'])
def template_info():
    template_kwargs_print = {}

    for k, v in template_kwargs.items():
        template_kwargs_print[k] = str(v) # preserve the order of the keys and avoid reference copy

    keys = set(template_kwargs_print.keys())

    for k in keys:
        _k = f"MT_ICL_{k.upper()}"
        template_kwargs_print[_k] = template_kwargs_print.pop(k)

    template_kwargs_print = json.dumps(template_kwargs_print, indent=4)

    return f"Available template variables (you can modify the values using envvars):\n{template_kwargs_print}"

@app.route('/hello-world', methods=["GET"])
def hello_world():
    return jsonify({"ok": "hello world! server is working!", "err": "null"})

@app.route('/translate', methods=["GET", "POST"])
def translate():
    route_name = request.base_url.rstrip('/').split('/')[-1]

    if route_name not in ("translate", "get_embedding_pooling"):
        logger.error("Unknown route: %s", route_name)

        return jsonify({"ok": "null", "err": f"unknown route: {route_name}"})

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
        src_lang = utils.string2list(request_method.getlist("src_lang"))
        trg_lang = utils.string2list(request_method.getlist("trg_lang"))
        src_sentences = utils.string2list(request_method.getlist("src_sentence"))
        src_examples = utils.string2list(request_method.getlist("src_example"))
        trg_examples = utils.string2list(request_method.getlist("trg_example"))
        icl_idx_src_sentences = utils.string2list(request_method.getlist("icl_idx_src_sentence"))

        if route_name == "get_embedding_pooling":
            trg_sentences = utils.string2list(request_method.getlist("trg_sentence"))
    except KeyError as e:
        logger.error("KeyError: %s", e)

        return jsonify({"ok": "null", "err": f"could not get some mandatory field: 'urls' are mandatory"})

    # Optional parameters
    if route_name == "translate":
        try:
            trg_sentences = utils.string2list(request_method.getlist("trg_sentence"))
        except KeyError as e:
            trg_sentences = []

    pooling = utils.string2list(request_method.getlist("pooling"))
    layer = list(map(int, utils.string2list(request_method.getlist("layer"))))
    get_representation = True if route_name == "get_embedding_pooling" else False
    get_representation = [get_representation] * len(src_sentences)

    if pooling is None or len(pooling) == 0:
        pooling = ["mean"] * len(src_sentences)

    if layer is None or len(layer) == 0:
        layer = [-1] * len(src_sentences)

    if len(src_sentences) == 0 or len(src_lang) == 0 or len(trg_lang) == 0:
        logger.error("No sentences: %s", src_sentences)

        return jsonify({"ok": "null", "err": "'src_sentence', 'src_lang' and 'trg_lang' are mandatory fields that cannot be empty"})

    logger.debug("Got %d sentences", len(src_sentences))

    if (len(src_lang) != 1 and len(src_lang) != len(src_sentences)) or (len(trg_lang) != 1 and len(trg_lang) != len(src_sentences)):
        logger.error("src_lang: %s vs trg_lang: %s", src_lang, trg_lang)

        return jsonify({"ok": "null", "err": "'src_lang' and 'trg_lang' should be lists with a single element or the same length as 'src_sentence'"})

    if (len(pooling) != 1 and len(pooling) != len(src_sentences)) or (len(layer) != 1 and len(layer) != len(src_sentences)):
        logger.error("pooling: %s layer: %s vs %d", pooling, layer, len(src_sentences))

        return jsonify({"ok": "null", "err": "'pooling' and 'layer' should be lists with a single element or the same length as 'src_sentence'"})

    for idx, p in enumerate(pooling):
        if p not in ("mean", "max", "last", "none", "features"):
            logger.error("Unknown pooling method: %s (idx: %d)", p, idx)

            return jsonify({"ok": "null", "err": f"unknown pooling method: {p} (idx: {idx})"})

    if len(src_lang) == 1:
        src_lang = [src_lang[0]] * len(src_sentences)
    if len(trg_lang) == 1:
        trg_lang = [trg_lang[0]] * len(src_sentences)
    if len(pooling) == 1:
        pooling = [pooling[0]] * len(src_sentences)
    if len(layer) == 1:
        layer = [layer[0]] * len(src_sentences)

    if len(set(pooling)) > 1 or len(set(layer)) > 1:
        # Although it is possible to have different pooling and layer values, this would make inference slower due to batches of different sizes (even 1)

        logger.error("pooling: %s vs layer: %s", pooling, layer)

        return jsonify({"ok": "null", "err": "'pooling' and 'layer' should be a list with a single element or all the same values"})

    if len(trg_sentences) > 0 and route_name != "get_embedding_pooling":
        logger.error("trg_sentences should not be provided for route: %s", route_name)

        return jsonify({"ok": "null", "err": f"trg_sentences should not be provided for route: {route_name}"})

    if len(src_sentences) != len(trg_sentences) and len(trg_sentences) > 0:
        logger.error("Results length mismatch with the provided sentences: %d vs %d: %s vs %s",
                    len(src_sentences), len(trg_sentences), src_sentences, trg_sentences)

        return jsonify({
            "ok": "null",
            "err": f"results length mismatch with the provided URLs: {len(src_sentences)} vs {len(trg_sentences)}",
        })

    if len(trg_sentences) > 0 and not get_representation[0]:
        logger.error("this should not happen: get_representation is False, but trg_sentences are provided: %s", trg_sentences)

        return jsonify({
            "ok": "null",
            "err": f"this should not happen: get_representation is False, but trg_sentences are provided: {trg_sentences}",
        })

    if len(trg_sentences) == 0:
        trg_sentences = [None] * len(src_sentences)

    try:
        src_sentences = [base64.b64decode(s.replace(' ', '+')).decode("utf-8", errors="backslashreplace").replace('\t', ' ').replace('\n', ' ').replace('\r', '').strip() for s in src_sentences]
        src_examples = [base64.b64decode(s.replace(' ', '+')).decode("utf-8", errors="backslashreplace").replace('\t', ' ').replace('\n', ' ').replace('\r', '').strip() for s in src_examples]
        trg_examples = [base64.b64decode(s.replace(' ', '+')).decode("utf-8", errors="backslashreplace").replace('\t', ' ').replace('\n', ' ').replace('\r', '').strip() for s in trg_examples]

        if trg_sentences[0] is not None:
            trg_sentences = [base64.b64decode(s.replace(' ', '+')).decode("utf-8", errors="backslashreplace").replace('\t', ' ').replace('\n', ' ').replace('\r', '').strip() for s in trg_sentences]
    except Exception as e:
        logger.error("Exception when decoding BASE64: %s", e)

        return jsonify({"ok": "null", "err": "error decoding BASE64 data"})

    icl_idx_src_sentences = list(map(lambda d: int(d) - 1, icl_idx_src_sentences)) # the number begins with 1, but we work with 0-based indexes
    icl_examples = [[] for _ in range(len(src_sentences))]

    if len(src_examples) != len(trg_examples):
        return jsonify({"ok": "null", "err": f"src_examples: {len(src_examples)} vs trg_examples: {len(trg_examples)}"})

    if len(src_examples) != len(icl_idx_src_sentences):
        return jsonify({"ok": "null", "err": f"src_examples: {len(src_examples)} vs icl_idx_src_sentences: {len(icl_idx_src_sentences)}"})

    if len(icl_idx_src_sentences) > 0:
        _min = min(icl_idx_src_sentences)
        _max = max(icl_idx_src_sentences)

        if 0 <= _min <= _max < len(src_sentences):
            pass
        else:
            return jsonify({"ok": "null", "err": f"icl_idx_src_sentences: {_min} vs {_max} vs {len(src_sentences)}: {icl_idx_src_sentences} vs {src_sentences}"})

        for icl_idx, src_example, trg_example in zip(icl_idx_src_sentences, src_examples, trg_examples):
            assert isinstance(icl_idx, int), f"icl_idx: {icl_idx} is not an integer: {icl_idx_src_sentences}"
            assert 0 <= icl_idx < len(src_sentences), f"icl_idx: {icl_idx} vs {len(src_sentences)}: {src_sentences}"

            icl_examples[icl_idx].append([src_example, trg_example])

    # Inference

    disable_streamer = global_conf["disable_streamer"]
    get_results = global_conf["streamer_llm"].predict if not disable_streamer else translate_batch
    data = list(zip(src_sentences, icl_examples, src_lang, trg_lang, pooling, layer, get_representation, trg_sentences))
    results = get_results(data)

    if get_representation[0]:
        if not disable_streamer and isinstance(results, list):
            results = torch.stack(results, dim=0)

        assert isinstance(results, torch.Tensor), f"Expected results to be a torch.Tensor, got: {type(results)}: {results}"
        assert len(results.shape) == 2, f"Expected results shape: (batch_size, hidden_dim), got: {results.shape}"

    # Return results
    if len(results) != len(src_sentences):
        logger.error("Results length mismatch with the provided sentences: %d vs %d: %s vs %s",
                     len(results), len(src_sentences), results, src_sentences)

        return jsonify({
            "ok": "null",
            "err": f"results length mismatch with the provided URLs: {len(results)} vs {len(src_sentences)}",
        })

    for idx, (src_sentence, result, trg_sentence, _get_representation) in enumerate(zip(src_sentences, results, trg_sentences, get_representation), 1):
        logger.debug("Results #%d (path: %s ; representation: %s ; target: %s): %s\t%s%s", idx, route_name, str(_get_representation), str(trg_sentence is not None), src_sentence, result, f"\t{trg_sentence}" if trg_sentence is not None else '')

    if get_representation[0]:
        logger.debug("Results shape: %s", results.shape)

        results = pickle.dumps(results)
        results = base64.b64encode(results).decode() # base64 tensor

    return jsonify({
        "ok": results,
        "err": "null",
    })

@app.route('/get_embedding_pooling', methods=["GET", "POST"])
def get_embedding_pooling():
    route_name = request.base_url.rstrip('/').split('/')[-1]

    if route_name not in ("translate", "get_embedding_pooling"):
        logger.error("Unknown route: %s", route_name)

        return jsonify({"ok": "null", "err": f"unknown route: {route_name}"})

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
        src_lang = utils.string2list(request_method.getlist("src_lang"))
        trg_lang = utils.string2list(request_method.getlist("trg_lang"))
        src_sentences = utils.string2list(request_method.getlist("src_sentence"))
        src_examples = utils.string2list(request_method.getlist("src_example"))
        trg_examples = utils.string2list(request_method.getlist("trg_example"))
        icl_idx_src_sentences = utils.string2list(request_method.getlist("icl_idx_src_sentence"))

        if route_name == "get_embedding_pooling":
            trg_sentences = utils.string2list(request_method.getlist("trg_sentence"))
    except KeyError as e:
        logger.error("KeyError: %s", e)

        return jsonify({"ok": "null", "err": f"could not get some mandatory field: 'urls' are mandatory"})

    # Optional parameters
    if route_name == "translate":
        try:
            trg_sentences = utils.string2list(request_method.getlist("trg_sentence"))
        except KeyError as e:
            trg_sentences = []

    pooling = utils.string2list(request_method.getlist("pooling"))
    layer = list(map(int, utils.string2list(request_method.getlist("layer"))))
    get_representation = True if route_name == "get_embedding_pooling" else False
    get_representation = [get_representation] * len(src_sentences)

    if pooling is None or len(pooling) == 0:
        pooling = ["mean"] * len(src_sentences)

    if layer is None or len(layer) == 0:
        layer = [-1] * len(src_sentences)

    if len(src_sentences) == 0 or len(src_lang) == 0 or len(trg_lang) == 0:
        logger.error("No sentences: %s", src_sentences)

        return jsonify({"ok": "null", "err": "'src_sentence', 'src_lang' and 'trg_lang' are mandatory fields that cannot be empty"})

    logger.debug("Got %d sentences", len(src_sentences))

    if (len(src_lang) != 1 and len(src_lang) != len(src_sentences)) or (len(trg_lang) != 1 and len(trg_lang) != len(src_sentences)):
        logger.error("src_lang: %s vs trg_lang: %s", src_lang, trg_lang)

        return jsonify({"ok": "null", "err": "'src_lang' and 'trg_lang' should be lists with a single element or the same length as 'src_sentence'"})

    if (len(pooling) != 1 and len(pooling) != len(src_sentences)) or (len(layer) != 1 and len(layer) != len(src_sentences)):
        logger.error("pooling: %s layer: %s vs %d", pooling, layer, len(src_sentences))

        return jsonify({"ok": "null", "err": "'pooling' and 'layer' should be lists with a single element or the same length as 'src_sentence'"})

    for idx, p in enumerate(pooling):
        if p not in ("mean", "max", "last", "none", "features"):
            logger.error("Unknown pooling method: %s (idx: %d)", p, idx)

            return jsonify({"ok": "null", "err": f"unknown pooling method: {p} (idx: {idx})"})

    if len(src_lang) == 1:
        src_lang = [src_lang[0]] * len(src_sentences)
    if len(trg_lang) == 1:
        trg_lang = [trg_lang[0]] * len(src_sentences)
    if len(pooling) == 1:
        pooling = [pooling[0]] * len(src_sentences)
    if len(layer) == 1:
        layer = [layer[0]] * len(src_sentences)

    if len(set(pooling)) > 1 or len(set(layer)) > 1:
        # Although it is possible to have different pooling and layer values, this would make inference slower due to batches of different sizes (even 1)

        logger.error("pooling: %s vs layer: %s", pooling, layer)

        return jsonify({"ok": "null", "err": "'pooling' and 'layer' should be a list with a single element or all the same values"})

    if len(trg_sentences) > 0 and route_name != "get_embedding_pooling":
        logger.error("trg_sentences should not be provided for route: %s", route_name)

        return jsonify({"ok": "null", "err": f"trg_sentences should not be provided for route: {route_name}"})

    if len(src_sentences) != len(trg_sentences) and len(trg_sentences) > 0:
        logger.error("Results length mismatch with the provided sentences: %d vs %d: %s vs %s",
                    len(src_sentences), len(trg_sentences), src_sentences, trg_sentences)

        return jsonify({
            "ok": "null",
            "err": f"results length mismatch with the provided URLs: {len(src_sentences)} vs {len(trg_sentences)}",
        })

    if len(trg_sentences) > 0 and not get_representation[0]:
        logger.error("this should not happen: get_representation is False, but trg_sentences are provided: %s", trg_sentences)

        return jsonify({
            "ok": "null",
            "err": f"this should not happen: get_representation is False, but trg_sentences are provided: {trg_sentences}",
        })

    if len(trg_sentences) == 0:
        trg_sentences = [None] * len(src_sentences)

    try:
        src_sentences = [base64.b64decode(s.replace(' ', '+')).decode("utf-8", errors="backslashreplace").replace('\t', ' ').replace('\n', ' ').replace('\r', '').strip() for s in src_sentences]
        src_examples = [base64.b64decode(s.replace(' ', '+')).decode("utf-8", errors="backslashreplace").replace('\t', ' ').replace('\n', ' ').replace('\r', '').strip() for s in src_examples]
        trg_examples = [base64.b64decode(s.replace(' ', '+')).decode("utf-8", errors="backslashreplace").replace('\t', ' ').replace('\n', ' ').replace('\r', '').strip() for s in trg_examples]

        if trg_sentences[0] is not None:
            trg_sentences = [base64.b64decode(s.replace(' ', '+')).decode("utf-8", errors="backslashreplace").replace('\t', ' ').replace('\n', ' ').replace('\r', '').strip() for s in trg_sentences]
    except Exception as e:
        logger.error("Exception when decoding BASE64: %s", e)

        return jsonify({"ok": "null", "err": "error decoding BASE64 data"})

    icl_idx_src_sentences = list(map(lambda d: int(d) - 1, icl_idx_src_sentences)) # the number begins with 1, but we work with 0-based indexes
    icl_examples = [[] for _ in range(len(src_sentences))]

    if len(src_examples) != len(trg_examples):
        return jsonify({"ok": "null", "err": f"src_examples: {len(src_examples)} vs trg_examples: {len(trg_examples)}"})

    if len(src_examples) != len(icl_idx_src_sentences):
        return jsonify({"ok": "null", "err": f"src_examples: {len(src_examples)} vs icl_idx_src_sentences: {len(icl_idx_src_sentences)}"})

    if len(icl_idx_src_sentences) > 0:
        _min = min(icl_idx_src_sentences)
        _max = max(icl_idx_src_sentences)

        if 0 <= _min <= _max < len(src_sentences):
            pass
        else:
            return jsonify({"ok": "null", "err": f"icl_idx_src_sentences: {_min} vs {_max} vs {len(src_sentences)}: {icl_idx_src_sentences} vs {src_sentences}"})

        for icl_idx, src_example, trg_example in zip(icl_idx_src_sentences, src_examples, trg_examples):
            assert isinstance(icl_idx, int), f"icl_idx: {icl_idx} is not an integer: {icl_idx_src_sentences}"
            assert 0 <= icl_idx < len(src_sentences), f"icl_idx: {icl_idx} vs {len(src_sentences)}: {src_sentences}"

            icl_examples[icl_idx].append([src_example, trg_example])

    # Inference

    disable_streamer = global_conf["disable_streamer"]
    get_results = global_conf["streamer_llm_embedding"].predict if not disable_streamer else translate_batch
    data = list(zip(src_sentences, icl_examples, src_lang, trg_lang, pooling, layer, get_representation, trg_sentences))
    results = get_results(data)

    if get_representation[0]:
        if not disable_streamer and isinstance(results, list):
            results = torch.stack(results, dim=0)

        assert isinstance(results, torch.Tensor), f"Expected results to be a torch.Tensor, got: {type(results)}: {results}"

        if pooling[0] in ("none", "features"):
            assert len(results.shape) == 3, f"Expected results shape: (batch_size, seq_len, dim), got: {results.shape}"
        else:
            assert len(results.shape) == 2, f"Expected results shape: (batch_size, hidden_dim), got: {results.shape}"

    # Return results
    if len(results) != len(src_sentences):
        logger.error("Results length mismatch with the provided sentences: %d vs %d: %s vs %s",
                     len(results), len(src_sentences), results, src_sentences)

        return jsonify({
            "ok": "null",
            "err": f"results length mismatch with the provided URLs: {len(results)} vs {len(src_sentences)}",
        })

    for idx, (src_sentence, result, trg_sentence, _get_representation) in enumerate(zip(src_sentences, results, trg_sentences, get_representation), 1):
        logger.debug("Results #%d (path: %s ; representation: %s ; target: %s): %s\t%s%s", idx, route_name, str(_get_representation), str(trg_sentence is not None), src_sentence, result, f"\t{trg_sentence}" if trg_sentence is not None else '')

    if get_representation[0]:
        logger.debug("Results shape: %s", results.shape)

        results = pickle.dumps(results)
        results = base64.b64encode(results).decode() # base64 tensor

    return jsonify({
        "ok": results,
        "err": "null",
    })

#def translate_batch(src_sentences, icl_examples, src_lang, trg_lang):
def translate_batch(data):
    src_sentences, icl_examples, src_lang, trg_lang, pooling, layer, get_representation, trg_sentences = zip(*data)
    src_sentences = list(src_sentences)
    icl_examples = list(icl_examples)
    src_lang = list(src_lang)
    trg_lang = list(trg_lang)
    pooling = list(pooling)
    layer = list(layer)
    get_representation = list(get_representation)
    trg_sentences = list(trg_sentences)

    assert len(icl_examples) == len(src_sentences) == len(src_lang) == len(trg_lang) == len(pooling) == len(layer) == len(get_representation) == len(trg_sentences), f"Length mismatch: {len(icl_examples)} vs {len(src_sentences)} vs {len(src_lang)} vs {len(trg_lang)} vs {len(pooling)} vs {len(layer)} vs {len(get_representation)} vs {len(trg_sentences)}"

    pooling = pooling[0]
    layer = layer[0]
    get_representation = get_representation[0]
    teacher_forcing = trg_sentences[0] is not None
    add_eos_token = teacher_forcing

    logger.debug("Data batch size: %d", len(src_sentences))
    logger.debug("Obtaining representation: %s", str(get_representation))
    logger.debug("Teacher forcing: %s", str(teacher_forcing))

    lazy_load_llm()

    model = global_conf["model_llm"]
    tokenizer = global_conf["tokenizer"]
    device = global_conf["device_map"]
    batch_size = global_conf["batch_size"]
    _max_new_tokens = global_conf["max_new_tokens"]

    assert not global_conf["disable_streamer"], "If this is assert is removed, you may get very long prompts as they are processed all at once"

    # Build prompts
    _device = device
    _bsz = batch_size
    results = []
    sentences = [(src_sentence, trg_sentence) for src_sentence, trg_sentence in zip(src_sentences, trg_sentences)] if teacher_forcing else src_sentences
    ## We build the prompt here because when get_representation is True and there is an OOM, we split the batch and we need to keep the same dimensionality (this does not harm if the streamer is enabled, so we force to be enabled)
    _prompts, _src_sentence_n_tokens = mt_icl.build_prompt(sentences, src_lang, trg_lang, tokenizer, icl_examples, len(sentences), teacher_forcing=teacher_forcing, add_eos_token=add_eos_token, lock=global_conf["lock"], **template_kwargs)

    while True:
        try:
            if model.device != _device:
                model = model.to(_device)

#            _src_sentences = src_sentences[:_bsz]
#            _trg_sentences = trg_sentences[:_bsz]
#            _icl_examples = icl_examples[:_bsz]
#            _src_lang = src_lang[:_bsz]
#            _trg_lang = trg_lang[:_bsz]
#
#            assert len(_icl_examples) == len(_src_sentences) == len(_src_lang) == len(_trg_lang), f"Length mismatch: {len(_icl_examples)} vs {len(_src_sentences)} vs {len(_src_lang)} vs {len(_trg_lang)}"
#
#            if teacher_forcing:
#                _sentences = [(src_sentence, trg_sentence) for src_sentence, trg_sentence in zip(_src_sentences, _trg_sentences)]
#            else:
#                _sentences = _src_sentences
#
#            prompts, src_sentence_n_tokens = mt_icl.build_prompt(_sentences, _src_lang, _trg_lang, tokenizer, _icl_examples, _bsz, teacher_forcing=teacher_forcing, add_eos_token=add_eos_token, lock=global_conf["lock"], **template_kwargs)
            prompts = _prompts[:_bsz]
            src_sentence_n_tokens = _src_sentence_n_tokens[:_bsz]

            assert isinstance(prompts, list), type(prompts)

            if len(prompts) > 0:
                assert isinstance(prompts[0], str), type(prompts[0])

            if global_conf["debug"]:
                logger.debug("Prompts: %s", str(prompts))

            if get_representation:
                # Get embeddings
                all_outputs = mt_icl.get_embedding_pooling(model, tokenizer, prompts, pooling=pooling, layer=layer, lock=global_conf["lock"])

#                assert len(all_outputs) == len(prompts) == len(src_sentences[:_bsz])
                assert len(all_outputs) == len(prompts) == len(_prompts[:_bsz])

                results.append(all_outputs)
            else:
                # Translate
                #max_new_tokens = min(_max_new_tokens, src_sentence_n_tokens * 10)
                max_new_tokens = _max_new_tokens

                logger.debug("src_sentence_n_tokens: %d", src_sentence_n_tokens)
                logger.debug("max_new_tokens: %d", max_new_tokens)

                new_prompts_map = {idx: s for idx, s in enumerate(prompts)}
                new_prompts_none = 0

                if global_conf["store_translations"]:
                    all_hits = [utils.get_hash(prompt) in global_conf["store_translations_buffer"] for prompt in prompts]

                    for idx, new_prompt in new_prompts_map.items():
                        assert new_prompts_map[idx] is not None

                        if utils.get_hash(new_prompt) in global_conf["store_translations_buffer"]:
                            new_prompts_map[idx] = None # Do not translate if the output it is stored
                            new_prompts_none += 1

                new_prompts = [prompts[idx] for idx in range(len(prompts)) if new_prompts_map[idx] is not None]
                storage_hits1 = len(prompts) - len(new_prompts)
                storage_hits2 = 0
                aux_all_outputs = []

                if len(new_prompts) > 0:
                    aux_all_outputs, aux_all_original_outputs = mt_icl.translate(model, tokenizer, new_prompts, max_new_tokens=max_new_tokens, stopping_criteria=None, lock=global_conf["lock"], num_beams=global_conf["num_beams"])

#                    assert len(aux_all_outputs) + new_prompts_none == len(aux_all_original_outputs) + new_prompts_none == len(prompts) == len(src_sentences[:_bsz])
                    assert len(aux_all_outputs) + new_prompts_none == len(aux_all_original_outputs) + new_prompts_none == len(prompts) == len(_prompts[:_bsz])

                all_outputs = []
                all_outputs_idx = 0

                for idx in range(len(prompts)):
                    new_prompt = new_prompts_map[idx]

                    if new_prompt is not None:
                        assert prompts[idx] == new_prompt

                        output = aux_all_outputs[all_outputs_idx]

                        all_outputs_idx += 1
                    else:
                        assert storage_hits1 > 0

                        output = global_conf["store_translations_buffer"][utils.get_hash(prompts[idx])]
                        storage_hits2 += 1

                    all_outputs.append(output)

                assert all_outputs_idx + new_prompts_none == len(prompts)
                assert len(all_outputs) == len(prompts)
                assert storage_hits1 == storage_hits2

                results.extend(all_outputs)

                if global_conf["store_translations"]:
                    stored = 0
                    all_hits2 = [utils.get_hash(prompt) in global_conf["store_translations_buffer"] for prompt in prompts]

                    assert all_hits == all_hits2, f"{all_hits} vs {all_hits2}"
                    assert isinstance(prompts, list), type(prompts)
                    assert isinstance(all_outputs, list), type(all_outputs)

                    update_values = {} # we store values to update after checking in order to avoid that a prompt is twice or more, then affecting the conditions below

                    for idx, (prompt, output) in enumerate(zip(prompts, all_outputs)):
                        assert isinstance(prompt, str), type(prompt)
                        assert isinstance(output, str), type(output)

                        if utils.get_hash(prompt) in global_conf["store_translations_buffer"]:
                            if global_conf["store_translations_buffer"][utils.get_hash(prompt)] != output:
                                logger.warning("Different results (updating): %s: %s vs %s", prompt, global_conf["store_translations_buffer"][utils.get_hash(prompt)], output)

                            assert new_prompts_map[idx] is None, f"{idx}: {all_hits2}: {new_prompts_map}: {new_prompts_map[idx]}"
                        else:
                            assert new_prompts_map[idx] is not None

                            stored += 1

                        update_values[utils.get_hash(prompt)] = output

                    for k, v in update_values.items():
                        global_conf["store_translations_buffer"][k] = v

                    logger.debug("Storage hits: %d out of %d", storage_hits1, len(prompts))
                    logger.debug("Stored in storage: %d out of %d", stored, len(prompts))

            _device = device
            _bsz = batch_size
#            src_sentences = src_sentences[len(prompts):]
#            trg_sentences = trg_sentences[len(prompts):]
#            icl_examples = icl_examples[len(prompts):]
#            src_lang = src_lang[len(prompts):]
#            trg_lang = trg_lang[len(prompts):]
            _prompts = _prompts[_bsz:]
            _src_sentence_n_tokens = _src_sentence_n_tokens[_bsz:]
        except torch.OutOfMemoryError as e:
            # Handle OOM

            if _bsz == 1:
                _device = "cpu"
                _bsz = batch_size

                logger.error("torch.OutOfMemoryError error: current batch size is 1: using CPU device and using original batch size: %d", batch_size)
            else:
                logger.error("torch.OutOfMemoryError error: current batch size is %d: using smaller batch size: %d", _bsz, _bsz // 2)

                _bsz = _bsz // 2

#        if len(src_sentences) == 0:
        if len(_prompts) == 0:
            break

    if get_representation:
        for idx in range(len(results) - 1):
            assert results[idx].shape == results[idx + 1].shape, f"Results shape mismatch at idx {idx}: {results[idx].shape} vs {results[idx + 1].shape}"

        results = torch.cat(results, dim=0)

    #return results[target_task] # TODO do we need a list if the streamer is used (it seems so)?
                                 # https://github.com/ShannonAI/service-streamer/issues/97

    return results

@app.route('/get_embedding_from_model_embedding_matrix', methods=["GET", "POST"])
def get_embedding_from_model_embedding_matrix():
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
        tokens = utils.string2list(request_method.getlist("token"))
    except KeyError as e:
        logger.error("KeyError: %s", e)

        return jsonify({"ok": "null", "err": f"could not get some mandatory field: 'urls' are mandatory"})

    if len(tokens) == 0:
        logger.error("No tokens: %s", tokens)

        return jsonify({"ok": "null", "err": "'tokens' is a mandatory field that cannot be empty"})

    logger.debug("Got %d tokens", len(tokens))

    try:
        tokens = [base64.b64decode(s.replace(' ', '+')).decode("utf-8", errors="backslashreplace").replace('\t', ' ').replace('\n', ' ').replace('\r', '').strip() for s in tokens]
    except Exception as e:
        logger.error("Exception when decoding BASE64: %s", e)

        return jsonify({"ok": "null", "err": "error decoding BASE64 data"})

    # Inference

    disable_streamer = global_conf["disable_streamer"]
    get_results = global_conf["streamer_llm_embedding_tokens"].predict if not disable_streamer else embedding_tokens_batch
    _results = get_results(tokens)
    results = _results[0] if disable_streamer else _results
    results_token_id = _results[1] if disable_streamer else [-1 for _ in range(len(tokens))]

    if not disable_streamer and isinstance(results, list):
        results = torch.stack(results, dim=0)

    assert len(results.shape) == 2, results.shape
    assert results.shape[0] == len(tokens), f"{results.shape} | {len(tokens)}"
    assert len(results) == len(results_token_id), f"Results length mismatch: {len(results)} vs {len(results_token_id)}"

    # Return results
    if len(results) != len(tokens):
        logger.error("Results length mismatch with the provided tokens: %s vs %d: %s vs %s",
                     results.shape, len(tokens), results, tokens)

        return jsonify({
            "ok": "null",
            "err": f"results length mismatch with the provided tokens: {results.shape} vs {len(tokens)}",
        })

    for idx, (token, result, result_token_id) in enumerate(zip(tokens, results, results_token_id), 1):
        logger.debug("Results tokens #%d: %s\t%s\t%s", idx, token, result.shape, result_token_id)

    results = pickle.dumps(results)
    results = base64.b64encode(results).decode() # base64 tensor

    return jsonify({
        "ok": results,
        "err": "null",
    })

def embedding_tokens_batch(tokens):
    assert isinstance(tokens, list), f"tokens must be a list, got: {type(tokens)}"

    logger.debug("Data batch size (tokens): %d", len(tokens))

    lazy_load_llm()

    model = global_conf["model_llm"]
    tokenizer = global_conf["tokenizer"]

    # Build prompts
    results = []
    results_token_id = []

    for token in tokens:
        assert isinstance(token, str), f"token must be a string, got: {type(token)}"

        _token, _token_id = mt_icl.get_token_embedding(token, tokenizer, model)

        assert len(_token.shape) == 1, f"Expected token embedding shape: (hidden_dim,), got: {_token.shape}"

        results.append(_token)
        results_token_id.append(_token_id)

    results = torch.stack(results)

    assert len(results.shape) == 2, f"Expected results shape: (batch_size, hidden_dim), got: {results.shape}"
    assert results.shape[0] == len(tokens), f"Results length mismatch: {results.shape[0]} vs {len(tokens)}"

    return results, results_token_id

@app.route('/get_embedding_from_given_model', methods=["GET", "POST"])
def get_embedding_from_given_model():
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
        name = utils.string2list(request_method.getlist("name"))
        lang = utils.string2list(request_method.getlist("lang"))
        sentences = utils.string2list(request_method.getlist("sentence"))
    except KeyError as e:
        logger.error("KeyError: %s", e)

        return jsonify({"ok": "null", "err": f"could not get some mandatory field: 'urls' are mandatory"})

    if len(set(name)) > 1 or len(set(lang)) > 1:
        logger.error("Only one 'name' or 'lang' is allowed, got: %s %s", name, lang)

        return jsonify({"ok": "null", "err": "'name' and 'lang' should be a single value"})

    if len(sentences) == 0:
        logger.error("No sentences: %s", sentences)

        return jsonify({"ok": "null", "err": "'sentences' is a mandatory field that cannot be empty"})

    name = [name[0]] * len(sentences)
    lang = [lang[0]] * len(sentences)

    logger.debug("Got %d sentences (embeddings)", len(sentences))

    try:
        sentences = [base64.b64decode(s.replace(' ', '+')).decode("utf-8", errors="backslashreplace").replace('\t', ' ').replace('\n', ' ').replace('\r', '').strip() for s in sentences]
    except Exception as e:
        logger.error("Exception when decoding BASE64: %s", e)

        return jsonify({"ok": "null", "err": "error decoding BASE64 data"})

    # Inference

    disable_streamer = global_conf["disable_streamer"]
    get_results = global_conf["streamer_embedding"].predict if not disable_streamer else embedding_from_given_model_batch
    data = list(zip(name, lang, sentences))
    results = get_results(data)

    if not disable_streamer and isinstance(results, list):
        results = torch.stack(results, dim=0)

    assert len(results.shape) == 2, results.shape
    assert results.shape[0] == len(sentences), f"{results.shape} | {len(sentences)}"

    # Return results
    if len(results) != len(sentences):
        logger.error("Results length mismatch with the provided tokens: %s vs %d: %s vs %s",
                     results.shape, len(sentences), results, sentences)

        return jsonify({
            "ok": "null",
            "err": f"results length mismatch with the provided tokens: {results.shape} vs {len(sentences)}",
        })

    for idx, (n, l, s, r) in enumerate(zip(name, lang, sentences, results), 1):
        logger.debug("Results embedding #%d: %s\t%s\t%s\t%s", idx, n, l, s, r.shape)

    results = results.numpy()
    results = pickle.dumps(results)
    results = base64.b64encode(results).decode() # base64 tensor

    return jsonify({
        "ok": results,
        "err": "null",
    })

@app.route('/get_embedding_from_given_model_close', methods=["GET", "POST"])
def get_embedding_from_given_model_close():
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
        name = utils.string2list(request_method.getlist("name"))
    except KeyError as e:
        logger.error("KeyError: %s", e)

        return jsonify({"ok": "null", "err": f"could not get some mandatory field: 'urls' are mandatory"})

    for n in name:
        if n in global_conf["model_embedding"]:
            del global_conf["model_embedding"][n]

    torch.cuda.empty_cache()

    logger.debug("Models removed: %s", name)

    return jsonify({
        "ok": "ok",
        "err": "null",
    })

def embedding_from_given_model_batch(data):
    name, lang, sentences = zip(*data)
    name = list(name)[0]
    lang = list(lang)[0]
    sentences = list(sentences)

    logger.debug("Data batch size (embedding): %d", len(sentences))

    device = global_conf["device"]
    batch_size = global_conf["batch_size"]
    model = global_conf["model_embedding"].get(name, None)
    max_seq_len = global_conf["max_seq_len"]

    # Get embeddings
    embeddings, model = embeddings_import.get_embeddings(name, sentences, lang, max_seq_len=max_seq_len, device=device,
                                                         batch_size=batch_size, model=model, return_model=True, numpy=False)

    # Store model for future usage
    global_conf["model_embedding"][name] = model

    return embeddings

def lazy_load_llm():
    if "model_llm" not in global_conf:
        # quantization with fp16
        # examples of quantization: https://github.com/jogonba2/llmixtic/blob/main/src/quantization.py
        global_conf["model_llm"] = AutoModelForCausalLM.from_pretrained(global_conf["pretrained_model"], torch_dtype=torch.float16, device_map=global_conf["device_map"])
        device = global_conf["model_llm"].device # data loading
        global_conf["device_map"] = device
    else:
        # We apply this step in order to avoid loading the model multiple times due to flask debug mode
        pass

def main(args):
    force_cpu = args.force_cpu
    use_cuda = utils.use_cuda(force_cpu=force_cpu)
    device_map = "auto" if use_cuda else torch.device("cpu") # auto allows to load a model using several GPUs
    device = torch.device("cuda") if use_cuda else torch.device("cpu")
    pretrained_model = args.pretrained_model
    flask_port = args.flask_port
    streamer_max_latency = args.streamer_max_latency
    run_flask_server = not args.do_not_run_flask_server
    disable_streamer = args.disable_streamer
    num_beams = args.num_beams

    if not disable_streamer:
        logger.warning("Since streamer is enabled, you might get slightly different results: not recommended for production")
        # Related to https://discuss.pytorch.org/t/slightly-different-results-in-same-machine-and-gpu-but-different-order/173581

    logger.debug("Device: %s | %s (CUDA_VISIBLE_DEVICES=%s)", device, device_map, os.environ.get("CUDA_VISIBLE_DEVICES", "NOT_DEFINED"))

    if "model_embedding" not in global_conf:
        global_conf["model_embedding"] = {}

    global_conf["tokenizer"] = AutoTokenizer.from_pretrained(pretrained_model)

    if global_conf["tokenizer"].pad_token is None:
        # https://github.com/meta-llama/llama3/issues/114#issuecomment-2127131096
        global_conf["tokenizer"].pad_token = global_conf["tokenizer"].eos_token

    worker_timeout = 300
    global_conf["pretrained_model"] = pretrained_model
    global_conf["device"] = device
    global_conf["device_map"] = device_map
    global_conf["batch_size"] = args.batch_size
    global_conf["max_new_tokens"] = args.max_new_tokens
    global_conf["max_seq_len"] = args.max_seq_len
    global_conf["streamer_llm"] = ThreadedStreamer(translate_batch, batch_size=args.batch_size, max_latency=streamer_max_latency, worker_timeout=worker_timeout)
    global_conf["streamer_llm_embedding"] = ThreadedStreamer(translate_batch, batch_size=args.batch_size, max_latency=streamer_max_latency, worker_timeout=worker_timeout)
    global_conf["streamer_llm_embedding_tokens"] = ThreadedStreamer(lambda d: embedding_tokens_batch(d)[0], batch_size=args.batch_size, max_latency=streamer_max_latency, worker_timeout=worker_timeout)
    global_conf["streamer_embedding"] = ThreadedStreamer(embedding_from_given_model_batch, batch_size=args.batch_size, max_latency=streamer_max_latency, worker_timeout=worker_timeout)
    global_conf["disable_streamer"] = disable_streamer
    global_conf["debug"] = args.debug
    global_conf["lock"] = Lock()
    global_conf["num_beams"] = num_beams
    global_conf["store_translations"] = args.store_translations
    global_conf["store_translations_shelve_path"] = f"{os.path.dirname(os.path.realpath(__file__)).rstrip('/')}/flask_server.{pretrained_model.replace('/', '_')}.num_beams_{num_beams}.max_new_tokens_{args.max_new_tokens}.shelve_storage"
    global_conf["store_translations_buffer"] = {}

    if global_conf["store_translations"]:
        if "SHELVE_FILE" in os.environ:
            global_conf["store_translations_shelve_path"] = os.environ["SHELVE_FILE"]

        logger.info("Translations are being stored/loaded (you can set SHELVE_FILE envvar): %s", global_conf["store_translations_shelve_path"])

        global_conf["store_translations_buffer"] = TranslationCache(global_conf["store_translations_shelve_path"])

    # Some guidance
    logger.info("Example: curl http://127.0.0.1:%d/hello-world", flask_port)
    logger.debug("Example: curl http://127.0.0.1:%d/translate -X POST -d \"" + \
                 r'src_lang=English&' + \
                 r'trg_lang=Italian&' + \
                 r'layer=-1&' + \
                 r'pooling=mean&' + \
                 r'src_sentence=TG9jYWwgbWVkaWEgcmVwb3J0cyBhbiBhaXJwb3J0IGZpcmUgdmVoaWNsZSByb2xsZWQgb3ZlciB3aGlsZSByZXNwb25kaW5nLgo=&' + \
                 r'src_lang=English&' + \
                 r'trg_lang=Spanish&' + \
                 r'src_sentence=IldlIG5vdyBoYXZlIDQtbW9udGgtb2xkIG1pY2UgdGhhdCBhcmUgbm9uLWRpYWJldGljIHRoYXQgdXNlZCB0byBiZSBkaWFiZXRpYywiIGhlIGFkZGVkLgo=&' + \
                 r'src_example=SW4gSnVuZSwgdGhlIENvbW1pc3Npb24gcHVibGlzaGVkIHRoZSByZXN1bHRzIG9mIGEgcHVibGljIGNvbnN1bHRhdGlvbiBvbiB0aGUgcHJvcG9zYWxzIHdoaWNoIGZvdW5kIGJyb2FkIHN1cHBvcnQgZm9yIGNhbGxpbmcgdGhlIGFzc2VtYmx5IGEgV2Vsc2ggUGFybGlhbWVudC4K&' + \
                 r'trg_example=RW4ganVuaW8sIGxhIENvbWlzacOzbiBwdWJsaWPDsyBsb3MgcmVzdWx0YWRvcyBkZSB1bmEgY29uc3VsdGEgcMO6YmxpY2Egc29icmUgbGFzIHByb3B1ZXN0YXMsIGVuIGRvbmRlIHNlIG9idHV2byB1biBhbXBsaW8gYXBveW8gcGFyYSBsbGFtYXIgYSBsYSBhc2FtYmxlYSB1biBQYXJsYW1lbnRvIGRlIEdhbGVzLgo=&' + \
                 r'icl_idx_src_sentence=2&' + \
                 r'src_example=V2F0ZXJzJyBzdGF0ZW1lbnQgcXVpY2tseSBkcmV3IGNyaXRpY2lzbSBvbmxpbmUsIGluY2x1ZGluZyBmcm9tIGZvcm1lciBXaGl0ZSBIb3VzZSBwcmVzcyBzZWNyZXRhcnkgQXJpIEZsZWlzY2hlci4K&' + \
                 r'trg_example=TGEgZGVjbGFyYWNpw7NuIGRlIFdhbHRlcnMgcHJvdm9jw7MgcsOhcGlkYW1lbnRlIGNyw610aWNhcyBlbiBJbnRlcm5ldCwgaW5jbHV5ZW5kbyB1bmEgZGVsIGFudGVyaW9yIHNlY3JldGFyaW8gZGUgcHJlbnNhIGRlIGxhIENhc2EgQmxhbmNhIEFyaSBGbGVpc2NoZXIuCg==&' + \
                 r'icl_idx_src_sentence=2&' + \
                 r'src_example=TGlrZSBzb21lIG90aGVyIGV4cGVydHMsIGhlIGlzIHNrZXB0aWNhbCBhYm91dCB3aGV0aGVyIGRpYWJldGVzIGNhbiBiZSBjdXJlZCwgbm90aW5nIHRoYXQgdGhlc2UgZmluZGluZ3MgaGF2ZSBubyByZWxldmFuY2UgdG8gcGVvcGxlIHdobyBhbHJlYWR5IGhhdmUgVHlwZSAxIGRpYWJldGVzLgo=&' + \
                 r'trg_example=w4AgbCdpbnN0YXIgZCdhdXRyZXMgZXhwZXJ0cywgaWwgc2UgbW9udHJlIHNjZXB0aXF1ZSBxdWFudCDDoCBsYSBwb3NzaWJpbGl0w6kgZGUgZ3XDqXJpciBsZSBkaWFiw6h0ZSwgZmFpc2FudCByZW1hcnF1ZXIgcXVlIGNlcyByw6lzdWx0YXRzIG5lIHNvbnQgcGFzIGFwcGxpY2FibGVzIGF1eCBwZXJzb25uZXMgcXVpIHNvdWZmcmVudCBkw6lqw6AgZGUgZGlhYsOodGUgZGUgdHlwZSAxLgo=&' + \
                 r'icl_idx_src_sentence=3&' + \
                 r'src_example=SXQgd2FzIGEgdGhpcmQgRWxpdGUgTGVhZ3VlIGRlZmVhdCBvZiB0aGUgc2Vhc29uIGZvciBBZGFtIEtlZWZlJ3MgbWVuLCB3aG8gaGFkIGNvbWUgZnJvbSBiZWhpbmQgdG8gYmVhdCBEdW5kZWUgMi0xIGluIEJlbGZhc3Qgb24gRnJpZGF5IG5pZ2h0Lgo=&' + \
                 r'trg_example=RnVlIGxhIHRlcmNlcmEgZGVycm90YSBkZSBsYSB0ZW1wb3JhZGEgZGUgbGEgRWxpdGUgTGVhZ3VlIHBhcmEgZWwgZXF1aXBvIGRlIEFkYW0gS2VlZmUsIHF1aWVuZXMgdHV2aWVyb24gcXVlIGp1Z2FyIGRlc2RlIHVuYSBwb3NpY2nDs24gZW4gZGVzdmVudGFqYSBwYXJhIHZlbmNlciBhIER1bmRlZSAyIGEgMSBlbiBCZWxmYXN0IGVsIHZpZXJuZXMgZW4gbGEgbm9jaGUuCg==&' + \
                 r'icl_idx_src_sentence=2&' + \
                 r'src_sentence=TGlrZSBzb21lIG90aGVyIGV4cGVydHMsIGhlIGlzIHNrZXB0aWNhbCBhYm91dCB3aGV0aGVyIGRpYWJldGVzIGNhbiBiZSBjdXJlZCwgbm90aW5nIHRoYXQgdGhlc2UgZmluZGluZ3MgaGF2ZSBubyByZWxldmFuY2UgdG8gcGVvcGxlIHdobyBhbHJlYWR5IGhhdmUgVHlwZSAxIGRpYWJldGVzLgo=&' + \
                 r'src_lang=English&' + \
                 r'trg_lang=French&' + \
                '"', flask_port)
    logger.debug("Example: curl http://127.0.0.1:%d/get_embedding_from_model_embedding_matrix -X POST -d \"" + \
                 r'token=PC9zPg==&' + \
                 r'token=PHM+&' + \
                '"', flask_port)
    logger.debug("Example: curl http://127.0.0.1:%d/get_embedding_pooling -X POST -d \"" + \
                 r'src_lang=English&' + \
                 r'trg_lang=Spanish&' + \
                 r'layer=-1&' + \
                 r'pooling=mean&' + \
                 r'src_sentence=SW4gSnVuZSwgdGhlIENvbW1pc3Npb24gcHVibGlzaGVkIHRoZSByZXN1bHRzIG9mIGEgcHVibGljIGNvbnN1bHRhdGlvbiBvbiB0aGUgcHJvcG9zYWxzIHdoaWNoIGZvdW5kIGJyb2FkIHN1cHBvcnQgZm9yIGNhbGxpbmcgdGhlIGFzc2VtYmx5IGEgV2Vsc2ggUGFybGlhbWVudC4K&' + \
                 r'trg_sentence=RW4ganVuaW8sIGxhIENvbWlzacOzbiBwdWJsaWPDsyBsb3MgcmVzdWx0YWRvcyBkZSB1bmEgY29uc3VsdGEgcMO6YmxpY2Egc29icmUgbGFzIHByb3B1ZXN0YXMsIGVuIGRvbmRlIHNlIG9idHV2byB1biBhbXBsaW8gYXBveW8gcGFyYSBsbGFtYXIgYSBsYSBhc2FtYmxlYSB1biBQYXJsYW1lbnRvIGRlIEdhbGVzLgo=&' + \
                 r'src_lang=English&' + \
                 r'trg_lang=French&' + \
                 r'layer=-1&' + \
                 r'pooling=mean&' + \
                 r'src_sentence=TGlrZSBzb21lIG90aGVyIGV4cGVydHMsIGhlIGlzIHNrZXB0aWNhbCBhYm91dCB3aGV0aGVyIGRpYWJldGVzIGNhbiBiZSBjdXJlZCwgbm90aW5nIHRoYXQgdGhlc2UgZmluZGluZ3MgaGF2ZSBubyByZWxldmFuY2UgdG8gcGVvcGxlIHdobyBhbHJlYWR5IGhhdmUgVHlwZSAxIGRpYWJldGVzLgo=&' + \
                 r'trg_sentence=w4AgbCdpbnN0YXIgZCdhdXRyZXMgZXhwZXJ0cywgaWwgc2UgbW9udHJlIHNjZXB0aXF1ZSBxdWFudCDDoCBsYSBwb3NzaWJpbGl0w6kgZGUgZ3XDqXJpciBsZSBkaWFiw6h0ZSwgZmFpc2FudCByZW1hcnF1ZXIgcXVlIGNlcyByw6lzdWx0YXRzIG5lIHNvbnQgcGFzIGFwcGxpY2FibGVzIGF1eCBwZXJzb25uZXMgcXVpIHNvdWZmcmVudCBkw6lqw6AgZGUgZGlhYsOodGUgZGUgdHlwZSAxLgo=&' + \
                '"', flask_port)
    logger.debug("Example: curl http://127.0.0.1:%d/get_embedding_from_given_model -X POST -d \"" + \
                 r'name=SONAR&' + \
                 r'lang=English&' + \
                 r'sentence=SW4gSnVuZSwgdGhlIENvbW1pc3Npb24gcHVibGlzaGVkIHRoZSByZXN1bHRzIG9mIGEgcHVibGljIGNvbnN1bHRhdGlvbiBvbiB0aGUgcHJvcG9zYWxzIHdoaWNoIGZvdW5kIGJyb2FkIHN1cHBvcnQgZm9yIGNhbGxpbmcgdGhlIGFzc2VtYmx5IGEgV2Vsc2ggUGFybGlhbWVudC4K&' + \
                 r'sentence=TGlrZSBzb21lIG90aGVyIGV4cGVydHMsIGhlIGlzIHNrZXB0aWNhbCBhYm91dCB3aGV0aGVyIGRpYWJldGVzIGNhbiBiZSBjdXJlZCwgbm90aW5nIHRoYXQgdGhlc2UgZmluZGluZ3MgaGF2ZSBubyByZWxldmFuY2UgdG8gcGVvcGxlIHdobyBhbHJlYWR5IGhhdmUgVHlwZSAxIGRpYWJldGVzLgo=&' + \
                '"', flask_port)
    logger.info("Examples might not work if you are not using Flask (e.g., you are using gunicorn) and you may be necesasry to adapt them to the used configuration")
    logger.info("Sentences are expected to be provided in BASE64 format")

    if run_flask_server:
        # Run flask server
        app.run(debug=args.flask_debug, port=flask_port)

def initialization():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="MT_ICL flask server")

    parser.add_argument('--batch-size', type=int, default=16, help="Batch size")
    parser.add_argument('--pretrained-model', default="meta-llama/Llama-2-7b-chat-hf", help="Pretrained model")
    parser.add_argument('--max-new-tokens', type=int, default=256, help="Max. length for the generated tokens")
    parser.add_argument('--max-seq-len', type=int, default=512, help="Max. length for the input sequences (for embeddings from given model)")
    parser.add_argument('--force-cpu', action="store_true", help="Run on CPU (i.e. do not check if GPU is possible)")
    parser.add_argument('--disable-streamer', action="store_true", help="Do not use streamer (it might lead to slower inference and OOM errors)")
    parser.add_argument('--flask-port', type=int, default=5000, help="Flask port")
    parser.add_argument('--streamer-max-latency', type=float, default=0.1,
                        help="Streamer max latency. You will need to modify this parameter if you want to increase the GPU usage")
    parser.add_argument('--do-not-run-flask-server', action="store_true", help="Do not run app.run")
    parser.add_argument('--num-beams', type=int, default=4, help="Number of beams for beam search")
    parser.add_argument('--store-translations', action="store_true", help="Store in memory translations")
    parser.add_argument('--debug', action="store_true", help="Debug mode")

    parser.add_argument('-v', '--verbose', action="store_true", help="Verbose logging mode")
    parser.add_argument('--flask-debug', action="store_true", help="Flask debug mode. Warning: this option might load the model multiple times")

    args = parser.parse_args()

    return args

def cli():
    global logger

    args = initialization()
    logger = utils.set_up_logging_logger(logging.getLogger("MT_ICL.flask_server"), level=logging.DEBUG if args.verbose else logging.INFO)

    logger.debug("Arguments processed: {}".format(str(args)))

    main(args)

    if not args.do_not_run_flask_server:
        logger.info("Bye!")
    else:
        logger.info("Execution has finished")

if __name__ == "__main__":
    cli()
