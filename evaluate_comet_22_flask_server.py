
import os
import json
import base64
import pickle
import logging
import argparse

import comet
import evaluate_comet_22 as eval_comet
import utils

import torch
import numpy as np
from flask import (
    Flask,
    request,
    jsonify,
)
from service_streamer import ThreadedStreamer

app = Flask("MT-ICL-evaluate_comet_22-flask-server")

global_conf = {} # Empty since it will be filled once main is run
logger = utils.set_up_logging_logger(logging.getLogger("MT_ICL.evaluate_comet_22_flask_server"), level=logging.INFO)

# Disable (less verbose) 3rd party logging
logging.getLogger("werkzeug").setLevel(logging.WARNING)

@app.route('/', methods=['GET'])
def info():
    available_routes = json.dumps(
        {
            "/hello-world": ["GET"],
            "/evaluate_comet_22": ["GET", "POST"],
        },
        indent=4).replace('\n', '<br/>').replace(' ', '&nbsp;')

    return f"Available routes:<br/>{available_routes}"

@app.route('/hello-world', methods=["GET"])
def hello_world():
    return jsonify({"ok": "hello world! server is working!", "err": "null"})

@app.route('/evaluate_comet_22', methods=["GET", "POST"])
def evaluate_comet_22():
    if request.method not in ("GET", "POST"):
        return jsonify({"ok": "null", "err": "method is not: GET, POST"})

    if request.method == "GET":
        # GET method should be used only for testing purposes since HTML encoding is not being handled
        request_method = request.args
    elif request.method == "POST":
        request_method = request.form
    else:
        logger.warning("Unknown method: %s", request.method)

        return jsonify({"ok": "null", "err": f"unknown method: {request.method}"})

    # Get parameters
    try:
        src_sentences = utils.string2list(request_method.getlist("src_sentence"))
        mt_sentences = utils.string2list(request_method.getlist("mt_sentence"))
        ref_sentences = utils.string2list(request_method.getlist("ref_sentence"))
    except KeyError as e:
        logger.warning("KeyError: %s", e)

        return jsonify({"ok": "null", "err": f"could not get some mandatory field: 'urls' are mandatory"})

    # Optional parameters
    #try:
    #    foo = request_method.getlist("foo")
    #except KeyError as e:
    #    foo = None

    if len(src_sentences) == 0 or len(mt_sentences) == 0 or len(ref_sentences) == 0:
        logger.warning("No sentences: %s | %s | %s", src_sentences, mt_sentences, ref_sentences)

        return jsonify({"ok": "null", "err": "'mt_sentence', 'mt_sentence' and 'ref_sentence' are mandatory fields that cannot be empty"})

    logger.debug("Got %d sentences", len(src_sentences))

    if len(src_sentences) != len(mt_sentences) or len(mt_sentences) != len(ref_sentences):
        return jsonify({"ok": "null", "err": f"'src_sentence', 'mt_sentence' and 'ref_sentence' must contain the same number of elements: {len(src_sentences)} vs {len(mt_sentences)} vs {len(ref_sentences)}"})

    try:
        src_sentences = [base64.b64decode(s.replace(' ', '+')).decode("utf-8", errors="backslashreplace").replace('\t', ' ').replace('\n', ' ').replace('\r', '').strip() for s in src_sentences]
        mt_sentences = [base64.b64decode(s.replace(' ', '+')).decode("utf-8", errors="backslashreplace").replace('\t', ' ').replace('\n', ' ').replace('\r', '').strip() for s in mt_sentences]
        ref_sentences = [base64.b64decode(s.replace(' ', '+')).decode("utf-8", errors="backslashreplace").replace('\t', ' ').replace('\n', ' ').replace('\r', '').strip() for s in ref_sentences]
    except Exception as e:
        logger.error("Exception when decoding BASE64: %s", e)

        return jsonify({"ok": "null", "err": "error decoding BASE64 data"})

    # Inference

    disable_streamer = global_conf["disable_streamer"]
    get_results = global_conf["streamer"].predict if not disable_streamer else evaluate_comet_22_batch
    data = list(zip(src_sentences, mt_sentences, ref_sentences))
    results = list(map(str, get_results(data)))

    # Return results
    if len(results) != len(src_sentences):
        logger.error("Results length mismatch with the provided sentences: %d vs %d: %s vs %s",
                     len(results), len(src_sentences), results, src_sentences)

        return jsonify({
            "ok": "null",
            "err": f"results length mismatch with the provided URLs: {len(results)} vs {len(src_sentences)}",
        })

    for idx, (src_sentence, mt_sentence, ref_sentence, result) in enumerate(zip(src_sentences, mt_sentences, ref_sentences, results), 1):
        logger.debug("Results #%d: %s\t%s\t%s\t%s", idx, src_sentence, mt_sentence, ref_sentence, result)

    return jsonify({
        "ok": results,
        "err": "null",
    })

def evaluate_comet_22_batch(data):
    src_sentences, mt_sentences, ref_sentences = zip(*data)
    src_sentences = list(src_sentences)
    mt_sentences = list(mt_sentences)
    ref_sentences = list(ref_sentences)

    assert len(src_sentences) == len(mt_sentences) == len(ref_sentences), f"Length mismatch: {len(src_sentences)} vs {len(mt_sentences)} vs {len(ref_sentences)}"

    logger.debug("Data batch size: %d", len(src_sentences))

    model = global_conf["model"]
    gpus = global_conf["gpus"]
    batch_size = global_conf["batch_size"]
    clip_min = global_conf["clip_min"]
    clip_max = global_conf["clip_max"]

    # Build prompts
    _gpus = gpus
    _bsz = batch_size
    results = []

    while True:
        try:
            _src_sentences = src_sentences[:_bsz]
            _mt_sentences = mt_sentences[:_bsz]
            _ref_sentences = ref_sentences[:_bsz]

            assert len(_src_sentences) == len(_mt_sentences) == len(_ref_sentences), f"Length mismatch: {len(_src_sentences)} vs {len(_mt_sentences)} vs {len(_ref_sentences)}"

            # Evaluation
            all_outputs_avg, all_outputs = eval_comet.eval(model, _src_sentences, _mt_sentences, _ref_sentences, batch_size=_bsz, gpus=_gpus, clip_values=(clip_min, clip_max), logger=logger)

            assert len(all_outputs) == len(src_sentences[:_bsz])

            results.extend(all_outputs)

            _gpus = gpus
            _bsz = batch_size
            src_sentences = src_sentences[len(_src_sentences):]
            mt_sentences = mt_sentences[len(_mt_sentences):]
            ref_sentences = ref_sentences[len(_ref_sentences):]
        except torch.OutOfMemoryError as e:
            # Handle OOM

            if _bsz == 1:
                _gpus = 0
                _bsz = batch_size

                logger.error("torch.OutOfMemoryError error: current batch size is 1: using CPU device and using original batch size: %d", batch_size)
            else:
                logger.error("torch.OutOfMemoryError error: current batch size is %d: using smaller batch size: %d", _bsz, _bsz // 2)

                _bsz = _bsz // 2

        if len(src_sentences) == 0:
            break

    return results

def main(args):
    force_cpu = args.force_cpu
    use_cuda = utils.use_cuda(force_cpu=force_cpu)
    gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(',')) if "CUDA_VISIBLE_DEVICES" in os.environ and use_cuda else 1 if use_cuda else 0
    flask_port = args.flask_port
    streamer_max_latency = args.streamer_max_latency
    run_flask_server = not args.do_not_run_flask_server
    disable_streamer = args.disable_streamer
    pretrained_model = args.pretrained_model

    logger.info("Pretrained model: %s", pretrained_model)

    if not disable_streamer:
        logger.warning("Since streamer is enabled, you might get slightly different results: not recommended for production")
        # Related to https://discuss.pytorch.org/t/slightly-different-results-in-same-machine-and-gpu-but-different-order/173581

    logger.debug("GPUs: %d", gpus)

    if "model" not in global_conf:
        # Load COMET 22 DA
        model_path = comet.download_model(pretrained_model)
        global_conf["model"] = comet.load_from_checkpoint(model_path)
    else:
        # We apply this step in order to avoid loading the model multiple times due to flask debug mode
        pass

    worker_timeout = 300
    global_conf["gpus"] = gpus
    global_conf["batch_size"] = args.batch_size
    global_conf["streamer"] = ThreadedStreamer(evaluate_comet_22_batch, batch_size=args.batch_size, max_latency=streamer_max_latency, worker_timeout=worker_timeout)
    global_conf["disable_streamer"] = disable_streamer
    global_conf["clip_min"] = args.clip_min
    global_conf["clip_max"] = args.clip_max

    # Some guidance
    logger.info("Example: curl http://127.0.0.1:%d/hello-world", flask_port)
    ## Translation in example by google translate, july 2025
    logger.debug("Example: curl http://127.0.0.1:%d/evaluate_comet_22 -X POST -d \"" + \
                 r'src_sentence=IldlIG5vdyBoYXZlIDQtbW9udGgtb2xkIG1pY2UgdGhhdCBhcmUgbm9uLWRpYWJldGljIHRoYXQgdXNlZCB0byBiZSBkaWFiZXRpYywiIGhlIGFkZGVkLgo=&' + \
                 r'mt_sentence=IkFob3JhIHRlbmVtb3MgcmF0b25lcyBkZSBjdWF0cm8gbWVzZXMgcXVlIG5vIHNvbiBkaWFiw6l0aWNvcyB5IHF1ZSBhbnRlcyBlcmFuIGRpYWLDqXRpY29zIiwgYcOxYWRpw7MuCg==&' + \
                 r'ref_sentence=wqtBY3R1YWxtZW50ZSwgdGVuZW1vcyByYXRvbmVzIGRlIGN1YXRybyBtZXNlcyBkZSBlZGFkIHF1ZSBhbnRlcyBzb2zDrWFuIHNlciBkaWFiw6l0aWNvcyB5IHF1ZSB5YSBubyBsbyBzb27CuywgYWdyZWfDsy4K&' + \
                 r'src_sentence=IldlIG5vdyBoYXZlIDQtbW9udGgtb2xkIG1pY2UgdGhhdCBhcmUgbm9uLWRpYWJldGljIHRoYXQgdXNlZCB0byBiZSBkaWFiZXRpYywiIGhlIGFkZGVkLgo=&' + \
                 r'mt_sentence=IkFob3JhIHRlbmVtb3MgcmF0b25lcyBkZSBjdWF0cm8gbWVzZXMgcXVlIG5vIHNvbiBkaWFiw6l0aWNvcyB5IHF1ZSBhbnRlcyBlcmFuIGRpYWLDqXRpY29zIiwgYcOxYWRpw7MuCg==&' + \
                 r'ref_sentence=wqvCoE5vdXMgYXZvbnMgw6AgcHLDqXNlbnQgZGVzIHNvdXJpcyBkZSA0wqBtb2lzIHF1aSBuZSBzb250IHBhcyBkaWFiw6l0aXF1ZXMgYWxvcnMgcXUnZWxsZXMgbCfDqXRhaWVudCBhdXBhcmF2YW50wqDCuywgYS10LWlsIGFqb3V0w6kuCg==&' + \
                '"', flask_port)
    logger.info("Examples might not work if you are not using Flask (e.g., you are using gunicorn) and you may be necesasry to adapt them to the used configuration")
    logger.info("Sentences are expected to be provided in BASE64 format")

    if run_flask_server:
        # Run flask server
        app.run(debug=args.flask_debug, port=flask_port)

def initialization():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="MT_ICL evaluate_comet_22.py flask server")

    parser.add_argument('--batch-size', type=int, default=16, help="Batch size")
    parser.add_argument('--pretrained-model', default="Unbabel/wmt22-comet-da", help="Pretrained model")
    parser.add_argument('--force-cpu', action="store_true", help="Run on CPU (i.e. do not check if GPU is possible)")
    parser.add_argument('--disable-streamer', action="store_true", help="Do not use streamer (it might lead to slower inference and OOM errors)")
    parser.add_argument('--flask-port', type=int, default=5000, help="Flask port")
    parser.add_argument('--streamer-max-latency', type=float, default=0.1,
                        help="Streamer max latency. You will need to modify this parameter if you want to increase the GPU usage")
    parser.add_argument('--do-not-run-flask-server', action="store_true", help="Do not run app.run")
    parser.add_argument('--clip-min', type=float, default=0.0, help="Minimum clipping score value")
    parser.add_argument('--clip-max', type=float, default=1.0, help="Maximum clipping score value")

    parser.add_argument('-v', '--verbose', action="store_true", help="Verbose logging mode")
    parser.add_argument('--flask-debug', action="store_true", help="Flask debug mode. Warning: this option might load the model multiple times")

    args = parser.parse_args()

    return args

def cli():
    global logger

    args = initialization()
    logger = utils.set_up_logging_logger(logging.getLogger("MT_ICL.evaluate_comet_22_flask_server"), level=logging.DEBUG if args.verbose else logging.INFO)

    logger.debug("Arguments processed: {}".format(str(args)))

    main(args)

    if not args.do_not_run_flask_server:
        logger.info("Bye!")
    else:
        logger.info("Execution has finished")

if __name__ == "__main__":
    cli()
