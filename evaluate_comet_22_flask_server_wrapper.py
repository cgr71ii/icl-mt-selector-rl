
def init(batch_size=16, streamer_max_latency=0.1, use_all_gpus=False, pretrained_model="Unbabel/wmt22-comet-da"):
    import os
    import sys
    import logging

    import evaluate_comet_22_flask_server as flask_server

    devices = ''

    logging.warning("Watch out multi-GPU usage: if gunicorn is used, xCOMET uses pytorch lightning and there may be conflicts. Ideally, gunicorn should use -w and pytorch lightning use a single node")

    if "CUDA_VISIBLE_DEVICES" in os.environ:
        devices = os.environ["CUDA_VISIBLE_DEVICES"].split(',')

        if len(devices) > 1 and not use_all_gpus:
            cuda_device = devices[os.getpid() % len(devices)]
            os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)

            logging.warning("NOT a perfect approach for assigning GPUs: be aware that you might need to reset if GPUs are not allocated properly")
            logging.info("CUDA device (PID: %s): %s (available: %s)", os.getpid(), cuda_device, len(devices))

    logging.info("Devices: %s", devices)

    sys.argv = [sys.argv[0]] # Remove all provided args

    # Inject args that will be used by the Flask server
    sys.argv.extend([
        "--batch-size", str(batch_size),
        "--pretrained-model", pretrained_model,
        "--streamer-max-latency", str(streamer_max_latency),
        "--do-not-run-flask-server", # Necessary for gunicorn in order to work properly
        "--verbose",
        "--disable-streamer", # It should be enabled for crawls of multiple websites, but disabled for a few websites
    ])

    flask_server.cli()

    return flask_server.app
