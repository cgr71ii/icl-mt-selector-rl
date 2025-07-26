
def init(batch_size=16, streamer_max_latency=0.1, pretrained_model="meta-llama/Llama-2-7b-chat-hf"):
    import os
    import sys
    import flask_server

    if "CUDA_VISIBLE_DEVICES" in os.environ:
        devices = os.environ["CUDA_VISIBLE_DEVICES"].split(',')

        if len(devices) > 1:
            import logging

            cuda_device = devices[os.getpid() % len(devices)]
            os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)

            logging.warning("NOT a perfect approach for assigning GPUs: be aware that you might need to reset if GPUs are not allocated properly")
            logging.info("CUDA device (PID: %d): %d (available: %d)", os.getpid(), cuda_device, len(devices))

    sys.argv = [sys.argv[0]] # Remove all provided args

    # Inject args that will be used by the Flask server
    sys.argv.extend([
        "--batch-size", str(batch_size),
        "--streamer-max-latency", str(streamer_max_latency),
        "--pretrained-model", pretrained_model,
        "--do-not-run-flask-server", # Necessary for gunicorn in order to work properly
        "--verbose",
        "--disable-streamer", # It should be enabled for crawls of multiple websites, but disabled for a few websites
        "--debug",
    ])

    flask_server.cli()

    return flask_server.app
