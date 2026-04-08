
def init(batch_size=16, streamer_max_latency=0.1, pretrained_model="meta-llama/Llama-2-7b-chat-hf", use_all_gpus=False, num_beams=4, max_new_tokens=256, lazy_load=True):
    import os
    import sys
    import logging

    import flask_server

    devices = ''

    if "CUDA_VISIBLE_DEVICES" in os.environ:
        devices = os.environ["CUDA_VISIBLE_DEVICES"].split(',')

        if len(devices) > 1 and not use_all_gpus:
            cuda_device = devices[os.getpid() % len(devices)]
            os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)

            logging.warning("NOT a perfect approach for assigning GPUs: be aware that you might need to reset if GPUs are not allocated properly")
            logging.info("CUDA device (PID: %d): %d (available: %d)", os.getpid(), cuda_device, len(devices))

    logging.info("Devices: %s", devices)

    sys.argv = [sys.argv[0]] # Remove all provided args

    # Inject args that will be used by the Flask server
    sys.argv.extend([
        "--batch-size", str(batch_size),
        "--streamer-max-latency", str(streamer_max_latency),
        "--pretrained-model", pretrained_model,
        "--do-not-run-flask-server", # Necessary for gunicorn in order to work properly
        "--verbose",
#        "--disable-streamer", # It should be enabled for crawls of multiple websites, but disabled for a few websites
        "--num-beams", str(num_beams),
#        "--store-translations",
        "--max-new-tokens", str(max_new_tokens),
        "--debug",
    ])

    if not lazy_load:
        sys.argv.append("--do-not-lazy-load")

    flask_server.cli()

    return flask_server.app
