
import numpy as np

def init(dim, eos_token_str="</s>"):
    import sys

    import knn_flask_server

    sys.argv = [sys.argv[0]] # Remove all provided args

    # Inject args that will be used by the Flask server
    sys.argv.extend([
        "--dim", str(dim),
        "--eos-token-str", eos_token_str,
        "--do-not-run-flask-server", # Necessary for gunicorn in order to work properly
        "--verbose",
        "--debug",
    ])

    knn_flask_server.cli()

    return knn_flask_server.app
