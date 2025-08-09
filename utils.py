
import os
import random
import base64
import logging

import torch
import torch.nn.functional as F
import numpy as np

def use_cuda(force_cpu=False):
    use_cuda = torch.cuda.is_available()

    return True if use_cuda and not force_cpu else False

def set_up_logging_logger(logger, filename=None, level=logging.INFO, format="[%(asctime)s] [%(name)s] [%(levelname)s] [%(module)s:%(lineno)d] %(message)s",
                          display_when_file=False):
    handlers = [
        logging.StreamHandler()
    ]

    if filename is not None:
        if display_when_file:
            # Logging messages will be stored and displayed
            handlers.append(logging.FileHandler(filename))
        else:
            # Logging messages will be stored and not displayed
            handlers[0] = logging.FileHandler(filename)

    formatter = logging.Formatter(format)
    set_formatter = True

    for h in logger.handlers:
        if h.formatter is not None and h.formatter._fmt == formatter._fmt and isinstance(h, logging.StreamHandler):
            set_formatter = False

    if set_formatter:
        for h in handlers:
            h.setFormatter(formatter)
            logger.addHandler(h)

    logger.setLevel(level)

    logger.propagate = False # We don't want to see the messages multiple times

    return logger

def string2list(s):
    assert isinstance(s, str) or isinstance(s, list)

    if isinstance(s, str) and s.strip() == '':
        return []

    return [s] if isinstance(s, str) else s

def dict_or_default(d, k, default_value):
    return d[k] if k in d else default_value

def set_random_seed(seed: int, using_cuda: bool = False) -> None:
    """
    Seed the different random generators.

    :param seed:
    :param using_cuda:
    """
    # Seed python RNG
    random.seed(seed)
    # Seed numpy RNG
    np.random.seed(seed)
    # seed the RNG for all devices (both CPU and CUDA)
    torch.manual_seed(seed)

    if using_cuda:
        # Deterministic operations for CuDNN, it may impact performances
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def file_exists(path):
    r = os.path.isfile(path)

    return r

def insert_embeddings(urls, embeddings, index, urls_representation, urls_representation_url2idx, action_dim, update_representation=True):
    embeddings = embeddings_index_sanity_check(embeddings, last_dimmension_shape=action_dim)

    index.add(embeddings)

    if update_representation:
        assert len(urls) == embeddings.shape[0], f"Different length for embeddings and URLs: {embeddings.shape} vs {len(urls)}"

        for url in urls:
            assert url not in urls_representation.values()
            assert url not in urls_representation_url2idx.keys()
            assert len(urls_representation) == len(urls_representation_url2idx)

            urls_representation[len(urls_representation)] = url
            urls_representation_url2idx[url] = len(urls_representation_url2idx)

def embeddings_index_sanity_check(embeddings, last_dimmension_shape=-1, max_expected_dim=2):
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()
    else:
        embeddings = np.array(embeddings)

    if last_dimmension_shape >= 0 and embeddings.shape[-1] != last_dimmension_shape:
        raise Exception(f"Embeddings last dimmension was expected to be {last_dimmension_shape}, but got {embeddings.shape[-1]}")

    if len(embeddings.shape) != max_expected_dim:
        if len(embeddings.shape) == 1:
            embeddings = np.array([embeddings])
        else:
            raise Exception(f"The embeddings shape length was expected to be either 1 or 2, but got {embeddings.shape}")

    return embeddings

def encode_base64(s):
    return base64.b64encode(s.encode('utf-8')).decode('utf-8')

def batchify(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i:i+batch_size]

def last_one_indices(x):
    assert isinstance(x, torch.Tensor), "Input must be a PyTorch tensor"
    assert len(x.shape) == 2, "Input tensor must be 2D (batch_size, sequence_length)"

    # Reverse along dim 1
    reversed_x = x.flip(dims=[1])

    # Find first '1' in reversed (which is last '1' in original)
    idx_reversed = reversed_x.float().argmax(dim=1)

    # If a row has no 1s, argmax will return 0, so we mask those
    has_one = x.any(dim=1)

    # Compute original index by subtracting from size
    last_one_idx = x.size(1) - 1 - idx_reversed

    # Assign -1 to rows with no 1s
    last_one_idx[~has_one] = -1

    return last_one_idx

def softmax(x):
    softmax_f = lambda x: F.softmax(x, dim=-1)

    if isinstance(x, torch.Tensor):
        return softmax_f(x)
    elif isinstance(x, np.ndarray):
        return softmax_f(torch.tensor(x)).numpy()
    elif isinstance(x, list):
        return softmax_f(torch.tensor(x)).numpy().tolist()
    else:
        raise Exception(f"Unsupported type for softmax: {type(x)}. Expected torch.Tensor, np.ndarray, or list")

def l2_normalize(emb):
    assert isinstance(emb, np.ndarray), "Input must be a numpy array"
    assert len(emb.shape) == 2, "Input array must be 2D (batch_size, embedding_size)"

    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0

    return emb / norms
