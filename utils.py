
import os
import random
import base64
import logging
import hashlib
from urllib3.util.retry import Retry

import torch
import torch.nn.functional as F
import numpy as np
import requests
from requests.adapters import HTTPAdapter
import transformers

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

def dict_or_default(d, k, default_value, f=None):
    return (f(d[k]) if f is not None else d[k]) if k in d else default_value

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

def insert_embeddings(urls, embeddings, index, urls_representation, urls_representation_url2idx, action_dim, update_representation=True, check_l2_norm=True):
    embeddings = embeddings_index_sanity_check(embeddings, last_dimmension_shape=action_dim, check_l2_norm=check_l2_norm)

    index.add(embeddings)

    if update_representation:
        assert len(urls) == embeddings.shape[0], f"Different length for embeddings and URLs: {embeddings.shape} vs {len(urls)}"

        for url in urls:
            assert url not in urls_representation.values()
            assert url not in urls_representation_url2idx.keys()
            assert len(urls_representation) == len(urls_representation_url2idx)

            urls_representation[len(urls_representation)] = url
            urls_representation_url2idx[url] = len(urls_representation_url2idx)

def embeddings_index_sanity_check(embeddings, last_dimmension_shape=-1, max_expected_dim=2, check_l2_norm=True, check_not_nan=True, check_not_inf=True):
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy() # watch out!
    else:
        assert isinstance(embeddings, np.ndarray) or isinstance(embeddings, list), type(embeddings)

        embeddings = np.array(embeddings)

    if check_not_nan:
        count_nan = np.isnan(embeddings).any(axis=len(embeddings.shape) - 1).sum()

        assert not np.isnan(embeddings).any(), f"Embeddings contain NaN values ({count_nan}): {embeddings.shape}: {embeddings}"

    if check_not_inf:
        count_inf = np.isinf(embeddings).any(axis=len(embeddings.shape) - 1).sum()

        assert not np.isinf(embeddings).any(), f"Embeddings contain inf values ({count_inf}): {embeddings.shape}: {embeddings}"

    if check_l2_norm:
        c, v = check_l2_normalized(embeddings)
        assert c, f"Embeddings are not L2 normalized: {v}: {embeddings}"

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

def l2_normalize(emb, eps=1e-6):
    assert isinstance(emb, (np.ndarray, torch.Tensor)), "Input must be a numpy array or a PyTorch tensor"

    if isinstance(emb, np.ndarray):
        norms = np.linalg.norm(emb, axis=-1, keepdims=True)
        result = emb / (norms + eps)

        assert np.all((-1 <= result) & (result <= 1)), f"L2 normalization failed: {result}"
    else:
        norms = torch.norm(emb, dim=-1, keepdim=True)
        result = emb / (norms + eps)

        assert torch.all((-1 <= result) & (result <= 1)), f"L2 normalization failed: {result}"

    return result

def check_l2_normalized(emb, tol=1e-1):
    assert isinstance(emb, (np.ndarray, torch.Tensor)), "Input must be a numpy array or a PyTorch tensor"

    if isinstance(emb, np.ndarray):
        norms = np.linalg.norm(emb, axis=-1)
    else:
        norms = torch.norm(emb, dim=-1)

    v = np.abs(norms.cpu().numpy() - 1) if isinstance(norms, torch.Tensor) else np.abs(norms - 1)

    return np.all(v <= tol), np.sum(v).item()

def parse_args(raw_kwargs, sep='='):
    # expected format: key="value with spaces" or key=value
    # use case example: parse_args(sys.argv[1:])

    assert isinstance(raw_kwargs, list), "Expected raw_kwargs to be a list of strings"

    parsed_kwargs = {}

    for arg in raw_kwargs:
        key, _sep, value = arg.partition(sep)

        assert _sep == sep, f"Invalid argument format: {arg}"

        parsed_kwargs[key] = value

    return parsed_kwargs

def fixed_orthogonal_projection(v, out_dim, seed=42, random_matrix=None):
    assert isinstance(v, np.ndarray), "Input must be a numpy array"

    in_dim = v.shape[-1]
    rng = np.random.default_rng(seed)

    assert in_dim > out_dim, f"Input dimension {in_dim} must be greater than output dimension {out_dim}"

    # Generate random matrix and compute orthogonal basis
    _random_matrix = rng.random((in_dim, out_dim), dtype=np.float32) if random_matrix is None else random_matrix

    assert _random_matrix.shape == (in_dim, out_dim), f"Expected shape {(in_dim, out_dim)}, but got {_random_matrix.shape}"

    # QR decomposition gives orthonormal columns
    Q, _ = np.linalg.qr(_random_matrix)

    # Project the action
    result = v @ Q  # shape: (1024,)

    assert result.shape[-1] == out_dim, f"Expected output shape {out_dim}, but got {result.shape[-1]}"

    return result

def iterative_nonoverlapping_average(vec, out_dim):
    n = 0
    in_dim = vec.shape[-1]
    _out_dim = out_dim

    assert in_dim > _out_dim, f"Input dimension {in_dim} must be greater than output dimension {_out_dim}"

    while in_dim > _out_dim:
        _out_dim *= 2
        n += 1

    assert in_dim == _out_dim, f"Input dimension {vec.shape[-1]} cannot be reduced to {out_dim} by iterative non-overlapping averaging"
    assert isinstance(vec, np.ndarray)
    assert n >= 0

    for _ in range(n):
        size = vec.shape[-1]

        if size < 2:
            break

        # compute pairwise averages
        paired = (vec[...,0:size//2*2:2] + vec[...,1:size//2*2:2]) / 2
        vec = paired

        # if odd length, keep last element
        if size % 2 != 0:
            # expand last element to match shape for concatenation
            last = np.expand_dims(vec[...,-1], axis=-1)
            vec = np.concatenate([paired, last], axis=-1)

    assert vec.shape[-1] == out_dim, f"Output dimension {vec.shape[-1]} does not match expected {out_dim}" # what about when last elements are added?

    return vec

def get_hash(s, hashf="md5"):
    assert isinstance(s, str), type(s)

    f = getattr(hashlib, hashf, None)

    assert f is not None, f"{hashf} not available"

    return f(s.encode()).hexdigest()

def _requests(url, method, max_retries=5, backoff_factor=1.0, **kwargs):
    assert method in ["post", "get"], f"Unsupported method: {method}"

    retries = Retry(
        total=max_retries,
        backoff_factor=backoff_factor, # sleep: 1, 2, 4, 8, 16, ...
        allowed_methods=["HEAD", "GET", "OPTIONS", "POST"],
    )
    adapter = HTTPAdapter(max_retries=retries)
    timeout = kwargs.pop("timeout", (3 * 20, 3600)) # https://requests.readthedocs.io/en/latest/user/advanced/#timeouts

    with requests.Session() as session:
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        if method == "get":
            response = session.get(url, timeout=timeout, **kwargs)
        elif method == "post":
            response = session.post(url, timeout=timeout, **kwargs)
        else:
            raise Exception(f"Unsupported method: {method}")

    return response

def requests_post(url, max_retries=5, backoff_factor=1.0, **kwargs):
    return _requests(url, method="post", max_retries=max_retries, backoff_factor=backoff_factor, **kwargs)

def get_lr_scheduler(scheduler, optimizer, *args, **kwargs):
    scheduler_instance = None
    mandatory_args = ""

    def check_args(num_args, str_args):
        if len(args) != num_args:
            raise Exception(f"LR scheduler: '{scheduler}' mandatory args: {str_args}")

    if scheduler == "none":
        pass
    elif scheduler == "linear":
        mandatory_args = "num_warmup_steps, num_training_steps"

        check_args(2, mandatory_args)

        scheduler_instance = transformers.get_linear_schedule_with_warmup(optimizer, *args, **kwargs)
    elif scheduler == "CLR": # CyclicLR
        mandatory_args = "base_lr, max_lr"

        check_args(2, mandatory_args)

        scheduler_instance = torch.optim.lr_scheduler.CyclicLR(optimizer, *args, **kwargs)
    elif scheduler in ("inverse_sqrt", "inverse_sqrt_chichirau_et_al"):
        mandatory_args = "num_warmup_steps"

        check_args(1, mandatory_args)

        if optimizer is None:
            raise Exception(f"Optimizer not provided, so the selected LR scheduler can't be configured: {scheduler}")

        scheduler_instance = transformers.get_inverse_sqrt_schedule(optimizer, *args, **kwargs)
    else:
        raise Exception(f"Unknown LR scheduler: {scheduler}")

    return scheduler_instance, mandatory_args

def get_lr_scheduler_and_optimizer_using_argparse_values(optimizer_str, scheduler_str, optimizer_args, lr_scheduler_args, optimizer_args_params, learning_rate, training_steps, training_steps_per_epoch, logger, _optimizer=None):
    # Expected: optimizer_args and lr_scheduler_args have been processed through argparse using argparse_pytorch_conf
    # learning_rate: "used when a parameter group doesn't specify them" (class Optimizer: https://pytorch.org/docs/stable/_modules/torch/optim/optimizer.html)

    # Get optimizer
    logger.debug("Optimizer args: %s", optimizer_args)

    if _optimizer is not None:
        assert optimizer_str is None

        optimizer = _optimizer

    if optimizer_str is None:
        pass
    elif optimizer_str == "none":
        optimizer = None

        logger.debug("Be aware that even with the optimizer disabled minor changes might be observed while training since the model is "
                     "not in inference mode, so layers like Dropout have a random component which is enabled")
    elif optimizer_str == "adam":
        optimizer_kwargs = {
            "betas": tuple(optimizer_args[0:2]),
            "eps": optimizer_args[2],
            "weight_decay": optimizer_args[3],
        }
        optimizer = torch.optim.Adam(optimizer_args_params, lr=learning_rate, **optimizer_kwargs)
    elif optimizer_str in ("adamw", "adamw_no_wd", "adamw_amsgrad_no_wd"):
        optimizer_kwargs = {
            "betas": tuple(optimizer_args[0:2]),
            "eps": optimizer_args[2],
            "weight_decay": optimizer_args[3],
        }

        if optimizer_str == "adamw_amsgrad_no_wd":
            optimizer_kwargs["amsgrad"] = optimizer_args[4]

        optimizer = torch.optim.AdamW(optimizer_args_params, lr=learning_rate, **optimizer_kwargs)
    elif optimizer_str == "sgd":
        optimizer_kwargs = {
            "momentum": optimizer_args[0],
            "weight_decay": optimizer_args[1],
        }
        optimizer = torch.optim.SGD(optimizer_args_params, lr=learning_rate, **optimizer_kwargs)
    else:
        raise Exception(f"Unknown optimizer: {optimizer_str}")


    # Get LR scheduler args
    scheduler_args = []
    scheduler_kwargs = {}

    logger.debug("LR scheduler args: %s", lr_scheduler_args)

    if scheduler_str == "none":
        pass
    elif scheduler_str == "linear":
        if lr_scheduler_args[0][-1] == '%':
            scheduler_args = [int((float(lr_scheduler_args[0][:-1]) / 100.0) * training_steps), training_steps]
        else:
            scheduler_args = [int(lr_scheduler_args[0]), training_steps]
    elif scheduler_str == "CLR":
        scheduler_max_lr, scheduler_step_size, scheduler_mode, scheduler_gamma, scheduler_max_lr_factor, scheduler_step_size_factor \
            = lr_scheduler_args

        if learning_rate > scheduler_max_lr:
            new_scheduler_max_lr = learning_rate * scheduler_max_lr_factor # Based on the CLR paper (possible values are [3.0, 4.0])

            logger.warning("LR scheduler: '%s': provided LR (%f) is greater than provided max. LR (%f): setting max. LR to %f",
                        scheduler_str, learning_rate, scheduler_max_lr, new_scheduler_max_lr)

            scheduler_max_lr = new_scheduler_max_lr
        if scheduler_step_size <= 0:
            scheduler_step_size = scheduler_step_size_factor * training_steps_per_epoch # Based on the CLR paper (possible values are [2, ..., 8])

            logger.warning("LR scheduler: '%s': provided step size is 0 or negative: setting value to %d", scheduler_str, scheduler_step_size)

        scheduler_args = [learning_rate, scheduler_max_lr]
        scheduler_kwargs = {
            "step_size_up": scheduler_step_size,
            "step_size_down": scheduler_step_size,
            "mode": scheduler_mode,
            "gamma": scheduler_gamma,
            "cycle_momentum": False, # https://github.com/pytorch/pytorch/issues/73910
        }
    elif scheduler_str in ("inverse_sqrt", "inverse_sqrt_chichirau_et_al"):
        if lr_scheduler_args[0][-1] == '%':
            scheduler_args = [int((float(lr_scheduler_args[0][:-1]) / 100.0) * training_steps)]
        else:
            scheduler_args = [int(lr_scheduler_args[0])]
    else:
        raise Exception(f"Unknown LR scheduler: {scheduler}")

    scheduler, mandatory_args = get_lr_scheduler(scheduler_str, optimizer, *scheduler_args, **scheduler_kwargs)

    logger.debug("LR scheduler: '%s' mandatory args: %s: %s", scheduler_str, mandatory_args, str(scheduler_args))
    logger.debug("LR scheduler: '%s' optional args: %s", scheduler_str, str(scheduler_kwargs))

    return optimizer, scheduler
