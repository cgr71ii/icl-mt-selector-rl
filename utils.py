
import logging

import torch as th

def use_cuda(force_cpu=False):
    use_cuda = th.cuda.is_available()

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
