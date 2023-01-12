#!/usr/bin/env python
__author__ = "XXX"
__email__ = "XXX"

from time import time
from functools import wraps


def timing(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        ts = time()
        result = f(*args, **kwargs)
        te = time()
        print(
            "func:{}\n args:[{}, {}]\n took: {} sec".format(
                f.__name__, args, kwargs, te - ts
            )
        )
        return result

    return wrap