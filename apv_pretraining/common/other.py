import collections
import contextlib
import re
import time

import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from . import dists
from . import tfutils


def static_scan(fn, inputs, start, reverse=False):
    last = start
    outputs = [[] for _ in tf.nest.flatten(start)]
    indices = range(tf.nest.flatten(inputs)[0].shape[0])  # 将嵌套结构（例如列表中的列表，或字典中的列表等）“展平”为一个一维列表
    if reverse:
        indices = reversed(indices)
    for index in indices:
        ################################################################################################################
        # inp = (embed[index], is_first[index])，((16, 1536), (16, )) 
        ################################################################################################################
        inp = tf.nest.map_structure(lambda x: x[index], inputs)
        ################################################################################################################
        # fn: lambda prev, inputs: self.obs_step(prev[0], *inputs)
        # fn(last, inp)实际的形式为fn(last[0], inp[0], inp[1])
        # 由于last是fn上一次的输出(last_post, last_prior)，因此fn的实际形式为fn(last_post, embed[index], is_first[index])
        ################################################################################################################
        last = fn(last, inp)
        [o.append(l) for o, l in zip(outputs, tf.nest.flatten(last))]
    if reverse:
        outputs = [list(reversed(x)) for x in outputs]
    outputs = [tf.stack(x, 0) for x in outputs]
    return tf.nest.pack_sequence_as(start, outputs)


class CarryOverState:
    def __init__(self, fn):
        self._fn = fn
        self._state = None

    def __call__(self, *args):
        self._state, out = self._fn(*args, self._state)
        return out
