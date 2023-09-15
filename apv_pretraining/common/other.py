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
    for index in indices:  #range(0, 25)
        ################################################################################################################
        # 从inputs中抽取1个数据作为当前输入inp
        # inputs = (swap(embed), swap(is_first))    (tensor(25, 16, 1536), tensor(25, 16))
        # ↓
        # inp = (swap(embed)[index], swap(is_first)[index])，((16, 1536), (16, )) 
        ################################################################################################################
        inp = tf.nest.map_structure(lambda x: x[index], inputs)
        ################################################################################################################
        # fn: lambda prev, inputs: self.obs_step(prev[0], *inputs)
        # fn(last, inp)实际的形式为fn(last[0], inp[0], inp[1])
        # 由于last是fn上一次的输出, 即上一个(post, prior)
        # 因此fn的实际输入形式为fn(post, embed[index], is_first[index])
        #    fn的实际输出形式为(post, prior)
        ################################################################################################################
        last = fn(last, inp)
        ################################################################################################################
        # last=(post, prior)=((stoch_post, logit_post, deter0_post), (stoch_pre, logit_pre, deter0_pre))
        # 所以tf.nest.flatten(last)= [stoch_post, logit_post, deter0_post, stoch_pre, logit_pre, deter0_pre]
        ################################################################################################################
        [o.append(l) for o, l in zip(outputs, tf.nest.flatten(last))]
    if reverse:
        outputs = [list(reversed(x)) for x in outputs]
    ####################################################################################################################
    # outputs: [list_stoch_post, list_logit_post, list_deter0_post, list_stoch_ppre, list_logit_pre, list_deter0_pre]
    #          [25个16x1024,     25个16x32x32  ,  25个16x32x32    , 25个16x1024    , 25个16x32x32 ,  25个16x32x32   ]
    # 
    # after "outputs = [tf.stack(x, 0) for x in outputs])":
    #          [all_stoch_post, all_logit_post, all_deter0_post, all_stoch_ppre, all_logit_pre, all_deter0_pre]
    #          [25x16x1024,     25x16x32x32  ,  25x16x32x32    , 25x16x1024    , 25x16x32x32 ,  25x16x32x32   ]
    ####################################################################################################################
    outputs = [tf.stack(x, 0) for x in outputs]
    ####################################################################################################################
    # 将outputs封装成start的格式，即(post, prior)
    # [all_stoch_post, all_logit_post, all_deter0_post, all_stoch_ppre, all_logit_pre, all_deter0_pre]
    # [25x16x1024,     25x16x32x32  ,  25x16x32x32    , 25x16x1024    , 25x16x32x32 ,  25x16x32x32   ]
    # ↓
    # (post, prior)
    # ({'logit': 25x16x1024, 'stoch': 25x16x32x32, 'deter0': 25x16x32x32},
    #  {'logit': 25x16x1024, 'stoch': 25x16x32x32, 'deter0': 25x16x32x32})
    ####################################################################################################################
    return tf.nest.pack_sequence_as(start, outputs)

class CarryOverState:
    def __init__(self, fn):
        self._fn = fn
        self._state = None

    def __call__(self, *args):
        self._state, out = self._fn(*args, self._state)
        return out
