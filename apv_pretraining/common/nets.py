import re

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as tfkl
from tensorflow_probability import distributions as tfd
from tensorflow.keras.mixed_precision import experimental as prec

import common


class EnsembleRSSM(common.Module):
    def __init__(
        self,
        ensemble=5,
        stoch=30,
        deter=200,
        hidden=200,
        discrete=False,
        act="elu",
        norm="none",
        std_act="softplus",
        min_std=0.1,
        rnn_layers=1,
    ):
        super().__init__()
        ################################################################################################################
        # self._ensemble为1
        # self._stoch为32，可能是z的维度
        # self._deter为1024，可能是h的维度
        # self._hidden为1024，？
        # self._discrete为32，是否使用随机离散变量？
        # self._act为32，是一个激活函数指针？
        # self._norm为none，？
        # self._std_act是sigmoid2，标准偏差的激活函数，是一个函数指针？
        # self._min_std为0.1，？
        # self._rnn_layers，循环神经网络层的数量
        ################################################################################################################
        self._ensemble = ensemble
        self._stoch = stoch        
        self._deter = deter
        self._hidden = hidden
        self._discrete = discrete
        self._act = get_act(act)
        self._norm = norm
        self._std_act = std_act
        self._min_std = min_std
        self._rnn_layers = rnn_layers
        ################################################################################################################
        # self._cells是隐藏层单元，而self._cell是最后一个输出单元？？？
        ################################################################################################################
        self._cells = [GRUCell(self._deter, norm=True) for _ in range(self._rnn_layers)]
        self._cell = GRUCell(self._deter, norm=True)
        ################################################################################################################
        # 将x的数据类型变为float16
        ################################################################################################################
        self._cast = lambda x: tf.cast(x, prec.global_policy().compute_dtype)

    def initial(self, batch_size):
        dtype = prec.global_policy().compute_dtype
        if self._discrete:
            state = dict(
                logit=tf.zeros([batch_size, self._stoch, self._discrete], dtype),
                stoch=tf.zeros([batch_size, self._stoch, self._discrete], dtype),
            )
        else:
            state = dict(
                mean=tf.zeros([batch_size, self._stoch], dtype),
                std=tf.zeros([batch_size, self._stoch], dtype),
                stoch=tf.zeros([batch_size, self._stoch], dtype),
            )
        for i in range(self._rnn_layers):
            state[f"deter{i}"] = self._cells[i].get_initial_state(
                None, batch_size, dtype
            )
        return state

    def zero_action(self, batch_size):
        return self._cast(tf.zeros([batch_size, 50]))

    @tf.function
    def observe(self, embed, is_first, state=None):
        swap = lambda x: tf.transpose(x, [1, 0] + list(range(2, len(x.shape))))  # (16, 25, 1536) -> (25, 16, 1536)
        if state is None:
            state = self.initial(tf.shape(embed)[0])
        post, prior = common.static_scan(
            lambda prev, inputs: self.obs_step(prev[0], *inputs),
            (swap(embed), swap(is_first)),  # (tensor(25, 16, 1536), tensor(25, 16))
            (state, state),
        )
        ################################################################################################################
        # 下面以post为例说明(prior和post的例子一样):
        # post经过swap的变换前后:
        # {'logit': 25x16x1024, 'stoch': 25x16x32x32, 'deter0': 25x16x32x32}
        # ↓
        # {'logit': 16x25x1024, 'stoch': 16x25x32x32, 'deter0': 16x25x32x32} 
        ################################################################################################################
        post = {k: swap(v) for k, v in post.items()}
        prior = {k: swap(v) for k, v in prior.items()}
        return post, prior

    @tf.function
    def imagine(self, fake_data, state):
        swap = lambda x: tf.transpose(x, [1, 0] + list(range(2, len(x.shape))))
        if state is None:
            state = self.initial(tf.shape(fake_data)[0])
        assert isinstance(state, dict), state
        prior = common.static_scan(
            lambda prev, inputs: self.img_step(prev), swap(fake_data), state
        )
        prior = {k: swap(v) for k, v in prior.items()}
        return prior

    def get_feat(self, state):
        stoch = self._cast(state["stoch"])
        if self._discrete:
            shape = stoch.shape[:-2] + [self._stoch * self._discrete]
            stoch = tf.reshape(stoch, shape)
        return tf.concat([stoch, state[f"deter{self._rnn_layers - 1}"]], -1)

    def get_dist(self, state, ensemble=False):
        if ensemble:
            state = self._suff_stats_ensemble(state[f"deter{self._rnn_layers - 1}"])
        if self._discrete:
            logit = state["logit"]
            logit = tf.cast(logit, tf.float32)
            dist = tfd.Independent(common.OneHotDist(logit), 1)
        else:
            mean, std = state["mean"], state["std"]
            mean = tf.cast(mean, tf.float32)
            std = tf.cast(std, tf.float32)
            dist = tfd.MultivariateNormalDiag(mean, std)
        return dist

    @tf.function
    def obs_step(self, prev_state, embed, is_first, sample=True):
        # if is_first.any():
        (prev_state,) = tf.nest.map_structure(
            lambda x: tf.einsum("b,b...->b...", 1.0 - is_first.astype(x.dtype), x),
            (prev_state,),
        )
        prior = self.img_step(prev_state, sample)  # deter0: h_{t}, stoch: z^_{t}, logit^
        x = tf.concat([prior[f"deter{self._rnn_layers - 1}"], embed], -1)
        x = self.get("obs_out", tfkl.Dense, self._hidden)(x)
        x = self.get("obs_out_norm", NormLayer, self._norm)(x)
        x = self._act(x)
        stats = self._suff_stats_layer("obs_dist", x)
        dist = self.get_dist(stats)
        stoch = dist.sample() if sample else dist.mode()  # h_{t} + o_{t} -> z_{t}
        post = {"stoch": stoch, **stats}
        for i in range(self._rnn_layers):
            post[f"deter{i}"] = prior[f"deter{i}"]
        return post, prior

    @tf.function
    def img_step(self, prev_state, sample=True):
        ################################################################################################################
        # prev_state -> prev_state["stoch"] -> temp -> deter(h_{t}) -> stoch(zhat_{t})
        #       ↑                                        |                 |
        #       ↑                                          --------↓--------
        #       ------------------------------------------------prior 
        ################################################################################################################
        prev_stoch = self._cast(prev_state["stoch"])
        if self._discrete:
            ############################################################################################################
            # prev_stoch: (16, 32, 32) -> (16, 1024)
            # self._discrete是什么意思？
            ############################################################################################################
            shape = prev_stoch.shape[:-2] + [self._stoch * self._discrete]
            prev_stoch = tf.reshape(prev_stoch, shape)
        ################################################################################################################
        # 计算先验h_{t}，利用公式Appendix B中公式(11):
        # h_{t} = f_phai(h_{t-1}, z_{t-1})
        # 但是，看看Appendix B中公式(13)：
        # h_{t} = f_phai(h_{t-1}, z_{t-1}, a_{t-1})
        # 也就是在计算h_{t}时，正常的action_conditional计算过程是要考虑a_{t-1}的，而且是要两步来完成计算：
        # step1: temp = f_temp(z_{t-1}, a_{t-1})
        # step2: h_{t} = f_phai(h_{t-1}, temp)
        # 因此下面"4句"实际执行了step1得到temp，由于action_free的原因，用self.zero_action(prev_stoch.shape[0])来抹掉了行为的影响
        ################################################################################################################
        x = tf.concat([prev_stoch, self.zero_action(prev_stoch.shape[0])], -1)  # (16, 1074)
        x = self.get("img_in", tfkl.Dense, self._hidden)(x)  # (16, 1024)
        x = self.get("img_in_norm", NormLayer, self._norm)(x)
        x = self._act(x)  # -> x is temp
        ################################################################################################################
        # 假设self._cells有3个GRUCell类实例串联，计算过程如下：
        #    self._cells[0]          self._cells[1]          self._cells[2]
        #
        # prev_state["deter0"]    prev_state["deter1"]    prev_state["deter2"]
        #          +                      +                       +
        #        temp           -------- temp           -------- temp
        #          ↓            |         ↓             |         ↓
        #       deter0 ----------       deter1 ----------       deter2
        #
        #     prior["deter0"]        prior["deter1"]         prior["deter2"]
        # 实际self._cells仅有1个GRUCell类实例，因此只输出了deter0，也就是h_{t}
        ################################################################################################################
        deters = []
        for i in range(self._rnn_layers):
            deter = prev_state[f"deter{i}"]
            ############################################################################################################
            # 由于self._cells[i]的call函数return output, [output]，所以x就是deter[0]
            ############################################################################################################
            x, deter = self._cells[i](x, [deter])
            deters.append(deter[0])
        deter = deter[0]  # Keras wraps the state in a list.
        stats = self._suff_stats_ensemble(x)
        index = tf.random.uniform((), 0, self._ensemble, tf.int32)
        stats = {k: v[index] for k, v in stats.items()}
        dist = self.get_dist(stats)
        stoch = dist.sample() if sample else dist.mode()
        prior = {"stoch": stoch, **stats}
        for i in range(self._rnn_layers):
            prior[f"deter{i}"] = deters[i]
        return prior

    def _suff_stats_ensemble(self, inp):
        bs = list(inp.shape[:-1])
        inp = inp.reshape([-1, inp.shape[-1]])
        stats = []
        for k in range(self._ensemble):
            x = self.get(f"img_out_{k}", tfkl.Dense, self._hidden)(inp)
            x = self.get(f"img_out_norm_{k}", NormLayer, self._norm)(x)
            x = self._act(x)
            stats.append(self._suff_stats_layer(f"img_dist_{k}", x))
        stats = {k: tf.stack([x[k] for x in stats], 0) for k, v in stats[0].items()}
        stats = {
            k: v.reshape([v.shape[0]] + bs + list(v.shape[2:]))
            for k, v in stats.items()
        }
        return stats

    def _suff_stats_layer(self, name, x):
        if self._discrete:
            x = self.get(name, tfkl.Dense, self._stoch * self._discrete, None)(x)
            logit = tf.reshape(x, x.shape[:-1] + [self._stoch, self._discrete])
            return {"logit": logit}
        else:
            x = self.get(name, tfkl.Dense, 2 * self._stoch, None)(x)
            mean, std = tf.split(x, 2, -1)
            std = {
                "softplus": lambda: tf.nn.softplus(std),
                "sigmoid": lambda: tf.nn.sigmoid(std),
                "sigmoid2": lambda: 2 * tf.nn.sigmoid(std / 2),
            }[self._std_act]()
            std = std + self._min_std
            return {"mean": mean, "std": std}

    def kl_loss(self, post, prior, forward, balance, free, free_avg):
        kld = tfd.kl_divergence
        sg = lambda x: tf.nest.map_structure(tf.stop_gradient, x)
        lhs, rhs = (prior, post) if forward else (post, prior)
        mix = balance if forward else (1 - balance)
        if balance == 0.5:
            value = kld(self.get_dist(lhs), self.get_dist(rhs))
            loss = tf.maximum(value, free).mean()
        else:
            value_lhs = value = kld(self.get_dist(lhs), self.get_dist(sg(rhs)))
            value_rhs = kld(self.get_dist(sg(lhs)), self.get_dist(rhs))
            if free_avg:
                loss_lhs = tf.maximum(value_lhs.mean(), free)
                loss_rhs = tf.maximum(value_rhs.mean(), free)
            else:
                loss_lhs = tf.maximum(value_lhs, free).mean()
                loss_rhs = tf.maximum(value_rhs, free).mean()
            loss = mix * loss_lhs + (1 - mix) * loss_rhs
        return loss, value


class Encoder(common.Module):
    def __init__(
        self,
        shapes,
        cnn_keys=r".*",
        mlp_keys=r".*",
        act="elu",
        norm="none",
        cnn_depth=48,
        cnn_kernels=(4, 4, 4, 4),
        mlp_layers=[400, 400, 400, 400],
    ):
        self.shapes = shapes  
        self.cnn_keys = [
            k for k, v in shapes.items() if re.match(cnn_keys, k) and len(v) == 3
        ]
        self.mlp_keys = [
            k for k, v in shapes.items() if re.match(mlp_keys, k) and len(v) == 1
        ]
        print("Encoder CNN inputs:", list(self.cnn_keys))
        print("Encoder MLP inputs:", list(self.mlp_keys))
        self._act = get_act(act)
        self._norm = norm
        self._cnn_depth = cnn_depth
        self._cnn_kernels = cnn_kernels
        self._mlp_layers = mlp_layers

    @tf.function
    def __call__(self, data):
        ################################################################################################################
        # __call__将每一幅图像编码为1 x 1536的向量
        # self.shapes就是obs_space:
        # {
        #  'image': Box(0, 255, (64, 64,...3), uint8), 
        #  'reward': Box(-inf, inf, (), float32), 
        #  'is_first': Box(False, True, (), bool),
        #  'is_last': Box(False, True, (), bool),
        #  'is_terminal': Box(False, True, (), bool),
        #  'state': Box([-0.525   0.348 ..., float32),
        #  'success': Box(False, True, (), bool)
        # }
        # key为'image', shape为(64, 64, 3)
        ################################################################################################################
        key, shape = list(self.shapes.items())[0]
        batch_dims = data[key].shape[: -len(shape)] 
        data = {
            k: tf.reshape(v, (-1,) + tuple(v.shape)[len(batch_dims) :])
            for k, v in data.items()
        }  # [16, 25, 64, 64, 3] -> [400, 64, 64, 3]
        outputs = []
        if self.cnn_keys:
            outputs.append(self._cnn({k: data[k] for k in self.cnn_keys}))
        if self.mlp_keys:
            outputs.append(self._mlp({k: data[k] for k in self.mlp_keys}))
        output = tf.concat(outputs, -1)
        return output.reshape(batch_dims + output.shape[1:])  # (400, 1536) -> (16, 25, 1536)

    def _cnn(self, data):
        ################################################################################################################
        # data实际为{'image': <tf.Tensor: shape=(400, 64, 64 ,3)}，400是因为16 * 25
        ################################################################################################################
        x = tf.concat(list(data.values()), -1)
        x = x.astype(prec.global_policy().compute_dtype)
        ################################################################################################################
        # (400, 64, 64 ,3) -> (400, 31, 31, 48)  -> (400, 14, 14, 96) -> (400, 6, 6, 192) -> (400, 2, 2, 384)
        ################################################################################################################
        for i, kernel in enumerate(self._cnn_kernels):
            depth = 2 ** i * self._cnn_depth  # ? 2^i * 48
            ############################################################################################################
            # 检查是否有名为"conv{i}"的层，没有则创建：channel数为depth，kernel size为kernel，步长为2
            ############################################################################################################
            x = self.get(f"conv{i}", tfkl.Conv2D, depth, kernel, 2)(x)
            x = self.get(f"convnorm{i}", NormLayer, self._norm)(x)
            x = self._act(x)
        ################################################################################################################
        # (400, 2, 2, 384) -> (400, 1536)
        ################################################################################################################
        return x.reshape(tuple(x.shape[:-3]) + (-1,))

    def _mlp(self, data):
        x = tf.concat(list(data.values()), -1)
        x = x.astype(prec.global_policy().compute_dtype)
        for i, width in enumerate(self._mlp_layers):
            x = self.get(f"dense{i}", tfkl.Dense, width)(x)
            x = self.get(f"densenorm{i}", NormLayer, self._norm)(x)
            x = self._act(x)
        return x


class Decoder(common.Module):
    def __init__(
        self,
        shapes,
        cnn_keys=r".*",
        mlp_keys=r".*",
        act="elu",
        norm="none",
        cnn_depth=48,
        cnn_kernels=(4, 4, 4, 4),
        mlp_layers=[400, 400, 400, 400],
    ):
        self._shapes = shapes
        self.cnn_keys = [
            k for k, v in shapes.items() if re.match(cnn_keys, k) and len(v) == 3
        ]
        self.mlp_keys = [
            k for k, v in shapes.items() if re.match(mlp_keys, k) and len(v) == 1
        ]
        print("Decoder CNN outputs:", list(self.cnn_keys))
        print("Decoder MLP outputs:", list(self.mlp_keys))
        self._act = get_act(act)
        self._norm = norm
        self._cnn_depth = cnn_depth
        self._cnn_kernels = cnn_kernels
        self._mlp_layers = mlp_layers

    def __call__(self, features):
        features = tf.cast(features, prec.global_policy().compute_dtype)
        outputs = {}
        if self.cnn_keys:
            outputs.update(self._cnn(features))
        if self.mlp_keys:
            outputs.update(self._mlp(features))
        return outputs

    def _cnn(self, features):
        channels = {k: self._shapes[k][-1] for k in self.cnn_keys}
        ConvT = tfkl.Conv2DTranspose
        x = self.get("convin", tfkl.Dense, 32 * self._cnn_depth)(features)
        x = tf.reshape(x, [-1, 1, 1, 32 * self._cnn_depth])
        for i, kernel in enumerate(self._cnn_kernels):
            depth = 2 ** (len(self._cnn_kernels) - i - 2) * self._cnn_depth
            act, norm = self._act, self._norm
            if i == len(self._cnn_kernels) - 1:
                depth, act, norm = sum(channels.values()), tf.identity, "none"
            x = self.get(f"conv{i}", ConvT, depth, kernel, 2)(x)
            x = self.get(f"convnorm{i}", NormLayer, norm)(x)
            x = act(x)
        x = x.reshape(features.shape[:-1] + x.shape[1:])
        means = tf.split(x, list(channels.values()), -1)
        dists = {
            key: tfd.Independent(tfd.Normal(mean, 1), 3)
            for (key, shape), mean in zip(channels.items(), means)
        }
        return dists

    def _mlp(self, features):
        shapes = {k: self._shapes[k] for k in self.mlp_keys}
        x = features
        for i, width in enumerate(self._mlp_layers):
            x = self.get(f"dense{i}", tfkl.Dense, width)(x)
            x = self.get(f"densenorm{i}", NormLayer, self._norm)(x)
            x = self._act(x)
        dists = {}
        for key, shape in shapes.items():
            dists[key] = self.get(f"dense_{key}", DistLayer, shape)(x)
        return dists


class MLP(common.Module):
    def __init__(self, shape, layers, units, act="elu", norm="none", **out):
        self._shape = (shape,) if isinstance(shape, int) else shape
        self._layers = layers
        self._units = units
        self._norm = norm
        self._act = get_act(act)
        self._out = out

    def __call__(self, features):
        x = tf.cast(features, prec.global_policy().compute_dtype)
        x = x.reshape([-1, x.shape[-1]])
        for index in range(self._layers):
            x = self.get(f"dense{index}", tfkl.Dense, self._units)(x)
            x = self.get(f"norm{index}", NormLayer, self._norm)(x)
            x = self._act(x)
        x = x.reshape(features.shape[:-1] + [x.shape[-1]])
        return self.get("out", DistLayer, self._shape, **self._out)(x)


class GRUCell(tf.keras.layers.AbstractRNNCell):
    def __init__(self, size, norm=False, act="tanh", update_bias=-1, **kwargs):
        super().__init__()
        self._size = size
        self._act = get_act(act)
        self._norm = norm
        self._update_bias = update_bias
        self._layer = tfkl.Dense(3 * size, use_bias=norm is not None, **kwargs)
        if norm:
            self._norm = tfkl.LayerNormalization(dtype=tf.float32)

    @property
    def state_size(self):
        return self._size

    @tf.function
    def call(self, inputs, state):
        state = state[0]  # Keras wraps the state in a list.
        parts = self._layer(tf.concat([inputs, state], -1))
        if self._norm:
            dtype = parts.dtype
            parts = tf.cast(parts, tf.float32)
            parts = self._norm(parts)
            parts = tf.cast(parts, dtype)
        reset, cand, update = tf.split(parts, 3, -1)
        reset = tf.nn.sigmoid(reset)
        cand = self._act(reset * cand)
        update = tf.nn.sigmoid(update + self._update_bias)
        output = update * cand + (1 - update) * state
        return output, [output]


class DistLayer(common.Module):
    def __init__(self, shape, dist="mse", min_std=0.1, init_std=0.0):
        self._shape = shape
        self._dist = dist
        self._min_std = min_std
        self._init_std = init_std

    def __call__(self, inputs):
        out = self.get("out", tfkl.Dense, np.prod(self._shape))(inputs)
        out = tf.reshape(out, tf.concat([tf.shape(inputs)[:-1], self._shape], 0))
        out = tf.cast(out, tf.float32)
        if self._dist in ("normal", "tanh_normal", "trunc_normal"):
            std = self.get("std", tfkl.Dense, np.prod(self._shape))(inputs)
            std = tf.reshape(std, tf.concat([tf.shape(inputs)[:-1], self._shape], 0))
            std = tf.cast(std, tf.float32)
        if self._dist == "mse":
            dist = tfd.Normal(out, 1.0)
            return tfd.Independent(dist, len(self._shape))
        if self._dist == "normal":
            dist = tfd.Normal(out, std)
            return tfd.Independent(dist, len(self._shape))
        if self._dist == "binary":
            dist = tfd.Bernoulli(out)
            return tfd.Independent(dist, len(self._shape))
        if self._dist == "tanh_normal":
            mean = 5 * tf.tanh(out / 5)
            std = tf.nn.softplus(std + self._init_std) + self._min_std
            dist = tfd.Normal(mean, std)
            dist = tfd.TransformedDistribution(dist, common.TanhBijector())
            dist = tfd.Independent(dist, len(self._shape))
            return common.SampleDist(dist)
        if self._dist == "trunc_normal":
            std = 2 * tf.nn.sigmoid((std + self._init_std) / 2) + self._min_std
            dist = common.TruncNormalDist(tf.tanh(out), std, -1, 1)
            return tfd.Independent(dist, 1)
        if self._dist == "onehot":
            return common.OneHotDist(out)
        raise NotImplementedError(self._dist)


class NormLayer(common.Module):
    def __init__(self, name):
        if name == "none":
            self._layer = None
        elif name == "layer":
            self._layer = tfkl.LayerNormalization()
        else:
            raise NotImplementedError(name)

    def __call__(self, features):
        if not self._layer:
            return features
        return self._layer(features)


def get_act(name):
    if name == "none":
        return tf.identity
    if name == "mish":
        return lambda x: x * tf.math.tanh(tf.nn.softplus(x))
    elif hasattr(tf.nn, name):
        return getattr(tf.nn, name)
    elif hasattr(tf, name):
        return getattr(tf, name)
    else:
        raise NotImplementedError(name)


def get_decoder_dist(name):
    if name == "l2":
        return tfd.Normal
    elif name == "l1":
        return tfd.Laplace
    else:
        raise NotImplementedError(name)
