import tensorflow as tf
from tensorflow.keras import mixed_precision as prec

import common


class Agent(common.Module):
    def __init__(self, config, obs_space, act_space, step):
        self.config = config
        self.obs_space = obs_space
        self.act_space = act_space["action"]
        self.step = step
        self.tfstep = tf.Variable(int(self.step), tf.int64)
        self.wm = WorldModel(config, obs_space, self.tfstep)

    @tf.function
    def train(self, data, state=None):
        metrics = {}
        # state >>
        # {'logit': *, 'stoch': *, 'deter0': *}
        # outputs >>
        # {'embed': *, 'feat': *, 'post': *, 'prior': *, 'likes': *, 'kl': *}
        # mets >>
        # {'kl_loss' : *, 'image_loss': *, 'model_kl': *, 'prior_ent': *, 
        #  'post_ent': *, 'model_loss': *, 'model_loss_scale': *, 
        #  'model_grad_norm': *}
        state, outputs, mets = self.wm.train(data, state)
        metrics.update(mets)
        start = outputs["post"]  # start not used?
        return state, metrics

    @tf.function
    def report(self, data):
        report = {}
        data = self.wm.preprocess(data)
        for key in self.wm.heads["decoder"].cnn_keys:
            name = key.replace("/", "_")
            report[f"openl_{name}"] = self.wm.video_pred(data, key)
        return report


class WorldModel(common.Module):
    def __init__(self, config, obs_space, tfstep):
        shapes = {k: tuple(v.shape) for k, v in obs_space.items()}
        self.config = config
        self.tfstep = tfstep
        self.rssm = common.EnsembleRSSM(**config.rssm)
        self.encoder = common.Encoder(shapes, **config.encoder)
        self.heads = {}
        self.heads["decoder"] = common.Decoder(shapes, **config.decoder)
        for name in config.grad_heads:
            assert name in self.heads, name
        self.model_opt = common.Optimizer("model", **config.model_opt)

    def train(self, data, state=None):
        # 创建上下文，在前向传播过程中的中间变量，这些中间变量用于计算梯度dW：
        # (1) trainable=True的tf.Varable，例如W
        # (2) 神经元的激活值，例如A
        with tf.GradientTape() as model_tape:
            model_loss, state, outputs, metrics = self.loss(data, state)
        # [self.encoder, self.rssm, *self.heads.values()]
        # *self.heads.values()是解包操作，等价于
        # [self.encoder, self.rssm] + list(self.heads.values())
        modules = [self.encoder, self.rssm, *self.heads.values()]
        # self.model_opt(model_tape, model_loss, modules)更新模型参数，返回
        # {"model_loss": *, "model_grad_norm": *, "model_loss_scale": *}
        # 该字典添加到metrics中
        metrics.update(self.model_opt(model_tape, model_loss, modules))
        return state, outputs, metrics

    def loss(self, data, state=None):
        data = self.preprocess(data)
        embed = self.encoder(data)
        post, prior = self.rssm.observe(embed, data["is_first"], state)
        # self.config.kl: {'free': 0.0, 'forward': False, 'balance': 0.8, 'free_avg': True}
        kl_loss, kl_value = self.rssm.kl_loss(post, prior, **self.config.kl)
        assert len(kl_loss.shape) == 0
        likes = {}
        losses = {"kl": kl_loss}
        feat = self.rssm.get_feat(post)
        for name, head in self.heads.items():
            grad_head = name in self.config.grad_heads
            inp = feat if grad_head else tf.stop_gradient(feat)
            out = head(inp)
            dists = out if isinstance(out, dict) else {name: out}
            for key, dist in dists.items():
                like = tf.cast(dist.log_prob(data[key]), tf.float32)  # 可以理解为解码输出的dist和data['image']的相似性？
                likes[key] = like
                losses[key] = -like.mean()  # losses["image"]
        model_loss = sum(
            self.config.loss_scales.get(k, 1.0) * v for k, v in losses.items()
        )  # 这里将（1）post与prior之间的相似性，（2）以及解码后与输入图像之间的相似性，进行了综合计算得到model_loss用于更新模型？
        outs = dict(
            embed=embed, feat=feat, post=post, prior=prior, likes=likes, kl=kl_value
        )
        metrics = {f"{name}_loss": value for name, value in losses.items()}
        metrics["model_kl"] = kl_value.mean()
        metrics["prior_ent"] = self.rssm.get_dist(prior).entropy().mean()
        metrics["post_ent"] = self.rssm.get_dist(post).entropy().mean()
        last_state = {k: v[:, -1] for k, v in post.items()}
        return model_loss, last_state, outs, metrics

    @tf.function
    def preprocess(self, obs):
        dtype = prec.global_policy().compute_dtype
        obs = obs.copy()
        for key, value in obs.items():
            if key.startswith("log_"):
                continue
            if value.dtype == tf.int32:
                value = value.astype(dtype)
            if value.dtype == tf.uint8:
                value = value.astype(dtype) / 255.0 - 0.5
            obs[key] = value
        return obs

    @tf.function
    def video_pred(self, data, key):
        decoder = self.heads["decoder"]
        truth = data[key][:6] + 0.5
        embed = self.encoder(data)
        states, _ = self.rssm.observe(embed[:6, :5], data["is_first"][:6, :5])
        recon = decoder(self.rssm.get_feat(states))[key].mode()[:6]
        init = {k: v[:, -1] for k, v in states.items()}
        prior = self.rssm.imagine(data["is_first"][:6, 5:], init)
        openl = decoder(self.rssm.get_feat(prior))[key].mode()
        model = tf.concat([recon[:, :5] + 0.5, openl + 0.5], 1)
        error = (model - truth + 1) / 2
        video = tf.concat([truth, model, error], 2)
        B, T, H, W, C = video.shape
        return video.transpose((1, 2, 0, 3, 4)).reshape((T, H, B * W, C))
