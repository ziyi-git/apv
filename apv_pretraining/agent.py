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
        ##############################################################################################################
        # 疑问: 根据"train_agent = common.CarryOverState(agnt.train)"，state为一个chunk的最后一个post，该state会作为
        # "self.wm.train(data, state)"中的state继续参与下一个chunk的训练过程，即作为下一个chunk的起点的z_t-1角色。问题是:
        # 由于_sample_sequence()每一次都会随机选择一个episode且随机选择一个sequence，因此两个chunk之间是不连续的？那么下面的
        # 调用方法，似乎是有问题的。
        ##############################################################################################################
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
        ################################################################################################################
        # config.encoder:
        # {
        #     'mlp_keys': '$^',
        #     'cnn_keys': 'image', 
        #     'act': 'elu', 
        #     'norm': 'none', 
        #     'cnn_depth': 48, 
        #     'cnn_kernels': (4, 4, 4, 4), 
        #     'mlp_layers': (400, 400, 400, 400)
        # }
        # 1. 在encoder中没有要mlp, 所以用了一个很难被匹配的正则表达式'$^'
        # 2. config.encoder的类型并不是dict, type(config.encoder) >>> <class 'common.config.Config'>，但仍然可以用
        #    **config.encoder解包传给形参
        # NOTE: config到底是如何构建的，还需要详细理解train_video.py中的代码.
        ################################################################################################################
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
        ################################################################################################################
        #
        #
        ################################################################################################################
        feat = self.rssm.get_feat(post)
        for name, head in self.heads.items():
            grad_head = name in self.config.grad_heads
            inp = feat if grad_head else tf.stop_gradient(feat)
            out = head(inp)
            dists = out if isinstance(out, dict) else {name: out}
            for key, dist in dists.items():
                ########################################################################################################
                # [1] 假设dist是decoder的输出, shape为(64, 64, 3)。data['image']也是(64, 64, 3)
                # [2] dist.log_prob将dist和data['image']进行逐元素的比较
                # [3] 假设用p_d代表dist分布，逐元素比较其实是(以data['image'][i][j][k]为例)：将data['image'][i][j][k]代入p_d,
                #     计算出y_hat = p_d(data['image'][i][j][k])，而目标的y应该是1(data['image'][i][j][k]是已经发生的事)，所以
                #     y_hat应该是越接近1越好。从代码来看，dist.log_prob(data[key]应该越大越好。
                # 用一个具体的例子来帮助理解:
                # 我们用形状为(3,)的dist和image来代替说明形状为(64, 64, 3)的dist和image:
                # 假设image为: [1.0, 2.0, 3.0]
                # 假设dist是一个多元高斯分布，平均值为: [0.0, 1.0, 2.0], 标准差为: [1.0, 1.0, 1.0]
                # 那么对于image中的每个元素，log_prob在内部用
                # \log p(x) = \frac{1}{2} \log (2 \pi \sigma ^2) - \frac{(x - \mu ^2)}{2 \sigma ^2}
                # 来进行计算
                # 对于image[0], 配套的参数为 \mu = 0.0, \sigma = 1.0, 根据上述公式的计算结果是-1.42
                # 对于image[1], 配套的参数为 \mu = 1.0, \sigma = 1.0, 根据上述公式的计算结果是-1.42
                # 对于image[2], 配套的参数为 \mu = 2.0, \sigma = 1.0, 根据上述公式的计算结果是-1.42
                # 因此，dist.log_prob(image) = -4.26
                # 每个p(x)肯定是越接近1越好，那么logp(x)肯定是越接近0越好，对应的-logp(x)也是越接近0越好，越小越好。
                # 相当于image中的x是已经发生的事，那么这个已经发生的事，代入p(x)时，当然是越接近1越好。
                ########################################################################################################
                like = tf.cast(dist.log_prob(data[key]), tf.float32)  # 可以理解为解码输出的dist和data['image']的相似性？
                likes[key] = like
                losses[key] = -like.mean()  # losses["image"]
        ################################################################################################################
        # losses是一个字典 {'kl': 100, 'image': 100}，这里假设'kl'和'image'都是100.
        # self.config.loss_scales是一个字典{'kl': 0.1}，它代表losses['kl']的权重为0.1
        # 注意没有存储'image'的权重，因此在self.config.loss_scales.get(k, 1.0)时，用1.0这个参数来代替'image'的权重，他表示如果找
        # 不到对应的权重，就用1.0代替。
        # 因此下面的计算过程是：
        # (self.config.loss_scales['kl'] * losses['kl']) + (self.config.loss_scales['image'] * 0.1)
        ################################################################################################################
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
        ################################################################################################################
        # data['image'] --> func encoder --> embed --> func observe --> states --> post --> func get_feat --> feat
        #                                         
        #                                                                                                  <-- recon
        #  
        # truth(data['image'])
        ################################################################################################################
        decoder = self.heads["decoder"]
        truth = data[key][:6] + 0.5  # 从batch中取前6个
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
