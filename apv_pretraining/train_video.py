import collections
import functools
import logging
import os
import pathlib
import re
import sys
import warnings

try:
    import rich.traceback

    rich.traceback.install()
except ImportError:
    pass

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger().setLevel("ERROR")
warnings.filterwarnings("ignore", ".*box bound precision lowered.*")

sys.path.append(str(pathlib.Path(__file__).parent))
sys.path.append(str(pathlib.Path(__file__).parent.parent))

import numpy as np
import ruamel.yaml as yaml

import agent
import common


def main():

    configs = yaml.safe_load(
        (pathlib.Path(sys.argv[0]).parent / "configs.yaml").read_text()
    )
    parsed, remaining = common.Flags(configs=["defaults"]).parse(known_only=True)
    config = common.Config(configs["defaults"])
    for name in parsed.configs:
        config = config.update(configs[name])
    config = common.Flags(config).parse(remaining)

    logdir = pathlib.Path(config.logdir).expanduser()
    load_logdir = pathlib.Path(config.load_logdir).expanduser()
    logdir.mkdir(parents=True, exist_ok=True)
    config.save(logdir / "config.yaml")
    print(config, "\n")
    print("Logdir", logdir)
    print("Loading Logdir", load_logdir)

    import tensorflow as tf

    # tf.config.experimental_run_functions_eagerly(True or False)用于配制TF的计算模式
    # True: Eager Execution模式用于debug，不创建后续的计算图
    # False: JIT(Just-In-Time)模式在运行时将计算图编译成机器码，提高执行速度
    # tf.config.experimental_run_functions_eagerly(not config.jit)
    tf.config.experimental_run_functions_eagerly(True)
    message = "No GPU found. To actually train on CPU remove this assert."
    assert tf.config.experimental.list_physical_devices("GPU"), message
    # 设置每一块GPU的内存增长方式
    # True: 在需要时再申请内存，而不是一开始就申请所有可用的GPU内存
    for gpu in tf.config.experimental.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    # 断言条件config.precision in (16, 32)为False则将config.precision作为异常抛出
    assert config.precision in (16, 32), config.precision
    if config.precision == 16:
        from tensorflow.keras.mixed_precision import experimental as prec

        prec.set_policy(prec.Policy("mixed_float16"))

    # 来自于GPT-4的code interpreter的解释（未验证）：
    # ReplayWithoutAction继承自Replay，主要覆盖了_generate_chunks方法
    # 从论文中的“Action-free Pre-training from Videos”部分来看，它关注
    # 的应该是三个模型中的transition model，即状态的转移过程。但不关注是什么
    # action导致了状态的转移。
    #
    # transition model的作用是什么？
    # transition model仅仅从上一个状态来预测当前状态。这可能是用于生成一个状态
    # 转移的序列。这种状态转移的序列是否准确，即是否可以用作智能体在环境中的经验？
    # 论文中加入了一个前置条件：transition model需要和representation model
    # 的预测越接近越越好。
    train_replay = common.ReplayWithoutAction(
        logdir / "train_episodes",
        load_directory=load_logdir / "train_episodes",
        **config.replay
    )
    # 创建一个计数器对象common.Counter: step
    # step使用train_replay对象中存储的总步数作为初始值开始计数训练的步数
    step = common.Counter(train_replay.stats["total_steps"])
    outputs = [
        common.TerminalOutput(),  # 终端输出
        common.JSONLOutput(logdir),  # 将输出保存为json格式
        common.TensorBoardOutput(logdir),  # 将输出保存为tensorboard可读取的格式
    ]
    # 解释multiplier=config.action_repeat
    # 在强化学习中，每个训练步骤可能包含了几个时间步骤，几个时间步骤会共享（或重复）相同的
    # 动作（动作重复），因此总的训练步骤应该是总的训练步骤乘以重复次数。
    # 为什么会有动作重复？举例来说，如果智能体通过观测视频来做出决策，当视频的帧率为25时，
    # 智能体并不需要在每秒都做出25次决策，可以每5帧做出1次决策，这样动作重复就是5。
    logger = common.Logger(step, outputs, multiplier=config.action_repeat)
    metrics = collections.defaultdict(list)

    should_log = common.Every(config.log_every)
    should_save = common.Every(config.eval_every)

    def make_env(mode):
        suite, task = config.task.split("_", 1)
        if suite == "dmc":
            env = common.DMC(
                task, config.action_repeat, config.render_size, config.dmc_camera
            )
            env = common.NormalizeAction(env)
        elif suite == "atari":
            env = common.Atari(
                task, config.action_repeat, config.render_size, config.atari_grayscale
            )
            env = common.OneHotAction(env)
        elif suite == "crafter":
            assert config.action_repeat == 1
            outdir = logdir / "crafter" if mode == "train" else None
            reward = bool(["noreward", "reward"].index(task)) or mode == "eval"
            env = common.Crafter(outdir, reward)
            env = common.OneHotAction(env)
        elif suite == "metaworld":
            task = "-".join(task.split("_"))
            env = common.MetaWorld(
                task,
                config.seed,
                config.action_repeat,
                config.render_size,
                config.camera,
            )
            env = common.NormalizeAction(env)
        else:
            raise NotImplementedError(suite)
        env = common.TimeLimit(env, config.time_limit)
        return env

    print("Create envs.")
    env = make_env("train")
    act_space, obs_space = env.act_space, env.obs_space

    print("Create agent.")
    train_dataset = iter(train_replay.dataset(**config.dataset))
    report_dataset = iter(train_replay.dataset(**config.dataset))
    agnt = agent.Agent(config, obs_space, act_space, step)
    # def train(self, data, state=None)
    # CarryOverState在内部将上一次的结果保存为state，在新一次计算时train的输入
    # 为data，而CarryOverState会按照train(data, state=state)来计算，并将结
    # 果继续更新为state保存在内部。
    train_agent = common.CarryOverState(agnt.train)
    train_agent(next(train_dataset))

    if (logdir / "variables.pkl").exists():
        agnt.load(logdir / "variables.pkl")

    print("Train a video prediction model.")
    # for step in tqdm(range(step, config.steps), total=config.steps, initial=step):
    for _ in range(int(step.value), int(config.steps)):
        mets = train_agent(next(train_dataset))
        [metrics[key].append(value) for key, value in mets.items()]
        step.increment()

        if should_log(step):
            for name, values in metrics.items():
                logger.scalar(name, np.array(values, np.float64).mean())
                metrics[name].clear()
            logger.add(agnt.report(next(report_dataset)), prefix="train")
            logger.write(fps=True)

        if should_save(step):
            agnt.save(logdir / "variables.pkl")
            agnt.wm.rssm.save(logdir / "rssm_variables.pkl", verbose=False)
            agnt.wm.encoder.save(logdir / "encoder_variables.pkl", verbose=False)
            agnt.wm.heads["decoder"].save(
                logdir / "decoder_variables.pkl", verbose=False
            )

    env.close()
    agnt.save(logdir / "variables.pkl")
    agnt.wm.rssm.save(logdir / "rssm_variables.pkl", verbose=False)
    agnt.wm.encoder.save(logdir / "encoder_variables.pkl", verbose=False)
    agnt.wm.heads["decoder"].save(logdir / "decoder_variables.pkl", verbose=False)


if __name__ == "__main__":
    main()
