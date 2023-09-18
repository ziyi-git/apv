import collections
import datetime
import io
import pathlib
import uuid

import numpy as np
import tensorflow as tf


class Replay:
    def __init__(
        self,
        directory,
        load_directory=None,
        capacity=0,
        ongoing=False,
        minlen=1,
        maxlen=0,
        prioritize_ends=False,
    ):
        ################################################################################################################
        # expanduser()处理包含“~”(主目录)的路径，替换为完整路径
        # self._capacity: Replay对象的dataset最多包含2000000个step，2000000个step消耗完后如何向dataset添加新数据？
        # self._ongoing: _False？
        # self._prioritize_ends: True？
        ################################################################################################################
        self._directory = pathlib.Path(directory).expanduser()  # 
        self._directory.mkdir(parents=True, exist_ok=True)
        self._capacity = capacity
        self._ongoing = ongoing
        self._minlen = minlen
        self._maxlen = maxlen
        self._prioritize_ends = prioritize_ends
        self._random = np.random.RandomState()

        if load_directory is None:
            load_directory = self._directory
        else:
            ############################################################################################################
            # load_directory是预训练数据的加载目录############################################################################################################
            load_directory = pathlib.Path(load_directory).expanduser()

        # filename -> key -> value_sequence
        ################################################################################################################
        # 一次性加载所有训练数据到self._complete_eps中，eps是episodes的全称
        ################################################################################################################
        self._complete_eps = load_episodes(load_directory, capacity, minlen)
        # worker -> key -> value_sequence
        ################################################################################################################
        # 这里还不清楚self._ongoing_eps的具体用途
        # 抛开具体用途解释 collections.defaultdict( lambda: collections.defaultdict(list) ):
        # 1. 先看外层的 collections.defaultdict: 假设dd = collections.defaultdict( lambda: ... ) ，当我们查询一个dd中不存在的
        #    key时，dd会调用lambda函数返回的对象作为key的value;
        # 2. 再看内层的 collections.defaultdict(list): 假设dd = collections.defaultdict(list)，当我们查询一个dd中不存在的key
        #    时，dd会返回一个空list作为key的value；
        # 
        # 举例:
        # dd = collections.defaultdict( lambda: collections.defaultdict(list) )
        # dd['a'] >>> defaultdict(<class 'list'>, {})
        # dd >>> {'a': defaultdict(<class 'list'>, {})}
        # 
        # 猜测self._ongoing_eps = collections.defaultdict( lambda: collections.defaultdict(list) )的具体用途:
        # 1. self._ongoing_eps的内部构造应当与self._complete_eps保持一致;
        # 2. 外层collections.defaultdict接受一个文件名作为key, 例如'beat_the_buzz_front_rgb_episode-0-140.npz';
        # 3. 内层collections.defaultdict接受{'image': list, 'action': list, 'reward': list, 'is_terminal': list, ...}
        # 因此，self._ongoing_eps与self._complete_eps在形式上一致:
        # {
        #  'beat_the_buzz_front_rgb_episode-0-140.npz': {'image': .., 'action': .., 'reward': .., 'is_terminal': ..},
        #  ...,
        #  ...,
        # }
        ################################################################################################################
        self._ongoing_eps = collections.defaultdict(
            lambda: collections.defaultdict(list)
        )
        ################################################################################################################
        # self._total_episodes, self._total_steps是指训练过程已经经历了多少step，因为中间结果保存在directory
        # self._loaded_episodes是指加载了多少episode，因为load_directory是原始的pretraining数据的目录
        # self._loaded_steps是指加载了多少step，因为load_directory是原始的pretraining数据的目录
        # 
        # NOTE:
        # self._loaded_steps = sum(eplen(x) for x in self._complete_eps.values())
        # eplen(x) for x in self._complete_eps.values()这一句作为参数，实现了循环累加.
        ################################################################################################################
        self._total_episodes, self._total_steps = count_episodes(directory)
        self._loaded_episodes = len(self._complete_eps)
        self._loaded_steps = sum(eplen(x) for x in self._complete_eps.values())

    @property
    def stats(self):
        return {
            "total_steps": self._total_steps,
            "total_episodes": self._total_episodes,
            "loaded_steps": self._loaded_steps,
            "loaded_episodes": self._loaded_episodes,
        }

    def add_step(self, transition, worker=0):
        episode = self._ongoing_eps[worker]
        for key, value in transition.items():
            episode[key].append(value)
        if transition["is_last"]:
            self.add_episode(episode)
            episode.clear()

    def add_episode(self, episode):
        length = eplen(episode)
        if length < self._minlen:
            print(f"Skipping short episode of length {length}.")
            return
        self._total_steps += length
        self._loaded_steps += length
        self._total_episodes += 1
        self._loaded_episodes += 1
        episode = {key: convert(value) for key, value in episode.items()}
        filename = save_episode(self._directory, episode)
        self._complete_eps[str(filename)] = episode
        self._enforce_limit()

    def dataset(self, batch, length):
        example = next(iter(self._generate_chunks(length)))
        dataset = tf.data.Dataset.from_generator(
            lambda: self._generate_chunks(length),
            {k: v.dtype for k, v in example.items()},
            {k: v.shape for k, v in example.items()},
        )
        # <BatchDataset shapes: 
        # {image: (16, 50, 64, 64, 3), is_terminal: (16, 50), is_first: (16, 50), action: (16, 50, 1)}, 
        #  types: {image: tf.uint8, is_terminal: tf.bool, is_first: tf.bool, action: tf.float32}>
        dataset = dataset.batch(batch, drop_remainder=True)
        dataset = dataset.prefetch(5)  # 确保始终有5个batch已经加载到内存中供模型调用.
        return dataset

    def _generate_chunks(self, length):
        sequence = self._sample_sequence()
        ################################################################################################################
        # 持续从上述self._sample_sequence()采样的sequence中按顺序截取长度为length的chunk:
        # sequence剩余元素数量不足length: 内部while循环导致sequence中所有元素被取完并触发len(sequence["reward"]) < 1，此时重新执行# 采样 sequence = self._sample_sequence() 得到新的sequence.  
        ################################################################################################################
        while True:
            chunk = collections.defaultdict(list)
            added = 0
            while added < length:
                needed = length - added
                adding = {k: v[:needed] for k, v in sequence.items()}
                sequence = {k: v[needed:] for k, v in sequence.items()}
                for key, value in adding.items():
                    chunk[key].append(value)
                added += len(adding["reward"])
                if len(sequence["reward"]) < 1:
                    sequence = self._sample_sequence()
            chunk = {k: np.concatenate(v) for k, v in chunk.items()}
            yield chunk

    def _sample_sequence(self):
        """
        随机选择一个episode, 并从episode中随机获取一个length大小处于[selg._minlen, self._maxlen]范围内的sequence.
        """
        episodes = list(self._complete_eps.values())
        if self._ongoing:
            episodes += [
                x for x in self._ongoing_eps.values() if eplen(x) >= self._minlen
            ]
        episode = self._random.choice(episodes)
        total = len(episode["reward"])
        length = total
        ################################################################################################################
        # 从episode中选择一个起点，向后截取长度为length的序列作为sequence:
        # 1. sequence的length必须是[self._minlen, self._maxlen]内的随机数.
        # 2. sequence的upper(起点上限)为total - length + 1.
        # NOTE: 按照python数组的下标法, upper应该是total - 1 - length + 1 = total - length. 对此第4条可解释.
        # 3. 为了更靠近末尾(self._prioritize_ends为True), uppper可以后移self._minlen个位点.
        # 4. sequence的index(真正的采样起点)从[0, upper)中随机获取, 右开区间可解释NOTE。此外由于第3条，需要用min(..., total - 
        #    length)加以限制避免越界.
        # NOTE: 为了将所有的sequence的length限定为等长25, self._minlen和self._maxlen都设定为25.
        ################################################################################################################
        if self._maxlen:
            length = min(length, self._maxlen)
        # Randomize length to avoid all chunks ending at the same time in case the
        # episodes are all of the same length.
        length -= np.random.randint(self._minlen)
        length = max(self._minlen, length)
        upper = total - length + 1
        if self._prioritize_ends:
            upper += self._minlen
        index = min(self._random.randint(upper), total - length)
        ################################################################################################################
        # 依次episode中的'image', 'action', 'reward', 'is_terminal', 'is_first'放入sequence
        # NOTE: 注意convert方法对值类型做了转换
        ################################################################################################################
        sequence = {
            k: convert(v[index : index + length])
            for k, v in episode.items()
            if not k.startswith("log_")
        }
        sequence["is_first"] = np.zeros(len(sequence["reward"]), np.bool)
        sequence["is_first"][0] = True
        if self._maxlen:
            assert self._minlen <= len(sequence["reward"]) <= self._maxlen
        return sequence

    def _enforce_limit(self):
        if not self._capacity:
            return
        while self._loaded_episodes > 1 and self._loaded_steps > self._capacity:
            # Relying on Python preserving the insertion order of dicts.
            oldest, episode = next(iter(self._complete_eps.items()))
            self._loaded_steps -= eplen(episode)
            self._loaded_episodes -= 1
            del self._complete_eps[oldest]


class ReplayWithoutAction(Replay):
    def __init__(
        self,
        directory,
        load_directory=None,
        capacity=0,
        ongoing=False,
        minlen=1,
        maxlen=0,
        prioritize_ends=False,
    ):
        super().__init__(
            directory=directory,
            load_directory=load_directory,
            capacity=capacity,
            ongoing=ongoing,
            minlen=minlen,
            maxlen=maxlen,
            prioritize_ends=prioritize_ends,
        )

    def _generate_chunks(self, length):
        """
        注释参考Replay类中的_generate_chunks方法.
        NOTE: 与Replay类中的_generate_chunks方法不同的是，ReplayWithoutAction
        去掉了sequence中的"reward"，保留"is_first", "is_last", "is_terminal",
        "image".
        """
        sequence = self._sample_sequence()
        sequence = {
            k: v
            for k, v in sequence.items()
            if k in ["is_first", "is_last", "is_terminal", "image"]
        }
        sequence["action"] = np.zeros((sequence["image"].shape[0], 1), dtype=np.float32)

        while True:
            chunk = collections.defaultdict(list)
            added = 0
            while added < length:
                needed = length - added
                adding = {k: v[:needed] for k, v in sequence.items()}
                sequence = {k: v[needed:] for k, v in sequence.items()}
                for key, value in adding.items():
                    chunk[key].append(value)
                added += len(adding["image"])
                if len(sequence["image"]) < 1:
                    sequence = self._sample_sequence()
                    sequence = {
                        k: v
                        for k, v in sequence.items()
                        if k
                        in [
                            "is_first",
                            "is_last",
                            "is_terminal",
                            "image",
                        ]
                    }
                    sequence["action"] = np.zeros(
                        (sequence["image"].shape[0], 1), dtype=np.float32
                    )
            ###########################################################################################################
            # 为什么使用np.concatenate(v)?
            # chunk = collections.defaultdict(list)会使chunk中的所有v都是一个list，例如
            # chunk['is_terminal'] >>> [array(False, False, ..., False)]
            # np.concatenate(chunk['is_terminal']) >>> array(False, False, ..., False)
            # np.concatenate(...)将list转为了数组.
            ###########################################################################################################
            chunk = {k: np.concatenate(v) for k, v in chunk.items()}
            yield chunk


def count_episodes(directory):
    filenames = list(directory.glob("*.npz"))
    num_episodes = len(filenames)
    num_steps = sum(int(str(n).split("-")[-1][:-4]) - 1 for n in filenames)
    return num_episodes, num_steps


def save_episode(directory, episode):
    timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    identifier = str(uuid.uuid4().hex)
    length = eplen(episode)
    filename = directory / f"{timestamp}-{identifier}-{length}.npz"
    with io.BytesIO() as f1:
        np.savez_compressed(f1, **episode)
        f1.seek(0)
        with filename.open("wb") as f2:
            f2.write(f1.read())
    return filename

########################################################################################################################
# 数据集说明
# rlbench
# - 99个tasks: 
#       beat_the_buzz
#       block_pyramid
#       change_channel
#       ...
# - 5 camera views per task: 
#       beat_the_buzz_front
#       beat_the_buzz_left
#       beat_the_buzz_overhead
#       beat_the_buzz_wrist
#       beat_the_buzz_right.
# - 10 demonstrations (episode) per camera view: 
#       beat_the_buzz_front_rgb_episode-0-140.npz
#       beat_the_buzz_front_rgb_episode-1-162.npz
#       ...
# NOTE: 似乎丢掉了一些episode，按理应该有99 * 5 * 10 = 4950个episodes，但实际只有3987个.
########################################################################################################################
def load_episodes(directory, capacity=None, minlen=1):
    # The returned directory from filenames to episodes is guaranteed to be in
    # temporally sorted order.
    filenames = sorted(directory.glob("*.npz"))
    if capacity:
        num_steps = 0
        num_episodes = 0
        for filename in reversed(filenames):  # reversed指从最后一个元素开始向前取数据
            length = int(str(filename).split("-")[-1][:-4])  # beat_the_buzz_front_rgb_episode-0-140.npz -> length=140
            num_steps += length
            num_episodes += 1
            if num_steps >= capacity:
                break
        filenames = filenames[-num_episodes:]  # 假如取了10个episodes而第10个episode导致num_steps>=capacity，则取后9个
    episodes = {}
    for filename in filenames:
        try:
            with filename.open("rb") as f:
                ########################################################################################################
                # 以"beat_the_buzz_front_rgb_episode-0-140.npz"所存储的episode为例:
                # 'image':       (140, 64, 64, 3)
                # 'action':      (140, 4)         在pretraining数据集中，加载时已全部设定为[.0, .0, .0, .0]
                # 'reward':      (140,)           在pretraining数据集中，加载时已全部设定为.0
                # 'is_terminal': (140,)           仅episode最后一个元素为True
                # 'is_first':    (140,)           仅episode第一个元素为True ########################################################################################################
                episode = np.load(f)
                episode = {k: episode[k] for k in episode.keys()}
        except Exception as e:
            print(f"Could not load episode {str(filename)}: {e}")
            continue
        ################################################################################################################
        # episodes就是字典:
        # {
        #  'beat_the_buzz_front_rgb_episode-0-140.npz': {'image': .., 'action': .., 'reward': .., 'is_terminal': ..},
        #  ...,
        #  ...,
        # }
        ################################################################################################################
        episodes[str(filename)] = episode
    return episodes


def convert(value):
    value = np.array(value)
    if np.issubdtype(value.dtype, np.floating):
        return value.astype(np.float32)
    elif np.issubdtype(value.dtype, np.signedinteger):
        return value.astype(np.int32)
    elif np.issubdtype(value.dtype, np.uint8):
        return value.astype(np.uint8)
    return value


def eplen(episode):
    ####################################################################################################################
    # -1是指每个episode的第1步是起始步，不算在内？
    ####################################################################################################################
    return len(episode["image"]) - 1
