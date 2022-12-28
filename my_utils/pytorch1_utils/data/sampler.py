import numpy as np
from torch.utils.data.sampler import Sampler


# Like ContinuousIndexSampler in python_utils
# But perform much much faster
# Used frequently
class ContinuousBatchSampler(Sampler):
    def __init__(self, data_size, num_repeats, batch_size,
                 shuffle=False, seed=None):

        assert isinstance(data_size, int) and data_size > 0, f"data_size={data_size}"

        self.data_size = data_size
        self.num_repeats = num_repeats
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rs = np.random.RandomState(seed)

        self._ids = []
        self._start_point = 0

    def _refill(self):
        if self.shuffle:
            for _ in range(self.num_repeats):
                self._ids.extend(self.rs.permutation(self.data_size))
        else:
            for _ in range(self.num_repeats):
                self._ids.extend(range(self.data_size))

    def __iter__(self):
        while True:
            end_point = self._start_point + self.batch_size

            if end_point > len(self._ids):
                # print(f"_ids (before): {self._ids}")
                # print(f"len(_ids) (before): {len(self._ids)}")
                # print(f"_start_point (before): {self._start_point}")

                self._ids = self._ids[self._start_point:]
                self._start_point = 0
                self._refill()

                # print(f"_ids (after): {self._ids}")
                # print(f"len(_ids) (after): {len(self._ids)}")
                # print(f"_start_point (after): {self._start_point}")

            else:
                yield self._ids[self._start_point: end_point]
                self._start_point = end_point


# Simply take random samples with replacement
# Used frequently
class ContinuousRandomBatchSampler(Sampler):
    def __init__(self, data_size, batch_size, buffer_size=10000, seed=None):

        assert isinstance(data_size, int) and data_size > 0, f"data_size={data_size}"

        self.data_size = data_size
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.rs = np.random.RandomState(seed)

        self._ids = []
        self._start_point = 0

    def _refill(self):
        self._ids.extend(self.rs.randint(low=0, high=self.data_size, size=self.buffer_size))

    def __iter__(self):
        while True:
            end_point = self._start_point + self.batch_size

            if end_point > len(self._ids):
                self._ids = self._ids[self._start_point:]
                self._start_point = 0
                self._refill()
            else:
                yield self._ids[self._start_point: end_point]
                self._start_point = end_point


# Take random samples with replacement
# Support sampling probability over samples
# Support changing ids
class ContinuousRandomBatchSampler_v2(Sampler):
    def __init__(self, data_size_or_ids, batch_size, sampling_probs=None,
                 buffer_size=10000, seed=None):
        if isinstance(data_size_or_ids, int):
            assert data_size_or_ids > 0, f"data_size_or_ids={data_size_or_ids}"
            self.ids = np.arange(data_size_or_ids)
        else:
            self.ids = np.asarray(data_size_or_ids, dtype=np.int32)
        self.data_size = len(self.ids)

        if sampling_probs is not None:
            sampling_probs = np.asarray(sampling_probs, dtype=np.float32)
            assert len(sampling_probs) == self.data_size, \
                f"len(sampling_probs)={len(sampling_probs)} while " \
                f"self.data_size={self.data_size}!"
        self.sampling_probs = sampling_probs

        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.rs = np.random.RandomState(seed)

        self._ids_pool = []
        self._start_point = 0

    def _refill(self):
        self._ids_pool.extend(self.rs.choice(self.ids, size=self.buffer_size,
                                             replace=True, p=self.sampling_probs))

    def __iter__(self):
        while True:
            end_point = self._start_point + self.batch_size

            if end_point > len(self._ids_pool):
                self._ids_pool = self._ids_pool[self._start_point:]
                self._start_point = 0
                self._refill()
            else:
                yield self._ids_pool[self._start_point: end_point]
                self._start_point = end_point

    def set_ids_and_sampling_probs(self, ids, sampling_probs=None):
        self.ids = np.asarray(ids, dtype=np.int32)
        self.data_size = len(self.ids)

        if sampling_probs is not None:
            sampling_probs = np.asarray(sampling_probs, dtype=np.float32)
            assert len(sampling_probs) == self.data_size, \
                f"len(sampling_probs)={len(sampling_probs)} while " \
                f"self.data_size={self.data_size}!"
        self.sampling_probs = sampling_probs

        self._ids_pool = []
        self._start_point = 0

    def set_sampling_probs(self, sampling_probs):
        sampling_probs = np.asarray(sampling_probs, dtype=np.float32)
        assert len(sampling_probs) == self.data_size, \
            f"len(sampling_probs)={len(sampling_probs)} while " \
            f"self.data_size={self.data_size}!"

        self.sampling_probs = sampling_probs

        self._ids_pool = []
        self._start_point = 0
