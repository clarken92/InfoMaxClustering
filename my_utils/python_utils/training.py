from six import moves, iteritems
from os.path import join, isdir, exists
import random
import shutil
import heapq as hq
from operator import itemgetter
import numpy as np


# Use frequently
def iterate_data(data_size_or_ids, batch_size,
                 shuffle=False, seed=None, include_remaining=True):
    """
    V1.0: Stable running
    V1.1: Add seed and local RandomState
    :param data_size_or_ids:
    :param batch_size:
    :param shuffle:
    :param seed: None for complete randomisation
    :param include_remaining:
    :return:
    """
    if isinstance(data_size_or_ids, int):
        data_size = data_size_or_ids
        ids = list(range(data_size_or_ids))
    else:
        assert hasattr(data_size_or_ids, '__len__')
        ids = data_size_or_ids.tolist() if isinstance(data_size_or_ids, np.ndarray) \
            else list(data_size_or_ids)
        data_size = len(data_size_or_ids)

    rs = np.random.RandomState(seed)
    if shuffle:
        rs.shuffle(ids)
    nb_batch = len(ids) // batch_size

    for batch in moves.xrange(nb_batch):
        yield ids[batch * batch_size: (batch + 1) * batch_size]

    if include_remaining and nb_batch * batch_size < data_size:
        yield ids[nb_batch * batch_size:]


class BestResultsTracker(object):
    def __init__(self, keys_and_cmpr_types, num_best):
        """
        keys: A list of strings
        num_best: Number of best results we want to keep track
        """

        avail_cmpr_types = ("less", "greater")
        assert isinstance(keys_and_cmpr_types, (tuple, list)), \
            "'keys_and_cmpr_types' must be a list of 2-lists/tuples of the form (key, cmpr_type)!"
        for n in moves.xrange(len(keys_and_cmpr_types)):
            assert isinstance(keys_and_cmpr_types[n], (tuple, list)) and len(keys_and_cmpr_types[n]) == 2, \
                "The element {} of 'keys_and_cmpr_types' must be a list/tuple of length 2!".format(n)
            assert isinstance(keys_and_cmpr_types[n][0], (str, bytes)), \
                "The element {} of 'keys_and_cmpr_types' must have 'key' to be str or bytes!".format(n)
            assert keys_and_cmpr_types[n][1] in avail_cmpr_types, \
                "The element {} of 'keys_and_cmpr_types' must have 'type' in {}!".format(n, avail_cmpr_types)

        self.keys_and_cmpr_type = keys_and_cmpr_types
        self.num_best = num_best

        # A dict that store best results
        # Key is the key we want to compare
        # Value is a 3-tuple:
        # The first element is the 'compr_type'
        # The second element is a list of (cmpr_val, true_val, step)

        self._best_results = dict()
        for key, cmpr_type in keys_and_cmpr_types:
            inits = [(-1e10, -1e10, -1) for _ in range(self.num_best)]
            hq.heapify(inits)

            cmpr_type = 1.0 if cmpr_type == "greater" else -1.0
            self._best_results[key] = (cmpr_type, inits)

    def check_and_update(self, results, step, assert_keys=False):
        assert isinstance(results, dict), \
            f"'results' must be a dict. Found {type(results)}!"

        # True: The key exist and its current value is better than stored values
        # False: The key exist and its current value is not as good as stored values
        # None: The key does not exist
        is_better = dict()

        for key, val in iteritems(results):
            stored_results = self._best_results.get(key)

            if stored_results is None:
                if assert_keys:
                    raise ValueError(f"The values of '{key}' are not tracked!")
                else:
                    is_better[key] = None
            else:
                cmpr_type, item_heap = stored_results
                new_item = (cmpr_type * val, val, step)

                old_item = hq.heappushpop(item_heap, new_item)

                if old_item[-1] == step:
                    is_better[key] = False
                else:
                    is_better[key] = True

        return is_better

    def check_and_update_key(self, key, val, step):
        stored_results = self._best_results.get(key)

        if stored_results is None:
            raise ValueError(f"The values of '{key}' are not tracked!")
        else:
            cmpr_type, item_heap = stored_results
            new_item = (cmpr_type * val, val, step)

            old_item = hq.heappushpop(item_heap, new_item)

            if old_item[-1] == step:
                is_better = False
            else:
                is_better = True

        return is_better

    def get_best_results(self, sort_results=False):
        results = dict()

        for key, (cmpr_type, item_heap) in iteritems(self._best_results):
            if sort_results:
                item_heap = sorted(item_heap, key=itemgetter(0), reverse=True)
            results[key] = [item[1:] for item in item_heap]

        return results

    def set_best_results(self, results):
        self._validate_results(results)
        self._best_results = results

    def _validate_results(self, results):
        assert isinstance(results, dict), f"'results' must be a dict. Found {type(results)}!"
        for key, cmpr_type in self.keys_and_cmpr_type:
            val = results.get(key)

            assert val is not None, "'results' do not contain key {}!".format(key)
            assert isinstance(val, tuple), \
                f"'results[{key}]' is not a tuple. " \
                f"Found type(results[{key}])={type(val)}!"
            assert len(val) == 2, f"len(results[{key}])={val}"

            if cmpr_type == "greater":
                assert val[0] == 1.0, f"'cmpr_type' for key '{key}' is " \
                    f"'greater' but val[0] in 'results[{key}]' is {val[0]}!"
            else:
                assert val[0] == -1.0, f"'cmpr_type' for key '{key}' is " \
                    f"'less' but val[0] in 'results[{key}]' is {val[0]}!"


class SaveDirTracker(object):
    def __init__(self, max_save, dir_path_prefix):
        self.max_save = max_save
        self._saved_steps = []
        self._dir_path_prefix = dir_path_prefix

    def get_save_dir(self, step):
        return self._dir_path_prefix + "%d" % step

    def update_and_delete_old_save(self, step):
        # new_dir = self._dir_path_prefix + "%d" % step
        # assert isdir(new_dir), "Directory [{}] does not exist!"
        self._saved_steps.append(step)
        # print(f"step: {step}")
        # print(f"_saved_steps: {self._saved_steps}")

        if len(self._saved_steps) > self.max_save:
            old_step = self._saved_steps.pop(0)

            old_dir = self.get_save_dir(old_step)
            # assert isdir(old_dir), "Old save directory [{}] does not exist!".format(old_dir)
            if exists(old_dir):
                shutil.rmtree(old_dir)

    def get_saved_steps(self):
        return self._saved_steps

    def set_saved_steps(self, steps):
        self._validate_steps(steps)
        self._saved_steps = steps

    def _validate_steps(self, steps):
        assert isinstance(steps, list), "'steps' must be a list. Found {}!".format(type(steps))
        assert len(steps) <= self.max_save, \
            "'len(steps)' must be <= {}. Found {}!".format(self.max_save, len(steps))

        for i, step in enumerate(steps):
            assert isinstance(step, int), "'steps[{}]' must be an int. Found {}!".format(i, type(step))