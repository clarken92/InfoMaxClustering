import numpy as np
from torch.utils.data import Dataset


class DatasetContainer(Dataset):
    def __init__(self, base_dataset):
        assert isinstance(base_dataset, Dataset), "'base_dataset' " \
            "must be an instance of Dataset. Found {}!".format(type(Dataset))
        self.base_dataset = base_dataset
        # self.__dict__.update(base_dataset.__dict__)

    def __getattr__(self, attr):
        if hasattr(self.base_dataset, attr):
            return getattr(self.base_dataset, attr)
        else:
            raise AttributeError(f"Cannot find attribute '{attr}' for class '{self.__class__}'!")


class DataSubset(DatasetContainer):
    def __init__(self, base_dataset, size_or_ids, seed=None):
        super(DataSubset, self).__init__(base_dataset)

        if isinstance(size_or_ids, int):
            size = size_or_ids
            assert size < len(base_dataset), "'base_dataset' only have {} samples " \
                "while 'size_or_ids'={}!".format(len(base_dataset), size)
            self.ids = np.random.RandomState(seed).choice(
                list(range(len(base_dataset))), size, replace=False)
        else:
            assert hasattr(size_or_ids, '__len__')
            self.ids = size_or_ids

    def __getitem__(self, idx):
        base_idx = self.ids[idx]
        return self.base_dataset[base_idx]

    def __len__(self):
        return len(self.ids)


class DatasetWithMultipleTransforms(DatasetContainer):
    def __init__(self, base_dataset, transforms):
        super(DatasetWithMultipleTransforms, self).__init__(base_dataset)

        assert hasattr(base_dataset, 'transform'), "base_dataset must have the attribute 'transform'!"
        assert base_dataset.transform is None, f"base_dataset.transform must be None!"

        assert isinstance(transforms, (list, tuple)), \
            "'transforms' must be a list/tuple of Transform objects!"
        for i in range(len(transforms)):
            assert callable(transforms[i]), f"transforms[{i}] is not callable!"
        self.transforms = transforms

    def __getitem__(self, idx):
        output = self.base_dataset[idx]
        x = output[0]
        xs = tuple(tf(x) for tf in self.transforms)

        return xs + output[1:]

    def __len__(self):
        return len(self.base_dataset)


# Use case: When we want to combine both train and test set
class DatasetCombined2(DatasetContainer):
    def __init__(self, base_dataset, other_dataset):
        super(DatasetCombined2, self).__init__(base_dataset)

        assert other_dataset.__class__ is base_dataset.__class__, \
            f"base_dataset.class={base_dataset.__class__}, " \
            f"other_dataset.class={other_dataset.__class__}"

        self.other_dataset = other_dataset

    def __getitem__(self, idx):
        if idx < len(self.base_dataset):
            return self.base_dataset[idx]
        else:
            return self.other_dataset[idx - len(self.base_dataset)]

    def __len__(self):
        return len(self.base_dataset) + len(self.other_dataset)