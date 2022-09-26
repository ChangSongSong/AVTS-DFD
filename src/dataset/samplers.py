import torch
import torch.utils.data


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices: a list of indices
        num_samples: number of samples to draw
    """

    def __init__(self,
                 dataset,
                 indices: list = None,
                 num_samples: int = None,
                 replacement: bool = True):
        # if indices is not provided, all elements in the dataset will be considered
        self.indices = list(range(
            len(dataset))) if indices is None else indices

        # if num_samples is not provided, draw `len(indices)` samples in each iteration
        self.num_samples = len(
            self.indices) if num_samples is None else num_samples

        self.replacement = replacement

        # distribution of classes in the dataset
        labels = self._get_labels(dataset)
        label_to_count = labels.value_counts()
        weights = 1.0 / label_to_count[labels]
        self.weights = torch.DoubleTensor(weights.to_list())

    def _get_labels(self, dataset):
        return dataset.video_info.iloc[:, 2]

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=self.replacement))

    def __len__(self):
        return self.num_samples


class RandomSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, data_source, num_samples=None):
        self.data_source = data_source
        self._num_samples = num_samples

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(
                                 self.num_samples))

    @property
    def num_samples(self):
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self):
        n = len(self.data_source)
        return iter(
            torch.randperm(n, dtype=torch.int64)[:self.num_samples].tolist())

    def __len__(self):
        return self.num_samples
