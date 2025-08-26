import numpy as np
import numbers

from mytorch.utils.data.dataset import Dataset
from mytorch.tensor import Tensor
from mytorch.rand_generator import Generator


class DataLoader:
    def __init__(self, dataset: Dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            np.random.seed(Generator.instance().generate())
            np.random.shuffle(indices)
        for index in range(0, len(indices), self.batch_size):
            samples = [
                self.dataset[i]
                for i in range(index, min(index + self.batch_size, len(indices)))
            ]
            yield self._default_collate(samples)

    @staticmethod
    def _default_collate(samples):
        element = samples[0]
        if isinstance(element, list) or isinstance(element, tuple):
            batch = []
            for i in range(len(element)):
                array = []
                for j in range(len(samples)):
                    array.append(samples[j][i])

                if isinstance(element[i], np.ndarray):
                    tensor = Tensor(
                        cpu_array=np.concatenate(
                            [np.expand_dims(i, 0) for i in array], axis=0
                        )
                    )

                elif isinstance(element[i], numbers.Number):
                    tensor = Tensor(cpu_array=np.array(array))

                else:
                    assert (
                        False
                    ), f"Invalid data type in default collate: {type(element[i])}"

                batch.append(tensor)

            return batch
        else:
            assert False, f"Invalid element type in default collate: {type(element)}"
