import os
from os.path import dirname, realpath, abspath
import numpy as np
import random

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    # noinspection PyDeprecation
    from sklearn.cross_validation import StratifiedShuffleSplit


class DataLoader:
    def __init__(self, train_directory, test_folds, batch_size = 32, seed = 0, num_classes = 10, num_features = 128,
                 num_iterations = 1e3):
        self._dir = abspath(train_directory)

        self._num_classes = num_classes
        self._batch_size = batch_size
        self._num_features = num_features
        self._num_iterations = num_iterations

        self._files = []
        self._test_files = []
        for folder in os.listdir(train_directory):
            if int(folder[4:]) in test_folds:
                for filename in os.listdir(train_directory + "/" + folder):
                    if not filename.endswith(".txt"):
                        continue
                    self._test_files.append(folder + "/" + filename)
            else:
                for filename in os.listdir(train_directory + "/" + folder):
                    if not filename.endswith(".txt"):
                        continue
                    self._files.append(folder + "/" + filename)

        self._idcs_train, self._idcs_valid = next(iter(
            StratifiedShuffleSplit(self._files,
                                   n_iter=10,
                                   test_size=0.1,
                                   random_state=seed)))

    def get_train_files(self):
        return self._files

    def get_train_files_size(self):
        return len(self._files)

    def get_test_files(self):
        return self._test_files

    def get_test_files_size(self):
        return len(self._test_files)

    def shuffle(self, seed=0):
        if seed == 0:
            _seed = np.random.randint(1, 1000)
        else:
            _seed = seed
        random.seed(_seed)
        random.shuffle(self._files)

    def shuffle_train(self):
        np.random.shuffle(self._idcs_train)

    def gen_batch(self):
        return np.zeros((self.get_train_files_size(), self._num_features, self._num_features)), \
                np.zeros((self.get_train_files_size(), self._num_classes))

    def gen_train(self):
        batch = self.gen_batch()
        iteration = 0
        i = 0
        while True:
            self.shuffle_train()
            for idx in self._idcs_train:
                batch[i][0] = np.genfromtxt(self._dir + '/' + self._files[idx])
                batch[i][1][int(self._files[idx].split('-')[1])] = 1
                i += 1
                if i >= self._batch_size:
                    yield batch
                    batch = self.gen_batch()
                    i = 0
                    iteration += 1
                    if iteration >= self._num_iterations:
                        break

    def gen_valid(self):
        batch = self.gen_batch()
        i = 0
        for idx in self._idcs_valid:
            batch[i][0] = np.genfromtxt(self._dir + '/' + self._files[idx])
            batch[i][1][int(self._files[idx].split('-')[1])] = 1
            if i >= self._batch_size:
                yield batch, i
                batch = self.gen_batch()
                i = 0
        if i != 0:
            yield batch, i

    def get_train_data(self, start_index, batch_size):
        files_to_return = []
        if start_index + batch_size <= len(self._files):
            files_to_return.append(self._files[start_index:start_index + batch_size][0])
        else:
            files_to_return.append(self._files[start_index:][0])
            files_to_return.append(self._files[:batch_size - start_index + len(self._files)][0])

        data_to_return = np.zeros((batch_size, self._num_features, self._num_features))
        label_to_return = np.zeros((batch_size, self._num_classes))
        for idx, file in enumerate(files_to_return):
            data = np.genfromtxt(self._dir + '/' + file, delimiter=',')
            data_to_return[idx] = data
            label_to_return[idx][int(file.split('-')[1])] = 1

        return data_to_return, label_to_return

    def get_test_data(self):
        data_to_return = np.zeros((self.get_test_files_size(), self._num_features, self._num_features))
        label_to_return = np.zeros((self.get_test_files_size(), self._num_classes))
        for idx, file in enumerate(self._test_files):
            data = np.genfromtxt(self._dir + "/" + file, delimiter=',')
            data_to_return[idx] = data
            label_to_return[idx][int(file.split('-')[1])] = 1


class BatchLoader:
    def __init__(self, path, test_folds, num_classes=10, batch_size=4, seed=0):
        self.dataloader = DataLoader(path, test_folds)
        self._batch_size = batch_size
        self._num_classes = num_classes
        self._epoch = 0
        self._index_in_epoch = 0

        self.dataloader.shuffle(seed)

    def get_cur_epoch(self):
        return self._epoch

    def get_batch(self):
        index = self._index_in_epoch
        self._index_in_epoch = (self._index_in_epoch + self._batch_size) % self.dataloader.get_train_files_size()
        if index >= self._index_in_epoch:
            self._epoch += 1
        return self.dataloader.get_train_data(index, self._batch_size)


# path = dirname(dirname(realpath(__file__))) + "/Spectrograms/fold1"
# cDataLoader = DataLoader(path)
# print(cDataLoader.get_files()[:5])

cBatchLoader = BatchLoader("../Spectrograms", [4], batch_size=1)
loader = DataLoader("../Spectrograms", [4], batch_size=1, num_iterations=5)

print("Test files: " + str(loader.get_test_files_size()))
print("Training files: " + str(loader.get_train_files_size()))

for i, batch_train in enumerate(loader.gen_train()):
    print("Batch Train: \n", batch_train)

