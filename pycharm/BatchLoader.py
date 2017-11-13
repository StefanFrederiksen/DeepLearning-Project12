import os
from os.path import abspath
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


def get_label(filename):
    return int(filename.split('-')[1])


class BatchLoader:
    def __init__(self, train_directory, test_folds, batch_size=32, seed=0, num_classes=10, num_features=128,
                 num_iterations=10):
        self._dir = abspath(train_directory)

        self._num_classes = num_classes
        self._batch_size = batch_size
        self._num_features = num_features
        self._num_iterations = num_iterations

        self._files = []
        self._files_labels = []
        self._test_files = []
        self._test_files_labels = []
        for folder in os.listdir(train_directory):
            if int(folder[4:]) in test_folds:
                for filename in os.listdir(train_directory + "/" + folder):
                    if not filename.endswith(".txt"):
                        continue
                    self._test_files.append(folder + "/" + filename)
                    self._test_files_labels.append(get_label(filename))
            else:
                for filename in os.listdir(train_directory + "/" + folder):
                    if not filename.endswith(".txt"):
                        continue
                    self._files.append(folder + "/" + filename)
                    self._files_labels.append(get_label(filename))

        if seed == 0:
            self._seed = np.random.randint(1, 1000)
        else:
            self._seed = seed

        self._sss = StratifiedShuffleSplit(test_size=0.1, random_state=self._seed)
        self._idcs_train, self._idcs_valid = next(iter(self._sss.split(self._files, self._files_labels)))

    def load_data(self, file):
        return np.genfromtxt(self._dir + '/' + file, delimiter=',')

    def shuffle_train(self):
        np.random.shuffle(self._idcs_train)

    def gen_batch(self):
        batch_holder = dict()
        batch_holder["data"] = np.zeros((self._batch_size, self._num_features, self._num_features), dtype='float32')
        batch_holder["labels"] = np.zeros((self._batch_size, self._num_classes), dtype='float32')
        return batch_holder

    def gen_train(self):
        batch = self.gen_batch()
        iteration = 0
        i = 0
        while True:
            self.shuffle_train()
            for idx in self._idcs_train:
                batch["data"][i] = self.load_data(self._files[idx])
                batch["labels"][i][get_label(self._files[idx])] = 1
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
            batch["data"][i] = self.load_data(self._files[idx])
            batch["labels"][i][get_label(self._files[idx])] = 1
            i += 1
            if i >= self._batch_size:
                yield batch, i
                batch = self.gen_batch()
                i = 0
        if i != 0:
            yield batch, i

    def gen_test(self):
        batch = self.gen_batch()
        i = 0
        for idx in range(len(self._test_files)):
            batch["data"][i] = self.load_data(self._test_files[idx])
            batch["labels"][i][get_label(self._test_files[idx])] = 1
            i += 1
            if i >= self._batch_size:
                yield batch, i
                batch = self.gen_batch()
                i = 0
        if i != 0:
            yield batch, i
