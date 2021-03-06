import os
from os.path import abspath
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


def get_label(filename):
    return int(filename.split('-')[1])


class BatchLoader:
    def __init__(self, train_directory, test_folds, batch_size=32, seed=None, num_classes=10, num_features_1=128, num_features_2=64,
                 nchannels=1,num_iterations=10):
        self._dir = abspath(train_directory)

        self._num_classes = num_classes
        self._batch_size = batch_size
        self._num_features_1 = num_features_1
        self._num_features_2 = num_features_2
        self._num_channels = nchannels
        self._num_iterations = num_iterations
        self._cur_epoch = -1

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

        self._sss = StratifiedShuffleSplit(test_size=0.1, random_state=seed)
        self._idcs_train, self._idcs_valid = next(iter(self._sss.split(self._files, self._files_labels)))

    def get_cur_epoch(self):
        return self._cur_epoch

    def load_data(self, file):
        return np.reshape(np.loadtxt(self._dir + '/' + file, delimiter=','),(self._num_features_1, self._num_features_2, self._num_channels))

    def shuffle_train(self):
        np.random.shuffle(self._idcs_train)

    def gen_batch(self):
        batch_holder = dict()
        batch_holder["data"] = np.zeros((self._batch_size, self._num_features_1, self._num_features_2, self._num_channels), dtype='float32')
        batch_holder["labels"] = np.zeros((self._batch_size, self._num_classes), dtype='float32')
        return batch_holder

    def gen_train(self):
        batch = self.gen_batch()
        i = 0
        while True:
            self.shuffle_train()
            self._cur_epoch += 1
            for idx in self._idcs_train:
                batch["data"][i] = self.load_data(self._files[idx])
                batch["labels"][i][get_label(self._files[idx])] = 1
                i += 1
                if i >= self._batch_size:
                    yield batch
                    batch = self.gen_batch()
                    i = 0

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
