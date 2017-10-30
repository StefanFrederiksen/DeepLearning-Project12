import os
from os.path import dirname, realpath, abspath
import numpy as np
import random


class DataLoader:
    def __init__(self, train_directory, test_folds):
        self._dir = abspath(train_directory)
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

    def get_train_data(self, start_index, batch_size):
        files_to_return = []
        if start_index + batch_size <= len(self._files):
            files_to_return.append(self._files[start_index:start_index + batch_size][0])
        else:
            files_to_return.append(self._files[start_index:][0])
            files_to_return.append(self._files[:batch_size - start_index + len(self._files)][0])

        data_to_return = np.zeros((batch_size, 128, 128))
        label_to_return = np.zeros((batch_size, 10))  # Todo: dont hardcore num_classes
        for idx, file in enumerate(files_to_return):
            data = np.genfromtxt(self._dir + '/' + file, delimiter=',')
            data_to_return[idx] = data
            label_to_return[idx][int(file.split('-')[1])] = 1

        return data_to_return, label_to_return

    def get_test_data(self):
        data_to_return = np.zeros((self.get_test_files_size(), 128, 128))
        label_to_return = np.zeros((self.get_test_files_size(), 10))
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

print("Test files: " + str(cBatchLoader.dataloader.get_test_files_size()))
print("Training files: " + str(cBatchLoader.dataloader.get_train_files_size()))

for count in range(0, 1):
    print(cBatchLoader.get_cur_epoch())
    val, label = cBatchLoader.get_batch()
    print(val)
    print(label)
