import os
from os.path import abspath
import numpy as np
import random
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
                print("Idx in gen_train: ", idx)
                print("i in gen_train: ", i)
                print("iteration in gen_train: ", iteration)
                batch["data"][i] = np.genfromtxt(self._dir + '/' + self._files[idx], delimiter=',')
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
            batch["data"][i][0] = np.genfromtxt(self._dir + '/' + self._files[idx], delimiter=',')
            batch["labels"][i][1][get_label(self._files[idx])] = 1
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


# path = dirname(dirname(realpath(__file__))) + "/Spectrograms/fold1"
# cDataLoader = DataLoader(path)
# print(cDataLoader.get_files()[:5])

loader = BatchLoader("../Spectrograms", [4], batch_size=2, num_iterations=5)

print("Test files: " + str(loader.get_test_files_size()))
print("Training files: " + str(loader.get_train_files_size()))


lel = next(loader.gen_train())
print(lel)
#
# for indexx, batch_train in enumerate(loader.gen_train()):
#     # print("Batch Train: \n", batch_train)
#     continue
