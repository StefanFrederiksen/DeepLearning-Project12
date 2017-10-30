import os
import numpy as np


def onehot(t, num_classes):
    out = np.zeros(t.shape[0], num_classes)
    for row, col in enumerate(t):
        out[int(row), int(col)] = 1
    return out


class DataLoader:
    def __init__(self, train_directory):
        files = []
        for filename in os.listdir(train_directory):
            if not filename.endswith(".txt"):
                continue
            files.append(filename)

        data = []
        for file in files:
            t_data = np.genfromtxt(train_directory + '/' + file, delimiter=',')
            data.append(t_data)

        self._files = files
        self._data = data

    def get_files(self):
        return self._files

    def get_data(self):
        return self._data

class BatchLoader:
    def __init__(self, data, batch_size=4, num_classes=10,
                 num_iterations=10, num_features=5, seed=67, val_size=0.2):
        self.data = data
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.num_iterations = num_iterations
        self.num_features = num_features
        self.seed = seed
        self.val_size = val_size




path = os.path.dirname(os.path.realpath(__file__)) + "/files"
t = DataLoader(path)


print(t.get_files())

asd = t.get_data()


print(np.shape(asd))

