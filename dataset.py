import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class DatasetSelect:
    def __init__(self, dataset_dir, data_column, method):
        self.data_column = data_column
        self.dataset_dir = dataset_dir
        self.method = method

    def dataset(self):
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)

        raw_dataset = pd.read_csv(self.dataset_dir, encoding='utf-8', names=['tp','beta','hw', 'tw', 'bf', 'tf', 'Ap', 'Aw', 'Af', 'Z0', 'I', 'r', 'λ','uls'])
        print(raw_dataset.columns.tolist())
        input_data = raw_dataset.drop(columns=['uls'], axis=1)
        raw_dataset_2 = raw_dataset.drop(columns=['uls'], axis=1)
        output_data = pd.DataFrame(raw_dataset.drop(columns=['tp','beta','hw', 'tw', 'bf', 'tf', 'Ap', 'Aw', 'Af', 'Z0', 'I', 'r', 'λ'], axis=1))

        return input_data, output_data, raw_dataset_2

    def normalization_std(self):
        self.input_data = StandardScaler().fit_transform(self.input_data)

        return self.input_data

    def normalization_minmax(self):
        self.input_data = MinMaxScaler().fit_transform(self.input_data)
        return self.input_data

    def __call__(self):
        self.input_data, self.output_data, self.raw_dataset = self.dataset()

        if self.method == 'std':
            input_dataset = self.normalization_std()

        elif self.method == 'minmax':
            input_dataset = self.normalization_minmax()

        else:
            input_dataset = None

        input_dataset = pd.DataFrame(input_dataset, columns=self.data_column)
        x_train, x_test, y_train, y_test = train_test_split(input_dataset,
                                                            self.output_data,
                                                            test_size=0.2)

        return x_train, x_test, y_train, y_test, self.raw_dataset



