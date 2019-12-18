import os
from builtins import FileExistsError

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split

from Ancillary_Functions import get_label_field_name, _normalize_column

headers = ['TIME', 'CPU_FREQ_0', 'CPU_FREQ_1', 'CPU_FREQ_2', 'CPU_FREQ_3', 'CPU_FREQ_4', 'CPU_FREQ_5',
           'CPU_FREQ_6', 'CPU_FREQ_7', 'USED_MEM']


class DataPreprocessor:
    def __init__(self, system_vitals, attack_timings):
        self.system_vitals = pd.read_csv(system_vitals, delimiter=',')
        self.attack_timings = pd.read_csv(attack_timings, delimiter=",", names=['TYPE', 'FROM', 'TO'])

    def label_and_split_data(self):
        dataframe_to_return = pd.DataFrame(columns=['TIME', 'ATTACK'])
        i, j = np.where((self.system_vitals.TIME.values[:, None] >= self.attack_timings.FROM.values) & (
                self.system_vitals.TIME.values[:, None] <= self.attack_timings.TO.values))
        df = pd.DataFrame(np.column_stack([self.system_vitals.values[i], self.attack_timings.values[j]]),
                          columns=self.system_vitals.columns.append(self.attack_timings.columns)) \
            .append(self.system_vitals[~np.in1d(np.arange(len(self.system_vitals)), np.unique(i))], ignore_index=True)
        df['ATTACK'] = pd.notna(df.TYPE)
        dataframe_to_return = dataframe_to_return.append(df[['TIME', 'ATTACK']])
        system_vitals = self.system_vitals.merge(dataframe_to_return, left_on="TIME", right_on="TIME")

        train, test = train_test_split(system_vitals, test_size=0.2, shuffle=False)
        train, validation = train_test_split(train.sort_values(by=['TIME']), test_size=0.2, shuffle=False)
        return train, test, validation

    # def EDA(self, system_vitals):

    def dimension_reduction(self, data_frame, windows_size):
        pd.set_option('display.max_columns', None)
        list_df = [data_frame[i:i + windows_size] for i in range(0, data_frame.shape[0], windows_size)]
        flattened_dataframe = None
        for batch_df in list_df:
            if len(batch_df) < windows_size:
                continue
            mean = batch_df['ATTACK'].mean()
            result_value = False
            if mean > 0.5:
                result_value = True
            min_time = batch_df['TIME'].min()
            max_time = batch_df['TIME'].max()
            without_result = batch_df.drop(
                ['ATTACK', 'TIME', 'CPU_FREQ_0', 'CPU_FREQ_1', 'CPU_FREQ_2',
                 'CPU_FREQ_3', 'CPU_FREQ_4', 'CPU_FREQ_5', 'CPU_FREQ_6', 'CPU_FREQ_7'], axis=1
            )
            without_result['MEAN_CPU_FREQUENCY'] = batch_df[['CPU_FREQ_0', 'CPU_FREQ_1', 'CPU_FREQ_2',
                                                             'CPU_FREQ_3', 'CPU_FREQ_4', 'CPU_FREQ_5',
                                                             'CPU_FREQ_6', 'CPU_FREQ_7']].mean(axis=1)
            without_result['MIN_CPU_FREQUENCY'] = batch_df[['CPU_FREQ_0', 'CPU_FREQ_1', 'CPU_FREQ_2',
                                                            'CPU_FREQ_3', 'CPU_FREQ_4', 'CPU_FREQ_5',
                                                            'CPU_FREQ_6', 'CPU_FREQ_7']].min(axis=1)
            without_result['MAX_CPU_FREQUENCY'] = batch_df[['CPU_FREQ_0', 'CPU_FREQ_1', 'CPU_FREQ_2',
                                                            'CPU_FREQ_3', 'CPU_FREQ_4', 'CPU_FREQ_5',
                                                            'CPU_FREQ_6', 'CPU_FREQ_7']].max(axis=1)
            flattened_values = without_result.values.flatten()
            columns = [str(i) for i in range(0, len(flattened_values))]
            flattened_values = np.append(flattened_values, [min_time, max_time, result_value])
            columns.append('MIN_TIME')
            columns.append('MAX_TIME')
            columns.append(get_label_field_name())
            if flattened_dataframe is None:
                flattened_dataframe = pd.DataFrame(columns=columns)
            flattened_dataframe = flattened_dataframe.append(
                pd.DataFrame([list(pd.Series(flattened_values))], columns=columns))

        for i in list(flattened_dataframe):
            if i not in ['MIN_TIME', 'MAX_TIME', 'IS_ATTACK']:
                flattened_dataframe[i] = _normalize_column(flattened_dataframe[i])

        y_val = flattened_dataframe[get_label_field_name()]
        from_to_val = flattened_dataframe[['MIN_TIME', 'MAX_TIME']]
        x_val = flattened_dataframe.drop(get_label_field_name(), axis=1).drop('MIN_TIME', axis=1).drop('MAX_TIME', axis=1)
        return x_val.values, y_val.values.astype(int), from_to_val.values

    def load_dataset(self, windows_size=11):
        train, test, validation = self.label_and_split_data()
        train_PCA = self.dimension_reduction(train, windows_size)
        test_PCA = self.dimension_reduction(test, windows_size)
        validation_PCA = self.dimension_reduction(validation, windows_size)
        folder_name="window_size_" + str(windows_size)
        self.save_array_to_file("_train", train_PCA[0], train_PCA[1], train_PCA[2], folder_name)
        self.save_array_to_file("_test", test_PCA[0], test_PCA[1], test_PCA[2], folder_name)
        self.save_array_to_file("_validation", validation_PCA[0], validation_PCA[1], validation_PCA[2], folder_name)

    def save_array_to_file(self, name, x, y, times, folder):
        try:
            os.makedirs(folder)
        except FileExistsError:
            pass
        x_dataframe = DataFrame(data=x)
        y_dataframe = DataFrame(data=y)
        times_dataframe = DataFrame(data=times)
        x_dataframe.to_csv(folder + "/x" + name + ".csv", index=False)
        y_dataframe.to_csv(folder + "/y" + name + ".csv", index=False)
        times_dataframe.to_csv(folder + "/z" + name + ".csv", index=False)


t = DataPreprocessor("./service_data.txt", "./attack_timings.txt")
t.load_dataset(windows_size=3)