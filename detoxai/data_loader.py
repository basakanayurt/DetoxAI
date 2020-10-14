import random
import pandas as pd
from detoxai.preprocess import *
from sklearn import model_selection as ms
import os
import math


def get_n_samples(x,num_samples):
    x_sample_0 = x[x['label'] == 0][:math.floor(num_samples)]
    x_sample_1 = x[x['label'] == 1][:math.floor(num_samples)]
    x_ = pd.concat([x_sample_0, x_sample_1])
    x_ = x_.sample(frac=1).reset_index(drop=True)
    return x_


class LoadData:

    def __init__(self, task='all', stories_path=None):
        self.task = task
        self.data_dir = '../data/'

        self.models_dir = "../trained_models/"
        self.train_test_dir = "../data/train_test/"
        self.results_dir = "../data/predictions/"

        # training data path
        if stories_path:
            self.stories_path = stories_path
        else:
            self.stories_path = self.data_dir + 'stories.csv'
        self.selfharm_path = self.data_dir + 'selfharm.csv'
        self.poems_path = self.data_dir + 'poems.csv'
        self.hatespeech_path = self.data_dir + 'hatespeech.csv'
        self.spams_path = self.data_dir + 'spams.csv'
        self.check_directories()

    def check_directories(self):

        temp1, temp2, temp3 = '..', '..', '..'
        for i in range(1, len(self.models_dir.split('/'))):

            temp1 = '/'.join([temp1, self.models_dir.split('/')[i]])
            temp2 = '/'.join([temp2, self.train_test_dir.split('/')[i]])
            temp3 = '/'.join([temp3, self.results_dir.split('/')[i]])
            for temp in [temp1, temp2, temp3]:
                if not os.path.isdir(temp):
                    os.mkdir(temp)
                    print('directory created: ', temp)

    def create_train_test_set(self, train_percent, num_samples, save_files=True):

        random.seed(2)

        if self.task == 'spam':
            normal = pd.read_csv(self.stories_path)
            toxic = pd.read_csv(self.spams_path)

        if self.task == 'selfharm':
            normal = pd.read_csv(self.poems_path)
            toxic = pd.read_csv(self.selfharm_path)

        if self.task == 'hatespeech':
            all_data = pd.read_csv(self.hatespeech_path)
            normal = all_data[all_data['label'] == 0]
            toxic = all_data[all_data['label'] == 1]

        if self.task != 'spam':
            preprocess_rows(normal)
            preprocess_rows(toxic)

        normal_ = normal.sample(frac=1).reset_index(drop=True)
        toxic_ = toxic.sample(frac=1).reset_index(drop=True)

        if len(normal_)>len(toxic_)*2:
            normal = normal_.iloc[:int(len(toxic)*1.5),:]
            extra_test = normal_.iloc[int(len(toxic)*1.5):,:].reset_index(drop=True)
            toxic = toxic_
        elif len(toxic_)>len(normal_)*2:
            toxic = toxic_.iloc[:int(len(normal_)*1.5),:]
            extra_test = toxic_.iloc[int(len(normal)*1.5):,:].reset_index(drop=True)
            normal = normal_
        else:
            toxic = toxic_.reset_index(drop=True)
            normal = normal_.reset_index(drop=True)
            extra_test = []

        data = pd.concat([normal,toxic])
        sss = ms.StratifiedShuffleSplit(n_splits=2, test_size=len(data)-int(train_percent*len(data)), train_size=int(train_percent*len(data)))
        for train_index, test_index in sss.split(data['data'].values, data['label'].values):
            x_train = data.iloc[train_index,:]
            x_test = data.iloc[test_index, :]

        x_train.reset_index(drop=True, inplace=True)
        x_test.reset_index(drop=True, inplace=True)

        if len(extra_test)>0:
            x_test = pd.concat([x_test, extra_test])
            x_test.reset_index(drop=True, inplace=True)

        if num_samples!='all':
            print(num_samples, ' samples are going to be used in training')
            # take samples
            x_train = get_n_samples(x_train, num_samples)
            x_test = get_n_samples(x_test, num_samples)

        if save_files:
            data_dir = self.train_test_dir  + self.task
            x_train.to_csv( data_dir + '_train_set.csv', index=False)
            x_test.to_csv( data_dir + '_test_set.csv', index=False)
            print('training and test samples are saved in the folder ',  data_dir)
            print('num training samples ', len(x_train))
            print('num test samples ', len(x_test))
        else:
            print('training and test samples are not saved to disk')

        return x_train, x_test

    def load_by_name_or_path(self,file_path_or_name=None,num_samples='all'):
        print('loading from', self.train_test_dir  + self.task + '_' + file_path_or_name + '.csv')
        if file_path_or_name == 'train_set' or file_path_or_name == 'test_set':
            data = pd.read_csv(self.train_test_dir  + self.task + '_' + file_path_or_name + '.csv')
        elif file_path_or_name == 'stories':
            data = pd.read_csv(self.stories_path)
        else:
            data = pd.read_csv(file_path_or_name)
        print('total number of samples available: ', len(data))
        if num_samples == 'all':
            print('taking all the data available in ', file_path_or_name)
        else:
            print('taking only ', num_samples, ' samples')
            data = get_n_samples(data, num_samples)
        return data



