import random
import pandas as pd
from detoxai.preprocess import *
from sklearn import model_selection as ms
import os
import math
import json


class LoadData:

    def __init__(self, task='all'):

        with open('config.json', 'r') as f:
            config = json.load(f)

        self.paths = config['paths']
        self.task = task

        # self.data_dir = config['data']['data_dir']

        self.models_dir = config['paths']['trained_models_dir']
        self.train_test_dir = config['paths']['train_test_dir']
        self.results_dir = config['paths']['results_dir']

        if task == 'selfharm':
            self.task_path = config['paths']['selfharm_file_path']

        elif task == 'hatespeech':
            self.task_path = config['paths']['hatespeech_file_path']

        elif task == 'spam':
            self.task_path = config['paths']['spam_file_path']

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

    def load_by_name_or_path(self, file_path_or_name=None, num_samples='all'):
        print('loading for ',  self.task, ' from ', file_path_or_name)
        if file_path_or_name == 'train_set' or file_path_or_name == 'test_set':
            data = pd.read_csv(self.paths[self.task + '_' + file_path_or_name + '_path'])
        elif file_path_or_name == 'stories':
            data = pd.read_csv(self.paths['stories_file_path'])
            data['prediction'] = None
            data['probability'] = None
        else:
            data = pd.read_csv(file_path_or_name)
        print('total number of samples available: ', len(data))
        if num_samples == 'all':
            print('taking all the data available in ', file_path_or_name)
        else:
            print('taking only ', num_samples, ' samples')
            data = get_n_samples(data, num_samples)

        return data

    def split_data(self, train_percent, save_files=True):
        tasks = ["hatespeech", "selfharm", "spam"]
        if self.task !='all':
            tasks = [self.task]

        for task in tasks:
            data = pd.read_csv(self.paths[task+'_file_path'])

            normal = data[data['label'] == 0]
            normal.reset_index(drop=True, inplace=True)
            toxic = data[data['label'] == 1]
            toxic.reset_index(drop=True, inplace=True)

            normal_ = normal.sample(frac=1).reset_index(drop=True)
            toxic_ = toxic.sample(frac=1).reset_index(drop=True)

            if len(normal_)>len(toxic_)*2:
                normal = normal_.iloc[:int(len(toxic)*1.2),:]
                extra_test = normal_.iloc[int(len(toxic)*1.2):,:].reset_index(drop=True)
                toxic = toxic_
            elif len(toxic_)>len(normal_)*2:
                toxic = toxic_.iloc[:int(len(normal_)*1.2),:]
                extra_test = toxic_.iloc[int(len(normal)*1.2):,:].reset_index(drop=True)
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

            if save_files:
                x_train.to_csv( self.paths[task+'_train_set_path'], index=False)
                x_test.to_csv( self.paths[task+'_test_set_path'], index=False)
                print('training and test samples are saved in the folder ',  self.paths[task+'_train_set_path'])
                print('num training samples ', len(x_train))
                print('num test samples ', len(x_test))
            else:
                print('training and test samples are not saved to disk')

        return x_train, x_test


def get_n_samples(x,num_samples):
    x_sample_0 = x[x['label'] == 0][:math.floor(num_samples)]
    x_sample_1 = x[x['label'] == 1][:math.floor(num_samples)]
    x_ = pd.concat([x_sample_0, x_sample_1])
    x_ = x_.sample(frac=1).reset_index(drop=True)
    return x_

