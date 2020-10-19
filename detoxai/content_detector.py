from transformers import (
    DistilBertTokenizer,
    TFDistilBertForSequenceClassification,
    pipeline
)
from sklearn.metrics import accuracy_score, confusion_matrix
from detoxai.utils import *
import tensorflow as tf
import numpy as np
from detoxai.data_loader import *
from detoxai.preprocess import *


class Info():
    def __init__(self):

        with open('config.json', 'r') as f:
            config = json.load(f)
        self.paths = config['paths']
        self.tasks = config['tasks']
        self.training_step_size = config['training_step_size']
        self.prediction_step_size = config['prediction_step_size']


########################################## HATESPEECH ##################################################################
class Hatespeech:
    def __init__(self):
        config = Info()
        task_model = config.tasks['hatespeech']
        self.step_size = config.training_step_size
        self.prediction_step_size = config.prediction_step_size
        self.learning_rate = task_model['learning_rate']
        self.max_len = task_model['max_len']
        self.epochs = task_model['epochs']
        self.batch_size = task_model['batch_size']
        self.model = task_model['model']
        self.model_path = config.paths['trained_models_dir'] + 'hatespeech'
        self.predictions_path = config.paths['results_dir']

    def train(self, train_data=None, num_samples='all', return_model=True):
        if not train_data:
            train_data = LoadData('hatespeech').load_by_name_or_path('train_set', num_samples=num_samples)
        train_data = check_token_len(train_data, self.max_len)

        model = trainX(model_name=self.model,
                       model_path=self.model_path,
                       train_data=train_data,
                       max_len=self.max_len,
                       step_size=self.step_size,
                       epochs=self.epochs, batch_size=self.batch_size, learning_rate=self.learning_rate)
        if return_model:
            return model

    def predict(self, data=None, num_samples='all',save_file=True ):
        data_name = data
        if type(data) == str and data in ['train_set', 'test_set','stories']:
            data = LoadData('hatespeech').load_by_name_or_path(data, num_samples=num_samples)
        data = check_token_len(data, self.max_len)

        data = predictX(model_name=self.model, data=data, model_path=self.model_path, max_len=self.max_len, step_size=self.prediction_step_size)
        print_cf(data)
        data.rename(columns={"prediction": "hatespeech_pred", "probability": "hatespeech_prob"}, inplace=True)

        if save_file:
            save_prediction_file(data, data_name, 'hatespeech', self.predictions_path)
        return data

########################################## SELFHARM ##################################################################

class Selfharm:
    def __init__(self):
        config = Info()
        task_model = config.tasks['selfharm']
        self.step_size = config.training_step_size
        self.prediction_step_size = config.prediction_step_size
        self.learning_rate = task_model['learning_rate']
        self.max_len = task_model['max_len']
        self.epochs = task_model['epochs']
        self.batch_size = task_model['batch_size']
        self.model = task_model['model']
        self.model_path = config.paths['trained_models_dir'] + 'selfharm'
        self.predictions_path = config.paths['results_dir']

    def train(self, train_data=None, num_samples='all', return_model=True):
        if not train_data:
            train_data = LoadData('selfharm').load_by_name_or_path('train_set', num_samples=num_samples)
        train_data = check_token_len(train_data, self.max_len)

        model = trainX(model_name=self.model,
                       model_path=self.model_path,
                       train_data=train_data,
                       max_len=self.max_len,
                       step_size=self.step_size,
                       epochs=self.epochs, batch_size=self.batch_size, learning_rate=self.learning_rate)

        if return_model:
            return model

    def predict(self, data=None, num_samples='all', save_file=True):
        data_name = data
        if type(data) == str and data in ['train_set', 'test_set','stories']:
            data = LoadData('selfharm').load_by_name_or_path(data, num_samples=num_samples)
        data = check_token_len(data, self.max_len)

        data = predictX(model_name=self.model, data=data, model_path=self.model_path, max_len=self.max_len, step_size=self.prediction_step_size)

        print('selfharm ', self.model, ' prediction finished')

        data['sentiment_prediction'] = int(1)
        data['sentiment_probability'] = data['probability']

        data[['sentiment_prediction', 'sentiment_probability']] = predict_sentiment(data)

        data['selfharm_prob'] = data['probability']
        data['selfharm_pred'] = int(1)

        data['selfharm_pred'] = data['prediction'] & data['sentiment_prediction']
        data.loc[data['selfharm_pred'] == 1, 'selfharm_prob'] = (data.loc[data['selfharm_pred'] == 1, 'probability'] +
                                                                 data.loc[data[
                                                                              'selfharm_pred'] == 1, 'sentiment_probability']) / 2
        print('ensemble predictions for selfharm finished')

        print_cf(data)
        if save_file:
            save_prediction_file(data, data_name, 'selfharm', self.predictions_path)
        return data


########################################## SPAM ########################################################################

class Spam:
    def __init__(self):
        config = Info()
        task_model = config.tasks['spam']
        self.step_size = config.training_step_size
        self.prediction_step_size = config.prediction_step_size
        self.learning_rate = task_model['learning_rate']
        self.max_len = task_model['max_len']
        self.epochs = task_model['epochs']
        self.batch_size = task_model['batch_size']
        self.model = task_model['model']
        self.model_path = config.paths['trained_models_dir'] + 'spam'
        self.predictions_path = config.paths['results_dir']

    def train(self, train_data=None, num_samples='all', return_model=True):
        if not train_data:
            train_data = LoadData('spam').load_by_name_or_path('train_set', num_samples=num_samples)
        train_data = check_token_len(train_data, self.max_len)
        model = trainX(model_name=self.model,
                       model_path=self.model_path,
                       train_data=train_data,
                       max_len=self.max_len,
                       step_size=self.step_size,
                       epochs=self.epochs, batch_size=self.batch_size, learning_rate=self.learning_rate)

        if return_model:
            return model

    def predict(self, data=None, num_samples='all', save_file=True):
        data_name = data
        if type(data) == str and data in ['train_set', 'test_set','stories']:
            data = LoadData('spam').load_by_name_or_path(data, num_samples=num_samples)
        data = check_token_len(data, self.max_len)

        data = predictX(model_name=self.model, data=data, model_path=self.model_path, max_len=self.max_len, step_size=self.prediction_step_size)

        print_cf(data)
        data.rename(columns={"prediction": "spam_pred", "probability": "spam_prob"}, inplace=True)
        if save_file:
            save_prediction_file(data, data_name, 'spam', self.predictions_path)

        return data

########################################## all kinds of toxicity ########################################################


class AllToxicity:
    def __init__(self):
        self.predictions_path = Info().paths['results_dir'] + 'all_toxicity.csv'

    def train(self, num_samples='all'):
        Spam().train(num_samples=num_samples, return_model=False)
        Selfharm().train(num_samples=num_samples, return_model=False)
        Hatespeech().train(num_samples=num_samples, return_model=False)

    def predict(self, data='stories', num_samples='all', save_file=False):
        if type(data) == str:
            data = LoadData('all').load_by_name_or_path(data, num_samples=num_samples)
        else:
            data = pd.DataFrame({'data': data, 'prediction': [None], 'probability': [None]})
            preprocess_rows(data)

        data = Spam().predict(data=data, num_samples=num_samples, save_file=False)
        data = Hatespeech().predict(data=data, num_samples=num_samples, save_file=False)
        data = Selfharm().predict(data=data, num_samples=num_samples, save_file=False)

        data['prediction'] = 0

        for index, row in data.iterrows():
            if data.loc[index,'spam_pred'] == 1 and data.loc[index, 'prediction']>0.55:
                data.loc[index, 'prediction'] = 'spam'
                data.loc[index, 'probability'] = row['spam_prob']
            elif data.loc[index,'selfharm_pred'] == 1 and data.loc[index,'hatespeech_pred'] == 1:
                if data.loc[index,'selfharm_prob'] > data.loc[index,'hatespeech_prob']:
                    data.loc[index, 'prediction'] = 'selfharm'
                    data.loc[index, 'probability'] = data.loc[index,'selfharm_prob']
                else:
                    data.loc[index, 'prediction'] = 'hatespeech'
                    data.loc[index, 'probability'] = data.loc[index,'hatespeech_prob']
            elif data.loc[index,'hatespeech_pred'] == 1:
                data.loc[index, 'prediction'] = 'hatespeech'
                data.loc[index, 'probability'] = data.loc[index,'hatespeech_prob']
            elif data.loc[index,'selfharm_pred'] == 1:
                data.loc[index, 'prediction'] = 'selfharm'
                data.loc[index, 'probability'] = data.loc[index,'selfharm_prob']
        data['probability'].fillna(1, inplace=True)
        data.loc[(data['prediction'] != 0) & (data['probability'] < 0.51), 'prediction'] = 0

        if save_file:
            data.to_csv(self.predictions_path, index=False)
            print('predictions from saved to ', self.predictions_path)
        return data
