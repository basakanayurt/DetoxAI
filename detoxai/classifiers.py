from transformers import (
    DistilBertTokenizer,
    TFDistilBertForSequenceClassification,
    pipeline
)
from detoxai.spacy_utils import *
import tensorflow as tf
import numpy as np
from detoxai.data_loader import *
from detoxai.preprocess import *


class Models(LoadData):
    def __init__(self, task='all', learning_rate=5e-5, step_size=10000, epochs=(3,20), batch_size=64, max_len=256, save_model=True):
        self.task = task
        self.learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.step_size = step_size
        self.epochs_spacy = epochs[1]
        self.epochs_distilbert = epochs[0]
        self.batch_size = batch_size
        self.model_path = '../trained_models/'
        self.predictions_path = '../data/predictions/'
        self.save_model = save_model
        self.max_len = max_len

    def check_token_len(self,data):
        data['original'] = data['data']
        # if the text has more than max_len tokens get half from beginning half from the end:
        for index, row in data.iterrows():
            if len(tokenize_text(row['data'])) >= self.max_len-2:
                temp1 = ' '.join(tokenize_text(row['data'])[:(self.max_len//2)-10])
                temp2 = ' '.join(tokenize_text(row['data'])[-((self.max_len)//2):])
                temp = temp1 + ' ' + temp2
                data.loc[index, 'data'] = temp
        return data

    def _print_cf(self, data,type_prediction='prediction'):
        if 'label' in data.columns:
            accuracy = accuracy_score(data['label'].tolist(),
                                      data[type_prediction].tolist())  # (y_true, y_predictions)
            cf_matrix = confusion_matrix(data['label'].tolist(), data[type_prediction].tolist())
            print("accuracy: {}".format(accuracy))
            print("CF matrix: {}".format(cf_matrix))
########################################### TRAIN  #####################################################################

    ########################################### distilbert training  ####################################################

    def train_distilbert(self,train_data):
        # save text and labels into different variables
        train_texts = train_data['data'].to_list()
        train_labels = train_data['label'].to_list()
        # to avoid OOM
        steps = len(train_data) // self.step_size

        # Initialize a distilbert model
        print('start training of distilbert')
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True, add_special_tokens=True,
                                                        max_length=self.max_len, pad_to_max_length=True)
        model = TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels = 2)
        model.compile(optimizer=optimizer, loss=model.compute_loss)

        for step in range(steps+1):
            print ('step-', step, '/', steps+1)
            # break into chunks and update the model (otherwise OOM error)
            if step != steps+1:
                train_texts_i = train_texts[step*self.step_size: (step+1)*self.step_size]
                train_labels_i = train_labels[step*self.step_size: (step+1)*self.step_size]
            else:
                train_texts_i = train_texts[step*self.step_size:]
                train_labels_i = train_labels[step*self.step_size:]
            if train_texts_i:
                # encodings
                train_encodings = tokenizer(train_texts_i, truncation=True, padding='max_length', return_tensors='tf')
                train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings), train_labels_i))
                # model update
                model.fit(train_dataset.shuffle(2).batch(16), epochs=self.epochs_distilbert, batch_size=self.batch_size)

        print('distilbert training finished')
        return model

    ########################################### spacy training  ####################################################

    def train_spacy(self, train_data):
        nlp = spacy.blank("en")
        print('empty spacy model loaded')

        # Create the TextCategorizer with exclusive classes and "cnn" architecture
        text_cat = nlp.create_pipe("textcat", config={"exclusive_classes": True,"architecture": "cnn"})

        # Adding the TextCategorizer to the created empty model
        nlp.add_pipe(text_cat)
        # Add labels to text classifier
        text_cat.add_label("0")
        text_cat.add_label("1")
        optimizer = nlp.begin_training()
        steps = len(train_data) // self.step_size

        for step in range(steps + 1):
            print('step-', step, '/', steps + 1)

            # break into chunks and update the model (otherwise OOM error)
            if step != steps + 1:
                train_i = train_data[step * self.step_size: (step + 1) * self.step_size]
            else:
                train_i = train_data[step * self.step_size:]
            if len(train_i) > 0:
                # Create the train and val data for the spacy model
                train_labels_i = [{'cats': {'0': label == 0,
                                            '1': label == 1}} for label in train_i['label']]
                train_data_i = list(zip(train_i['data'], train_labels_i))

                # Train the model
                nlp, losses = train_spacy_model(nlp, train_data_i, optimizer, self.batch_size, self.epochs_spacy)

        print('training finished')
        return nlp

    ###########################################  training  tasks ########################################################

    def train(self,file_path_or_name='train_set', num_samples='all'):
        if self.task == 'all':
            tasks = ['hatespeech', 'selfharm','spam']
        else:
            tasks = [self.task]

        for task in tasks:
            train_data = LoadData(task).load_by_name_or_path( file_path_or_name=file_path_or_name,num_samples=num_samples)
            train_data[['prediction', 'probability']] = [None,None]
            if task != 'spam':
                model = self.train_distilbert(train_data)
                if self.save_model:
                    # save the model
                    model.save_pretrained( self.model_path + task)
                    print('model saved to ',self.model_path + task)
            elif task == 'spam':
                if self.save_model:
                    train_data = LoadData(task).load_by_name_or_path(file_path_or_name=file_path_or_name, num_samples=num_samples)
                    model =  self.train_spacy(train_data)
                    model.to_disk(self.model_path + 'spam')
                    print('spacy model saved to ', self.model_path + 'spam' )
            else:
                print("task doesn't exist-no training done")
                return


########################################### PREDICT  #####################################################################

    ########################################### predict distilbert #####################################################

    def predict_distilbert(self, data, task):
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True,
                                                        add_special_tokens=True,
                                                        max_length=self.max_len, pad_to_max_length=True)
        model = TFDistilBertForSequenceClassification.from_pretrained(self.model_path + task)
        # to avoid OOM
        steps = len(data) // self.step_size

        for step in range(steps + 1):
            print('step-', step, '/', steps + 1)
            begin = step * self.step_size

            try:
                if step != steps + 1:
                    end = (step + 1) * self.step_size
                    data_i = data[begin: end]
                else:
                    data_i = data[step * self.step_size:]
                    end = step * self.step_size + len(data_i)

                if len(data_i) != 0:
                    encodings = tokenizer(data_i['data'].tolist(), truncation=True, padding='max_length', return_tensors='tf')
                    output = model(encodings)
                    tf_predictions = tf.nn.softmax(output, axis=-1)
                    predictions = np.argmax(tf_predictions[0].numpy(), axis=1)
                    probs = np.max(tf_predictions[0].numpy(), axis=1)
                data.loc[begin:end - 1, 'probability'] = probs
                data.loc[begin: end - 1, 'prediction'] = predictions
            except:
                print('skipped step', step)
                data.loc[begin:end - 1, 'probability'] = 0
                data.loc[begin: end - 1, 'prediction'] = 0
        data['prediction'] = data['prediction'].apply(lambda x: int(x))

        return data

    ########################################### predict spacy #####################################################

    def predict_spacy(self, data):
        print('spacy predictions from trained model:')
        model = spacy.load(self.model_path+'spam')
        steps = len(data) // self.step_size

        for step in range(steps + 1):
            print('step-', step, '/', steps + 1)
            begin = step * self.step_size

            try:
                if step != steps + 1:
                    end = (step + 1) * self.step_size
                    data_i = data[begin: end]
                else:
                    data_i = data[step * self.step_size:]
                    end = step * self.step_size + len(data_i)

                if len(data_i) != 0:
                    # Use the model's tokenizer to tokenize each input text
                    docs = [model.tokenizer(text) for text in data_i['data'].tolist()]
                    # Use textcat to get the scores for each doc
                    textcat = model.get_pipe('textcat')
                    scores, _ = textcat.predict(docs)

                    # From the scores, find the label with the highest score/probability
                    predicted_labels = scores.argmax(axis=1)
                    predictions = [int(label) for label in predicted_labels]
                    probs = np.max(scores, axis=1).tolist()

                data.loc[begin:end - 1, 'probability'] = probs
                data.loc[begin: end - 1, 'prediction'] = predictions
            except:
                print('skipped step', step)
                data.loc[begin:end - 1, 'probability'] = 0
                data.loc[begin: end - 1, 'prediction'] = 0
        data['prediction'] = data['prediction'].apply(lambda x: int(x))

        print(self.task, ' prediction finished')
        self._print_cf(data)

        return data

    ########################################### predict tasks #####################################################

    def predict_sentiment(self, data):
        model = pipeline("sentiment-analysis")
        data['sentiment_prediction'] = int(1)
        data['sentiment_probability'] = data['probability']
        mapping = {'NEGATIVE': 1, 'POSITIVE': 0}
        for index, row in data.iterrows():
            try:
                temp = model( row['data'])[0]
                data.loc[index, 'sentiment_prediction'] = mapping[temp['label']]
                data.loc[index, 'sentiment_probability'] = temp['score']
            except:
                print('skipped index:', index)
                pass
        print('sentiment prediction finished')
        return data[['sentiment_prediction','sentiment_probability']]

    def predict_hatespeech(self, data, savefile=False):
        data = self.predict_distilbert(data,task='hatespeech')
        print('hatespeech distilbert prediction finished')
        self._print_cf(data)
        data.rename(columns={"prediction": "hatespeech_pred", "probability": "hatespeech_prob"}, inplace=True)
        if savefile:
            data.to_csv(self.predictions_path + 'only_hatespeech.csv', index=False)
        return data

    def predict_selfharm(self,data, savefile=False):
        data = self.predict_distilbert(data,task='selfharm')
        print('selfharm distilbert prediction finished')

        data['sentiment_prediction'] = int(1)
        data['sentiment_probability'] = data['probability']

        data[['sentiment_prediction', 'sentiment_probability']] = self.predict_sentiment(data)

        data['selfharm_prob'] = data['probability']
        data['selfharm_pred'] = int(1)

        data['selfharm_pred'] = data['prediction'] & data['sentiment_prediction']
        data.loc[data['selfharm_pred']==1,'selfharm_prob'] = (data.loc[data['selfharm_pred']==1,'probability'] +
                                                                 data.loc[data['selfharm_pred']==1,'sentiment_probability'])/2


        print ('ensemble predictions for selfharm finished')
        self._print_cf(data,'selfharm_pred')
        if savefile:
            data.to_csv(self.predictions_path + 'only_selfharm.csv', index=False)
        return data

    def predict_spam(self,data, savefile=False):
        data = self.predict_spacy(data)
        self._print_cf(data)
        data.rename(columns={"prediction": "spam_pred", "probability": "spam_prob"}, inplace = True)
        print('spam spacy prediction finished')
        if savefile:
            data.to_csv(self.predictions_path + 'only_spams.csv', index=False)
        return data

    def predict_from_data(self,data,savefile=False):

        data = self.check_token_len(data)
        data = self.predict_hatespeech(data, savefile=savefile)
        data = self.predict_selfharm(data, savefile=savefile)
        data = self.predict_spam(data, savefile=savefile)

        data['prediction'] = 0

        for index, row in data.iterrows():
            if row['spam_pred'] == 1 and data.loc[index, 'prediction']>0.55:
                data.loc[index, 'prediction'] = 'spam'
                data.loc[index, 'probability'] = row['spam_prob']
            elif row['selfharm_pred'] == 1 and row['hatespeech_pred'] == 1:
                if row['selfharm_prob'] > row['hatespeech_prob']:
                    data.loc[index, 'prediction'] = 'selfharm'
                    data.loc[index, 'probability'] = row['selfharm_prob']
                else:
                    data.loc[index, 'prediction'] = 'hatespeech'
                    data.loc[index, 'probability'] = row['hatespeech_prob']
            elif row['hatespeech_pred'] == 1:
                data.loc[index, 'prediction'] = 'hatespeech'
                data.loc[index, 'probability'] = row['hatespeech_prob']
            elif row['selfharm_pred'] == 1:
                data.loc[index, 'prediction'] = 'selfharm'
                data.loc[index, 'probability'] = row['selfharm_prob']
        data['probability'].fillna(1, inplace=True)
        data.loc[(data['prediction'] != 0) & (data['probability'] < 0.51), 'prediction'] = 0

        return data

    def predict_from_path(self, file_path_or_name, num_samples='all', savefile=False):

        data = LoadData('all').load_by_name_or_path(file_path_or_name=file_path_or_name,num_samples=num_samples)
        data = self.predict_from_data(data, savefile=savefile)

        if savefile:
            data.to_csv(self.predictions_path + 'stories_predictions.csv', index=False)
            print ('predictions from ', file_path_or_name, ' saved to ', self.predictions_path + 'stories.csv')

        return data

