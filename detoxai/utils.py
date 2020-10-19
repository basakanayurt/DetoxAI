import spacy
import pandas as pd
import numpy as np
import random
from spacy.util import minibatch
from transformers import (
    DistilBertTokenizer,
    TFDistilBertForSequenceClassification,
    pipeline
)
import tensorflow as tf
import numpy as np
from detoxai.preprocess import *
from sklearn.metrics import accuracy_score, confusion_matrix


def train_spacy_model_util(model, train_data, optimizer, batch_size, epochs=10):
    losses = {}
    # random.seed(1)
    for epoch in range(epochs):
        random.shuffle(train_data)

        batches = minibatch(train_data, size=batch_size)
        for batch in batches:
            # Split batch into texts and labels
            texts, labels = zip(*batch)

            # Update model with texts and labels
            model.update(texts, labels, sgd=optimizer, losses=losses)
        print("Loss: {}".format(losses['textcat']))

    return model, losses['textcat']

def spacy_predictions_util(model, texts):
    # Use the model's tokenizer to tokenize each input text
    docs = [model.tokenizer(text) for text in texts]
    # Use textcat to get the scores for each doc
    textcat = model.get_pipe('textcat')
    scores, _ = textcat.predict(docs)

    # From the scores, find the label with the highest score/probability
    predicted_labels = scores.argmax(axis=1)
    predicted_class = [int(label) for label in predicted_labels]
    prediction_probs = np.max(scores, axis=1).tolist()
    return prediction_probs, predicted_class


def train_distilbert(train_data, max_len=256, step_size=10000, epochs=3, batch_size=64, learning_rate=5e-5):

    # save text and labels into different variables
    train_texts = train_data['data'].to_list()
    train_labels = train_data['label'].to_list()
    # to avoid OOM
    steps = len(train_data) // step_size

    # Initialize a distilbert model
    print('start training of distilbert')
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True,
                                                    add_special_tokens=True,
                                                    max_length=max_len, pad_to_max_length=True)
    model = TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    model.compile(optimizer=optimizer, loss=model.compute_loss)

    for step in range(steps + 1):
        print('step-', step, '/', steps + 1)
        # break into chunks and update the model (otherwise OOM error)
        if step != steps + 1:
            train_texts_i = train_texts[step * step_size: (step + 1) * step_size]
            train_labels_i = train_labels[step * step_size: (step + 1) * step_size]
        else:
            train_texts_i = train_texts[step * step_size:]
            train_labels_i = train_labels[step * step_size:]
        if train_texts_i:
            # encodings
            train_encodings = tokenizer(train_texts_i, truncation=True, padding='max_length', return_tensors='tf')
            train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings), train_labels_i))
            # model update
            model.fit(train_dataset.shuffle(2).batch(16), epochs=epochs, batch_size=batch_size)

    print('distilbert training finished')

    return model



def predict_distilbert(model_path, max_len, data, step_size=150):
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True,
                                                    add_special_tokens=True,
                                                    max_length=max_len, pad_to_max_length=True)
    model = TFDistilBertForSequenceClassification.from_pretrained(model_path)
    # to avoid OOM
    steps = len(data) // step_size

    for step in range(steps + 1):
        print('step-', step, '/', steps + 1)
        begin = step * step_size

        try:
            if step != steps + 1:
                end = (step + 1) * step_size
                data_i = data[begin: end]
            else:
                data_i = data[step * step_size:]
                end = step * step_size + len(data_i)

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


def train_spacy(train_data, step_size=10000, epochs=20, batch_size=64):
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
    steps = len(train_data) // step_size

    for step in range(steps + 1):
        print('step-', step, '/', steps + 1)

        # break into chunks and update the model (otherwise OOM error)
        if step != steps + 1:
            train_i = train_data[step * step_size: (step + 1) * step_size]
        else:
            train_i = train_data[step * step_size:]
        if len(train_i) > 0:
            # Create the train and val data for the spacy model
            train_labels_i = [{'cats': {'0': label == 0,
                                        '1': label == 1}} for label in train_i['label']]
            train_data_i = list(zip(train_i['data'], train_labels_i))

            # Train the model
            nlp, losses = train_spacy_model_util(nlp, train_data_i, optimizer, batch_size, epochs)

    print('training finished')
    return nlp


def predict_spacy(model_path, data, step_size=1000):
    print('spacy predictions from trained model:')
    model = spacy.load(model_path)
    steps = len(data) // step_size

    for step in range(steps + 1):
        print('step-', step, '/', steps + 1)
        begin = step * step_size

        try:
            if step != steps + 1:
                end = (step + 1) * step_size
                data_i = data[begin: end]
            else:
                data_i = data[step * step_size:]
                end = step * step_size + len(data_i)

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

    print('spacy prediction finished')

    return data


def predict_sentiment(data):
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


def trainX(model_name, model_path, train_data, max_len, step_size, epochs, batch_size, learning_rate):
    if model_name == 'distilbert':
        model = train_distilbert(train_data, max_len=max_len, step_size=step_size, epochs=epochs, \
                                 batch_size=batch_size, learning_rate=learning_rate)
        # save the model
        model.save_pretrained(model_path)
        print('model saved to ', model_path)
    if model_name == 'spacy':
        model = train_spacy(train_data, step_size=step_size, epochs=epochs, batch_size=batch_size)
        # save the model
        model.to_disk(model_path)
        print('model saved to ', model_path)

    return model


def predictX(model_name,data, model_path,max_len,step_size):
    if model_name == 'distilbert':
        data = predict_distilbert(model_path=model_path, data=data, max_len=max_len, step_size=step_size)
    elif model_name == 'spacy':
        data = predict_spacy(model_path, data=data, step_size=step_size)
    return data


def check_token_len(data, max_len):
    data['original'] = data['data']
    # if the text has more than max_len tokens get half from beginning half from the end:
    for index, row in data.iterrows():
        if len(tokenize_text(row['data'])) >= max_len-2:
            temp1 = ' '.join(tokenize_text(row['data'])[:(max_len//2)-10])
            temp2 = ' '.join(tokenize_text(row['data'])[-((max_len)//2):])
            temp = temp1 + ' ' + temp2
            data.loc[index, 'data'] = temp
    return data


def print_cf(data,type_prediction='prediction'):
    if 'label' in data.columns:
        accuracy = accuracy_score(data['label'].tolist(),
                                  data[type_prediction].tolist())  # (y_true, y_predictions)
        cf_matrix = confusion_matrix(data['label'].tolist(), data[type_prediction].tolist())
        print("accuracy: {}".format(accuracy))
        print("CF matrix: {}".format(cf_matrix))


def save_prediction_file(data, data_name, task, predictions_path):
    print (data_name)
    if type(data_name) == str and data_name in ['train_set', 'test_set', 'stories']:
        data.to_csv(predictions_path + 'only_' + task + '_' + data_name + '.csv', index=False)
        print ('predictions saved to ', predictions_path + 'only_' + task + '_' + data_name + '.csv')
    else:
        data.to_csv(predictions_path + 'only_' + task + '.csv', index=False)
        print('predictions saved to ', predictions_path + 'only_' + task + '.csv')