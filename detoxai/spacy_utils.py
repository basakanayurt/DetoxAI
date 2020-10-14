import spacy
import pandas as pd
import numpy as np
import random
from spacy.util import minibatch


def train_spacy_model(model, train_data, optimizer, batch_size, epochs=10):
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

def spacy_predictions(model, texts):
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

