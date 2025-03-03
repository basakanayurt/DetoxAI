# DetoxAI
Undesidered Content Detection


# Motivation
This project is developed for Commaful - a multimedia storytelling platform. 

One of the biggest challenges of all the social media platforms is toxic content. Some types of toxic content like hatespeech or suicidal idealization can be highly dangerous while some others like spam are less dangerous but still undesired. Currently most of toxic content filtering is done by human moderators which is very time costly considering the vast amount of data that the platforms receive each day. DetoxAI brings AI to the toxic content filtering process to help ease the burden on human moderators. The aim is to decrease the amount of posts to be reviewed by humans by safely eliminating posts that are not toxic in content and flag the ones that possibly have spam, hate speech or self harm in it.

<p align="center"> <img src="/img/goal.png"  width="500"> </p>


# Data
Since there are 3 different classification tasks to be accomplished there are 3 different datasets that are used for training purposes. Spam data is provided by Commaful. However, hate speech and self harm data are curated by using various public sources. To be specific

* Spam : Commaful proprietary data

* Hate speech : https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data

* Self harm : combined different sourced as there isn't a dataset on this purpose

  * Scraped Reddit Suicide Watch : https://www.reddit.com/r/SWResources/
  
  * Poem data set : https://www.kaggle.com/johnhallman/complete-poetryfoundationorg-dataset
  
  * Sentiment140 : http://help.sentiment140.com/for-students


# Model
* Spam classifier is based on convolutional neural nets (Spacy - https://spacy.io/api/textcategorizer)
* DistilBERT is fine-tuned for hate speech and self harm classification purposes (Huggingface - https://huggingface.co/transformers/model_doc/distilbert.html)
* A pre-trained sentiment classifier is used in ensemble with fine-tuned DistilBERT for self harm classification (https://huggingface.co/transformers/main_classes/pipelines.html)

<p align="center"> <img src="/img/pipeline.jpg"  width="800"> </p>

# Setup

### Installation
```
git clone https://github.com/basakanayurt/DetoxAI
cd ./DetoxAI
```
Optional: Create a conda virtual environment with the yml file
```
$ conda env create -f environment.yml
$ conda activate detoxai
```
Or install the requirements
```
$ pip install -r requirements.txt   
```
### Streamlit app (Optional)
Run the Streamlit app for fun,
```
streamlit run app.py
```
### Docker (Optional)
To containerize the Streamlit app,
* Install Docker [Desktop](https://www.docker.com/products/docker-desktop) or [Engine](https://docs.docker.com/engine/)
* Run the following commands for docker build and run
```
docker build -t detoxai:latest -f DockerFile .
docker run -p 8501:8501 detoxai:latest
```

## Usage

Make predictions from the pretrained models
```
from detoxai.content_detector import *
from detoxai.config import *

config = Config("default")
config.paths['stories_file_path']='PATH/TO/CSV_FILE/FOR/PREDICTION' 
config.update()

model = AllToxicity()
model.predict(data="stories", save_file=True)
```

Each classifier (spam, hatespeech, self harm) can be trained from scratch and tested on a different dataset if preferred. Below is an example for training the hate speech classifier:
```
config = Config("default")
task = 'hatespeech'
config.paths['hatespeech_train_set_path'] = "PATH/TO/TRAIN_SET"
config.paths['hatespeech_test_set_path'] = "PATH/TO/TEST_SET"
config.paths['hatespeech_test_set_path'] = "PATH/TO/TEST_SET"
config.paths['trained_models_dir'] =  "PATH/TO/SAVE/THE/NEW/TRAINED/MODEL" # if not changed it would overwrite the existing 'hatespeech' model in the ./trained_models

config.tasks[task]['model'] = 'distilbert'
config.tasks[task]['learning_rate'] = 5e-5
config.tasks[task]['max_len'] = 512 #max token length
config.tasks[task]['epochs'] = 3 
config.update()

hatespeech_model = Hatespeech()
hatespeech_model.train()
hatespeech.predict(data="test_set", save_file=True)
```

Spam and self harm classifiers can be trained in similar fashion.

### Important Note:
During the overall content detection (AllToxicity().predict(data="stories", save_file=True)) the algorithm looks for the classifier models in the path provided by "config.paths['trained_models_dir']". This means that the path in "config.paths['trained_models_dir']" should have 3 folders named precisely as "spam", "hatespeech" and "selfharm".

If you retrain the models and save them under different paths do not forget to bring them altogether in a single folder and provide it's path in "config.paths['trained_models_dir']". 


