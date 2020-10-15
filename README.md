# DetoxAI
Undesidered Content Detection


# Motivation
This project is developed for Commaful - a multimedia storytelling platform. 

One of the biggest challenges of all the social media platforms is toxic content. Some types of toxic content like hatespeech or suicidal idealization can be highly dangerous while some others like spam are less dangerous but still undesired. Currently most of toxic content filtering is done by human moderators which is very time costly considering the vast amount of data that the platforms receive each day. DetoxAI brings AI to the toxic content filtering process to help ease the burden on human moderators. The aim is to decrease the amount of posts to be reviewed by humans by safely eliminating posts that are not toxic in content and flag the ones that possibly have spam, hate speech or self harm in it.

# Data



# Model



# Setup

### Installation
```
git clone https://github.com/basakanayurt/DetoxAI
cd ./DetoxAI
```

Optional Create a conda virtual environment with the yml file
```
$ conda env create -f environment.yml
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
