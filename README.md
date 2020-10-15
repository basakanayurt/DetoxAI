# DetoxAI
Toxic Content Filter


# Motivation

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
### Stream (Optional)
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
