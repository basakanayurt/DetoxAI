import json


class Config:

    def __init__(self, opt=None):
        if opt == 'default':
            config_file = 'config_default.json'
        else:
            config_file = 'config.json'
        with open(config_file, 'r') as f:
            config = json.load(f)

        self.paths = config["paths"]
        self.tasks = config["tasks"]
        self.training_step_size = config["training_step_size"]
        self.prediction_step_size = config["prediction_step_size"]

    def update(self):

        config = {'paths': self.paths, 'tasks': self.tasks, 'training_step_size': self.training_step_size, 'prediction_step_size': self.prediction_step_size}
        with open('config.json', 'w') as f:
            json.dump(config, f)

# config = Config()
# config.update()


