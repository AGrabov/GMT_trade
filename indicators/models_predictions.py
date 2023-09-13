import os
import backtrader as bt
import joblib
import pandas as pd
import tensorflow as tf


class ModelPredictor(): #(bt.Indicator):
    # lines = ('prediction',)
    
    def __init__(self):
        self.models = {}
        self.load_models()

    def load_models(self):
        for dirpath, dirnames, filenames in os.walk('./models/'):            
            for filename in filenames:
                if (filename.endswith('.pkl') or filename.endswith('.zip') or filename.endswith('.h5')):  # Replace with your model file extension
                    model_path = os.path.join(dirpath, filename)
                    dirname = dirpath.split('/')[-1]
                    print(f'Found model: {filename} \nat {dirname}')
                    # model_name = filename.split('.')[0]
                    # self.models[model_name] = joblib.load(model_path)  
        

    # def next(self):
    #     current_data = self.data  # Replace with your feature extraction logic
    #     for model_name, model in self.models.items():
    #         prediction = model.predict(current_data)  # Replace with your prediction logic
    #         self.lines.prediction[0] = prediction  # Store the prediction in the line object

# Usage in Backtrader
# cerebro = bt.Cerebro()
# data = bt.feeds.YourDataFeed(dataname='your_data.csv')
# cerebro.adddata(data)
# cerebro.addindicator(ModelPredictor)
# cerebro.run()
model_predictor = ModelPredictor()
model_predictor.load_models() 
