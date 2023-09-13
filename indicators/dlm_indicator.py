import logging
import os

import backtrader as bt
import joblib
import numpy as np
import pandas as pd
import datetime as dt
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, mean_absolute_percentage_error, median_absolute_error

from machine_learning.ta_indicators import TAIndicators

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DLMIndicator(bt.Indicator):
    lines = ('RF_prediction', 'GBM_prediction', 'ABR_prediction', 'ETR_prediction')
    params = (('observe_window', 500),)
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.load_models_and_scalers()        
        self.running_df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
        self.prediction_df = pd.DataFrame(columns=['model_type', 'prediction'])
        self.ta = TAIndicators(self.running_df)
        self.first_run = True
        
    def load_models_and_scalers(self):
        model_types = ['RF', 'GBM', 'ABR', 'ETR']
        for model_type in model_types:
            model_dir = f'./models/DLM/{model_type}/'
            if os.path.exists(model_dir):
                for filename in os.listdir(model_dir):
                    if filename.endswith('.pkl'):
                        file_path = os.path.join(model_dir, filename)
                        if 'scaler' in filename:
                            self.scalers[model_type] = joblib.load(file_path)
                            logger.info(f'Loaded scaler: {file_path}')
                        else:
                            self.models[model_type] = joblib.load(file_path)
                            logger.info(f'Loaded model: {file_path}')
                            
    # def get_predictions_accuracy(self):
        # current_close = self.data.close[0]
        # for model_type in self.prediction_df:
        #     r2 = r2_score(self.data.close[self.data.datetime.minute[0] in [0, 30]], self.prediction_df['prediction'])
        #     mape = mean_absolute_percentage_error(self.data.close[self.data.datetime.minute[0] in [0, 30]], self.prediction_df['prediction'])
        #     mae = mean_absolute_error(self.data.close[self.data.datetime.minute[0] in [0, 30]], self.prediction_df['prediction'])
        #     mse = mean_squared_error(self.data.close[self.data.datetime.minute[0] in [0, 30]], self.prediction_df['prediction'])
        #     rmse = mean_squared_error(self.data.close[self.data.datetime.minute[0] in [0, 30]], self.prediction_df['prediction'], squared=False)
    
        #     logger.info(f'Model {model_type}, R2 score: {r2}')
        #     logger.info(f'Model {model_type}, Mean Absolute Percentage Error: {mape}')
        #     logger.info(f'Model {model_type}, Mean Absolute Error: {mae}')
        #     logger.info(f'Model {model_type}, Mean Squared Error: {mse}')
        #     logger.info(f'Model {model_type}, Root Mean Squared Error: {rmse}')
        # return r2, mape, mae, mse, rmse
        

    # def predict(self, new_close, model_type=None):
    #     if model_type is None:
    #         model_type = self.params.model_type
    #     prediction = self.models[model_type].predict(new_close)
    #     self.prediction_df = self.prediction_df.append({'model_type': model_type, 'prediction': prediction}, ignore_index=True)
    #     return prediction
                   
        
        
    def next(self):
        new_df = []
        transformed_df = []
        # Collect current bar data
        current_data = {
            'timestamp': self.data.datetime.datetime(0),
            'open': self.data.open[0],
            'high': self.data.high[0],
            'low': self.data.low[0],
            'close': self.data.close[0],
            'volume': self.data.volume[0]
        }
       
        # Convert to DataFrame
        new_df = pd.DataFrame([current_data], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])    
        new_df['timestamp'] = pd.to_datetime(new_df['timestamp'], unit='ms')
        new_df.set_index('timestamp', inplace=True)
        
        # Extract the minute component of the timestamp in new_df
        minute_component = new_df.index.minute[0]

        # Check if the minute component is 00 or 30
        if minute_component in [0, 30]:            
            self.running_df = pd.concat([self.running_df, new_df], ignore_index=False).sort_index()
        else:
            return        

        if self.first_run:
            print(f'self.running_df:\n{self.running_df.head(3)}')

        if len(self.running_df) < 10:
            return

        # Limit the DataFrame to observe_window rows
        if len(self.running_df) > self.params.observe_window:
            self.running_df = self.running_df.iloc[-self.params.observe_window:]
        
        # Calculate indicators        
        self.running_df = TAIndicators(self.running_df)._calculate_indicators()

        if self.first_run:
            print(f'self.running_df head:\n{self.running_df.head(3)}')
            print(f'self.running_df tail:\n{self.running_df.tail(3)}')
        
        # Make predictions using loaded models
        for model_type, model in self.models.items():
            scaler = self.scalers.get(model_type)
            if self.first_run:
                logger.info(f'Loaded scaler: {scaler} for model: {model_type}')
            if scaler:
                transformed_df = self.running_df.copy()                
                transformed_df[transformed_df.columns] = scaler.fit_transform(transformed_df)
                # Replace infinite values with NaN
                transformed_df.replace([np.inf, -np.inf], np.nan, inplace=True)
                transformed_df.dropna(inplace=True)  # This will drop rows with NaN values                

                X_scaled = transformed_df.drop(columns=['close'], axis=1)  
                
            else:
                X_scaled = self.running_df.drop(columns=['close'])
                if self.first_run:
                    logger.info(f'No scaler loaded')
            
            prediction = model.predict(X_scaled)
            # logger.info(f'Prediction: {prediction}')
            if self.first_run:
                logger.info(f'Not scaled back prediction[-1] {model_type}: {prediction[-1]:.4f}')            
                logger.info(f'Not scaled back prediction[0] {model_type}: {prediction[0]:.4f}')            

            # Inverse transform the prediction using the scaler fitted on the 'close' column
            if scaler:
                scaler.fit_transform(self.running_df[['close']])
                prediction = scaler.inverse_transform(prediction.reshape(-1, 1)).flatten()

            if self.first_run:            
                logger.info(f'Scaled back prediction[-1] {model_type}: {prediction[-1]:.4f}')
                logger.info(f'Scaled back prediction[0] {model_type}: {prediction[0]:.4f}')
                       

            # Assign each model's prediction to its respective line
            if model_type == 'RF':
                self.lines.RF_prediction[0] = prediction[-1] if minute_component in [0, 30] else self.lines.RF_prediction[-1]
            elif model_type == 'GBM':
                self.lines.GBM_prediction[0] = prediction[-1] if minute_component in [0, 30] else self.lines.GBM_prediction[-1]
            elif model_type == 'ABR':
                self.lines.ABR_prediction[0] = prediction[-1] if minute_component in [0, 30] else self.lines.ABR_prediction[-1]
            elif model_type == 'ETR':
                self.lines.ETR_prediction[0] = prediction[-1] if minute_component in [0, 30] else self.lines.ETR_prediction[-1]

            
            # Save to DataFrame
            predict_new_data = {
                'timestamp': self.data.datetime.datetime(0),
                'model_type': model_type,
                'prediction': prediction[-1]
            }
            predict_new_data = pd.DataFrame([predict_new_data], columns=['timestamp', 'model_type', 'prediction'])
            predict_new_data['timestamp'] = pd.to_datetime(predict_new_data['timestamp'], unit='ms')
            predict_new_data.set_index('timestamp', inplace=True)

            min_component = predict_new_data.index.minute[0]
            # Check if the minute component is 00 or 30
            if min_component in [0, 30]:
                self.prediction_df = pd.concat([self.prediction_df, predict_new_data], ignore_index=False).sort_index()
                if self.first_run:
                    logger.info(f'Prediction DataFrame:\n{self.prediction_df.head(3)}')

            transformed_df = []
            # prediction = []            
            X_scaled = []
            scaler = None
        
        self.first_run = False


# # Usage in Backtrader
# if __name__ == '__main__':
#     cerebro = bt.Cerebro()
#     # Add data feed, strategy, etc.
    
#     cerebro.addindicator(DeapLearningIndicator)
#     cerebro.run()
