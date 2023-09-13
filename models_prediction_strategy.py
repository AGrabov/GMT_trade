
from collections import deque, defaultdict
import backtrader as bt
import numpy as np
import datetime as dtime
import pytz
import pandas as pd
from indicators.dlm_indicator import DLMIndicator
# from indicators.DLMindicator import DLMIndicator
from indicators.a2c_prediction import A2CPredictor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLModels(bt.Signal):        
    params = {
        'dch_period': 15,
        'tsf_period': 10,

        'trade_coef': 0.5,
        
        'num_past_trades': 10,
    }

    def log(self, txt, dt=None):
        ''' Logging function fot this strategy'''
        kiev_tz = pytz.timezone('Europe/Kiev')
        dt = dt or self.data0.datetime[0]
        if isinstance(dt, float):
            dt = bt.num2date(dt, tz=kiev_tz)
        print('%s, %s' % (dt.isoformat(sep=' '), txt))

    def __init__(self):
        self.values = []        
        self.current_position_size = 0
        self.long_position = None
        self.long_entry_price = 0
        self.short_entry_price = 0 
        self.in_position = 0               
        self.pp = []        
        self.past_trades = deque(maxlen=self.params.num_past_trades)  # Store the past trades   

               
        # HeikinAshi
        self.ha = bt.indicators.HeikinAshi(self.data1)
        self.ha.plotlines.ha_high._plotskip=True 
        self.ha.plotlines.ha_low._plotskip=True
        self.ha.plotlines.ha_open._plotskip=True
        self.ha.plotlines.ha_close._plotskip=True
        
        self.ha_green = (self.ha.lines.ha_close > self.ha.lines.ha_open)
        self.ha_red = (self.ha.lines.ha_close < self.ha.lines.ha_open)

               
        # Deep Learning Models Indicator
        self.dlm = DLMIndicator(self.data1)
        self.RF_prediction = self.dlm.lines.RF_prediction
        self.ABR_prediction = self.dlm.lines.ABR_prediction
        self.GBM_prediction = self.dlm.lines.GBM_prediction
        self.ETR_prediction = self.dlm.lines.ETR_prediction
        self.RF_predictions = []
        self.RF_predictions_accuracy = []
        self.ABR_predictions = []
        self.ABR_predictions_accuracy = []
        self.GBM_predictions = []
        self.GBM_predictions_accuracy = []
        self.ETR_predictions = []
        self.ETR_predictions_accuracy = []


        # # RLM Learning Models Indicator
        # self.rlm_a2c = A2CPredictor(data=self.data1, live=False)
        # self.a2c_prediction = self.rlm_a2c.lines.prediction

        self.order = None        
    
            
    def check_buy_condition(self):        
        pass
    
    def check_sell_condition(self):        
        pass

    def check_stop_buy_condition(self):        
        pass
    
    def check_stop_sell_condition(self):
        pass

    def calculate_dlm_performance(self):
        if not np.isnan(self.RF_prediction[0]):
            self.RF_predictions.append(self.RF_prediction[0])
        if not np.isnan(self.ABR_prediction[0]):
            self.ABR_predictions.append(self.ABR_prediction[0])
        if not np.isnan(self.GBM_prediction[0]):
            self.GBM_predictions.append(self.GBM_prediction[0])
        if not np.isnan(self.ETR_prediction[0]):
            self.ETR_predictions.append(self.ETR_prediction[0])

        rf_accuracy_ratio = (1 - abs(self.data1.close[0] - self.RF_predictions[-1]) / self.RF_predictions[-1])*100 if len(self.RF_predictions) > 1 else 0
        abr_accuracy_ratio = (1 - abs(self.data1.close[0] - self.ABR_predictions[-1]) / self.ABR_predictions[-1])*100 if len(self.ABR_predictions) > 1 else 0
        gbm_accuracy_ratio = (1 - abs(self.data1.close[0] - self.GBM_predictions[-1]) / self.GBM_predictions[-1])*100 if len(self.GBM_predictions) > 1 else 0    
        etr_accuracy_ratio = (1 - abs(self.data1.close[0] - self.ETR_predictions[-1]) / self.ETR_predictions[-1])*100 if len(self.ETR_predictions) > 1 else 0
        
        if rf_accuracy_ratio != 0:
            self.RF_predictions_accuracy.append(rf_accuracy_ratio)
        if abr_accuracy_ratio != 0:
            self.ABR_predictions_accuracy.append(abr_accuracy_ratio)
        if gbm_accuracy_ratio != 0:
            self.GBM_predictions_accuracy.append(gbm_accuracy_ratio)
        if etr_accuracy_ratio != 0:
            self.ETR_predictions_accuracy.append(etr_accuracy_ratio)
        
        RF_total_accuracy_mean = sum(self.RF_predictions_accuracy) / len(self.RF_predictions_accuracy) if len(self.RF_predictions_accuracy) > 0 else 0
        ABR_total_accuracy_mean = sum(self.ABR_predictions_accuracy) / len(self.ABR_predictions_accuracy) if len(self.ABR_predictions_accuracy) > 0 else 0
        GBM_total_accuracy_mean = sum(self.GBM_predictions_accuracy) / len(self.GBM_predictions_accuracy) if len(self.GBM_predictions_accuracy) > 0 else 0
        ETR_total_accuracy_mean = sum(self.ETR_predictions_accuracy) / len(self.ETR_predictions_accuracy) if len(self.ETR_predictions_accuracy) > 0 else 0
        
        rf_prediction = self.RF_prediction[-1] if np.isnan(self.RF_prediction[0]) else self.RF_prediction[0]
        gbm_prediction = self.GBM_prediction[-1] if np.isnan(self.GBM_prediction[0]) else self.GBM_prediction[0]
        abr_prediction = self.ABR_prediction[-1] if np.isnan(self.ABR_prediction[0]) else self.ABR_prediction[0]
        etr_prediction = self.ETR_prediction[-1] if np.isnan(self.ETR_prediction[0]) else self.ETR_prediction[0]

        pred_diff_rf = ((rf_prediction - self.data0.close[0])/self.data0.close[0]) * 100
        pred_diff_gbm = ((gbm_prediction - self.data0.close[0])/self.data0.close[0]) * 100
        pred_diff_abr = ((abr_prediction - self.data0.close[0])/self.data0.close[0]) * 100
        pred_diff_etr = ((etr_prediction - self.data0.close[0])/self.data0.close[0]) * 100

        lastpred_accuracy_rf = (1 - (abs(self.data0.close[0] - rf_prediction) / rf_prediction)) * 100 if rf_prediction != 0 else 0
        lastpred_accuracy_gbm = (1 - (abs(self.data0.close[0] - gbm_prediction) / gbm_prediction)) * 100 if gbm_prediction != 0 else 0
        lastpred_accuracy_abr = (1 - (abs(self.data0.close[0] - abr_prediction) / abr_prediction)) * 100 if abr_prediction != 0 else 0
        lastpred_accuracy_etr = (1 - (abs(self.data0.close[0] - etr_prediction) / etr_prediction)) * 100 if etr_prediction != 0 else 0

        if not np.isnan(rf_prediction):
            logger.info(f'Prediction RF: {round(rf_prediction, 4)}, Variance: {pred_diff_rf:.2f} %,'
                                    f'Last Prediction Accuracy: {lastpred_accuracy_rf:.2f} %, Total Accuracy: {RF_total_accuracy_mean:.2f} %')
        if not np.isnan(gbm_prediction):
            logger.info(f'Prediction GBM: {round(gbm_prediction, 4)}, Variance: {pred_diff_gbm:.2f} %,'
                                f'Last Prediction Accuracy: {lastpred_accuracy_gbm:.2f} %, Total Accuracy: {GBM_total_accuracy_mean:.2f} %')
        if not np.isnan(abr_prediction):
            logger.info(f'Prediction ABR: {round(abr_prediction, 4)}, Variance: {pred_diff_abr:.2f} %,'
                                    f'Last Prediction Accuracy: {lastpred_accuracy_abr:.2f} %, Total Accuracy: {ABR_total_accuracy_mean:.2f} %')
        if not np.isnan(etr_prediction):
            logger.info(f'Prediction ETR: {round(etr_prediction, 4)}, Variance: {pred_diff_etr:.2f} %,'
                                    f'Last Prediction Accuracy: {lastpred_accuracy_etr:.2f} %, Total Accuracy: {ETR_total_accuracy_mean:.2f} %')
    
    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        
        commission = abs(abs(trade.pnlcomm) - abs(trade.pnl))
        
        self.log('TRADE PROFIT: GROSS:  %.2f, NET:  %.2f, COMM:  %.2f' %
                 (trade.pnl, trade.pnlcomm, commission))
        
        self.past_trades.append(trade.pnlcomm)
    
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            self.order = order
            return

        # Check if an order has been completed
        if order.status in [order.Completed]:
            if order.isbuy():  
                self.long_entry_price = order.executed.price              
                self.current_position_size += order.executed.size
                self.long_position = True                
                self.log(f"BUY EXECUTED, Price: {order.executed.price:.4f}, Cost: {order.executed.value:.2f}, Comm {order.executed.comm:.2f}")

            else:  
                self.short_entry_price = order.executed.price              
                self.current_position_size -= order.executed.size
                self.long_position = False                
                self.log(f"SELL EXECUTED, Price: {order.executed.price:.4f}, Cost: {order.executed.value:.2f}, Comm {order.executed.comm:.2f}")

            if not self.position:  # if position is closed
                closed_size = self.current_position_size                
                if self.long_position:                    
                    profit_loss = (self.short_entry_price - order.executed.price) * closed_size
                    self.log(f"Closed SHORT position, Price: {order.executed.price:.4f},\t ----- PnL = {profit_loss:.2f} $") 
                                     
                else:  # short position
                    profit_loss = (order.executed.price - self.long_entry_price) * closed_size
                    self.log(f"Closed LONG position, Price: {order.executed.price:.4f},\t ----- PnL = {profit_loss:.2f} $")
                self.current_position_size = 0                    
                # self.past_trades.append(profit_loss)

        elif order.status in [order.Canceled, order.Rejected]:
            self.log('Order Canceled/Rejected')            

        elif order.status in [order.Margin]:
            self.log('Order Margin')   
            
        # Reset
        self.order = None

    def next(self): 
        if len(self.data0) < 5:
            return
        
        close0 = self.data0.close[0]
                
        # Check if the data point is behind more than 10 minutes   
        kiev_tz = pytz.timezone('Europe/Kiev')             
        current_time = bt.num2date(self.data0.datetime[0]).replace(tzinfo=pytz.utc)
        local_time = dtime.datetime.now(tz=kiev_tz)
        time_diff = local_time - dtime.timedelta(minutes=10)
        if current_time < time_diff:
            status = 'BACKTEST'
        else:
            status = 'LIVE'

        
        # cash, value = self.broker.get_wallet_balance('USDT') 
        value = self.broker.getvalue()
        cash = self.broker.getcash()

        self.in_position = self.broker.getposition(self.data).size 
        
        if self.in_position > 0:
            self.pos = 'Long'
        elif self.in_position < 0:
            self.pos = 'Short'
        else: 
            self.pos = 'None'
                
        self.log(f'O: {self.data0.open[0]:.4f}, H: {self.data0.high[0]:.4f}, L: {self.data0.low[0]:.4f}, C: {self.data0.close[0]:.4f}, Volume: {self.data0.volume[0]:.0f}    {status}')
        self.log(f'  \t Portfolio value:  {value:.2f} $,  Position:  {self.pos}')
        self.log('**************************'*2)
        
        # DLM predictions
         
        
        
        # self.log(f'Prediction RLM: {self.a2c_prediction[0]}')
         
        
                
        if self.in_position == 0 and \
            (self.check_buy_condition() or self.check_sell_condition()):            
                    
            if self.check_buy_condition():
                price = self.data1.high[0]  
                cash = self.broker.getcash()                                       
                free_money = cash * self.params.trade_coef
                size = self.broker.getcommissioninfo(self.data).getsize(price=price, cash=free_money) #* (kelly_coef)
                self.order = self.buy(size=size, exectype=bt.Order.Market)
                value = self.broker.getvalue()
                

            elif self.check_sell_condition():
                price = self.data1.low[0]
                cash = self.broker.getcash()
                
                free_money = cash * self.params.trade_coef
                size = self.broker.getcommissioninfo(self.data).getsize(price=price, cash=free_money) #* (kelly_coef)
                self.order = self.sell(size=size, exectype=bt.Order.Market)
                value = self.broker.getvalue()
                                
        elif ((self.in_position > 0) and (self.check_sell_condition() or self.check_stop_buy_condition())):            
            self.order = self.close()
            cash = self.broker.getcash()
            
                        
        elif ((self.in_position < 0) and (self.check_buy_condition() or self.check_stop_sell_condition())):            
            self.order = self.close()  
            cash = self.broker.getcash()                            
        
        
        self.log('DrawDown: %.2f' % self.stats.drawdown.drawdown[-1])
        self.log('MaxDrawDown: %.2f' % self.stats.drawdown.maxdrawdown[-1])                
        
        self.values.append(self.broker.getvalue())