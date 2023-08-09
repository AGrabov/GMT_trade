# gmt30_strat_1data1.py
# MFI doble limit line (upper and lower)

import asyncio
import datetime as dtime
from collections import defaultdict, deque

import backtrader as bt
import numpy as np
import pytz
from tabulate import tabulate

import api_config
from indicators.heikin_patterns_01 import HeikinPatterns
from indicators.movavs import MovAverages
from indicators.pivot_point import MyPivotPoint
from indicators.ta_patterns_02 import TaPatterns
from indicators.a2c_prediction import A2CPredictor


class HeikinAshiStrategy(bt.Strategy):        
    params = {
        'fast_ema': 14,                
        'slow_ema': 21,
        'hma_length': 5,                                      # [14, 21, 5, 13, 33, 6, 26, 24, 86, 61, 3, 10, 21, 15, 9, 42, 34, 35, 4, 2, 5, 3, 11, 0, 1, 3, 14, 7, 3]
        'atr_period': 13,                                           
        'atr_threshold': 33,
        'dmi_length': 6,        
        'dmi_threshold': 26,                                  # 5 min [5, 21, 22, 4, 24, 8, 69, 10, 5, 69, 5, 10, 31, 12, 4, 25, 31, 74, 24, 3, 8, 4, 19, 8, 5, 8, 7, 12, 0] 
        'cmo_period': 24,
        'cmo_threshold': 86,
        'volume_factor_perc': 61,  # Volume must be 2 times the average volume
        'ta_threshold': 3,
        'mfi_period': 10,
        'mfi_level': 21,
        'mfi_smooth': 15,
        'sl_percent': 9,  
        'kama_period': 42, 
        'dma_period': 34, 
        'dma_gainlimit': 35,
        'dma_hperiod': 4,
        'fast_ad': 2,
        'slow_ad': 5, 
        'fastk_period': 3,
        'fastd_period': 11,
        'fastd_matype': 0,      # SMA, EMA, WMA, DEMA, TEMA, TRIMA, KAMA, MAMA, T3
        'mama_fastlimit': 1,
        'mama_slowlimit': 3,
        'apo_fast': 7,
        'apo_slow': 14, 
        'apo_matype': 3,        # SMA, EMA, WMA, DEMA, TEMA, TRIMA, KAMA, MAMA, T3
        
        'aroon_period': 10,
        'sma_length': 10,
        
        'trade_coef': 0.5,
        'use_kelly': False,
        'num_past_trades': 10,
    }

    def log(self, txt, dt=None):
        ''' Logging function fot this strategy'''
        kiev_tz = pytz.timezone('Europe/Kiev')
        dt = dt or self.data1.datetime[0]
        if isinstance(dt, float):
            dt = bt.num2date(dt, tz=kiev_tz)
        print('%s, %s' % (dt.isoformat(sep=' '), txt))

    def __init__(self):
        
        self.data0 = self.datas[0]
        self.data1 = self.datas[1]

        self.values = []        
        self.current_position_size = 0
        self.long_position = False
        self.long_entry_price = 0
        self.short_entry_price = 0 
        self.in_position = 0       
        self.kelly_coef = 0        
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

        
        # Heikin Ashi Patterns
        self.patterns = HeikinPatterns(self.data1, plot=False)
        # self.patterns.plotlines.signal._plotskip=True
        # self.patterns.plotlines.stop_signal._plotskip=True

        self.pattern_buy1 = (self.patterns.lines.signal == 1)
        self.pattern_sell1 = (self.patterns.lines.signal == -1)
        self.pattern_stopBuy1 = (self.patterns.lines.stop_signal == 1)
        self.pattern_stopSell1 = (self.patterns.lines.stop_signal == -1)

        # TA-lib Patterns
        self.ta_patterns = TaPatterns(self.data0, 
                                      threshold=self.params.ta_threshold*100, 
                                      plot=False)

        # Moving averages
        self.movav = MovAverages(self.data1.close, 
                                 fast_ema=self.params.fast_ema, 
                                 slow_ema=self.params.slow_ema, 
                                 hma_length=self.params.hma_length,
                                 kama_period=self.params.kama_period,
                                 dma_period=self.params.dma_period,
                                 dma_gainlimit=self.params.dma_gainlimit,
                                 dma_hperiod=self.params.dma_hperiod, 
                                 plot=False
                                 )
        # self.movav.plotlines.signal._plotskip=True
        # self.movav.plotlines.stop_signal._plotskip=True
        self.mas_buy = (self.movav.lines.signal == 1)
        self.mas_sell = (self.movav.lines.signal == -1)
        self.mas_stop_buy = (self.movav.lines.stop_signal == 1)
        self.mas_stop_sell = (self.movav.lines.stop_signal == -1)        

        # # Hull MA Oscillator
        self.hmo = bt.indicators.HullMovingAverageOscillator(self.data1.close, 
                                                             period=self.params.hma_length, 
                                                             plot=False)
        self.hmo_buy1 = (self.hmo.hma > 0.0)
        self.hmo_sell1 = (self.hmo.hma < 0.0)

        # Dicson Moving Average Oscillator
        self.dma_osc = bt.indicators.DMAOscillator(self.data1.close,
                                                    period=self.params.dma_period,
                                                    gainlimit =self.params.dma_gainlimit,
                                                    hperiod = self.params.dma_hperiod, 
                                                    plot=False
                                                    )
        self.dma_osc_buy = (self.dma_osc > 0.0)
        self.dma_osc_sell = (self.dma_osc < 0.0)        

        # KAMA Oscillator
        self.kama_osc = bt.indicators.KAMAOsc(self.data1.close,
                                            period=self.params.kama_period,
                                            fast=5,
                                            slow=15, plot=False)        
        self.kama_osc_buy1 = (self.kama_osc > 0.0)
        self.kama_osc_sell1 = (self.kama_osc < 0.0)        
               
        # SMA
        self.simple_ma = bt.talib.SMA(self.data1.close, timeperiod=self.params.sma_length*10)
        self.sma_cross = bt.indicators.CrossOver(self.ha.lines.ha_close, self.simple_ma, plot=False)
        self.sma_buy = (self.sma_cross == 1)
        self.sma_sell = (self.sma_cross == -1)               

        # MAMA
        self.mama = bt.talib.MAMA(self.ha.lines.ha_close, 
                                  fastlimit=(self.params.mama_fastlimit/10), 
                                  slowlimit=(self.params.mama_slowlimit/100),
                                  )
        self.mama_cross = bt.indicators.CrossOver(self.mama.mama, self.mama.fama, plot=False)
        self.mama_cross.plotinfo.plotname = 'mama_cross'
        self.mama_buy = (self.mama_cross > 0)
        self.mama_sell = (self.mama_cross < 0)
        
        # Absolute Price Oscillator
        self.apo = bt.talib.APO(self.ha.lines.ha_close,
                                fastperiod=self.params.apo_fast,
                                slowperiod=self.params.apo_slow,
                                matype=self.params.apo_matype)
        self.apo.plotinfo.plothlines=[0.0]
        self.apo_buy = (self.apo > 0)
        self.apo_sell = (self.apo < 0)
        
        # Chaikin A/D Oscillator
        self.ad_osc = bt.talib.ADOSC(self.data1.high, 
                                     self.data1.low, 
                                     self.data1.close, 
                                     self.data1.volume, 
                                     fastperiod=self.params.fast_ad,
                                     slowperiod=self.params.slow_ad, 
                                     plot=False)       
        self.ad_osc_buy1 = (self.ad_osc > 0.0)
        self.ad_osc_sell1 = (self.ad_osc < 0.0)
        
        # Stochastic oscillator
        self.stoch = bt.talib.STOCHF(self.ha.lines.ha_high, 
                                    self.ha.lines.ha_low, 
                                    self.ha.lines.ha_close, 
                                    fastk_period=self.params.fastk_period, 
                                    fastd_period=self.params.fastd_period, 
                                    fastd_matype=self.params.fastd_matype,
                                    plot=False
                                    )
        self.stoch_k = self.stoch.lines.fastk
        self.stoch_d = self.stoch.lines.fastd
        self.stoch_buy1 = (self.stoch_k > self.stoch_d)                                 
        self.stoch_sell1 = (self.stoch_k < self.stoch_d)

        # CMO
        self.cmo = bt.talib.CMO(self.ha, timeperiod=self.params.cmo_period)
        self.cmo_buy = self.cmo.real > (self.params.cmo_threshold)
        self.cmo_sell = self.cmo.real < -(self.params.cmo_threshold)

        # ATR 
        self.atr = bt.talib.NATR(self.data1.high, 
                                 self.data1.low, 
                                 self.data1.close, 
                                 timeperiod=self.params.atr_period,                                   
                                 plot=False) 
        self.atr_treshhold = (self.params.atr_threshold / 100)
        self.atr.plotinfo.plotyhlines=[0.5, self.atr_treshhold, 1.5]
        self.high_volatility = (self.atr.real  >  self.atr_treshhold)

        # ADX
        self.dmi = bt.indicators.DirectionalMovementIndex(self.data1, 
                                                          period=self.params.dmi_length,  
                                                          plot=False)        
        self.adx = bt.talib.ADXR(self.data1.high, 
                                 self.data1.low, 
                                 self.data1.close, 
                                 timeperiod=self.params.dmi_length,  
                                 plot=False)        
        self.adx_buy = bt.And((self.dmi.DIplus > self.dmi.DIminus), (self.adx.real > self.params.dmi_threshold))
        self.adx_sell = bt.And((self.dmi.DIplus < self.dmi.DIminus), (self.adx.real > self.params.dmi_threshold))

        # Volume filter
        self.volume_averages = (bt.indicators.SMA(self.data1.volume, 
                                                  period=self.params.slow_ema, 
                                                  plot=False) * 
                                (self.params.volume_factor_perc / 10))
        self.volume_filter = (self.data1.volume > self.volume_averages)
        
        # MFI
        self.mfi = bt.talib.MFI(self.ha.lines.ha_high, 
                                self.ha.lines.ha_low, 
                                self.ha.lines.ha_close, 
                                self.data1.volume,
                                timeperiod=self.params.mfi_period)
         
        self.mfi_upper = (100 - self.params.mfi_level)
        self.mfi_lower = self.params.mfi_level
        self.mfi.plotinfo.plothlines = [self.mfi_upper, self.mfi_lower]
        self.mfi_smoothed = bt.indicators.MovingAverageSimple(self.mfi, period=self.params.mfi_smooth)
        self.mfi_smoothed.plotinfo.plotname = 'MFI smoothed'
        self.mfi_cross = bt.ind.CrossOver(self.mfi, self.mfi_smoothed, plot=False)

        self.mfi_buy = bt.Or((self.mfi_cross == 1), (self.mfi < self.mfi_lower))
        self.mfi_sell = bt.Or((self.mfi_cross == -1), (self.mfi > self.mfi_upper))       

        # Pivot Point levels
        self.pivot = MyPivotPoint(self.data1)
        # self.p, self.r1, self.r2, self.s1, self.s2 = self.pivot.lines

        # A2C Predictor
        self.a2c_pred = A2CPredictor(self.data1)

        self.order = None        
        self.entry_price = None        
    
    def adx_growing(self):
        return (self.adx.real[-1] < self.adx.real[0]) and \
                (self.adx.real[-2] <= self.adx.real[-1]) #and \
                # (self.adx.real[-3] >= self.adx.real[-2])
        
    def check_buy_condition(self):        
        condition1 = (self.mas_buy[0] or self.pivot.buy() or self.mama_buy[0] or self.sma_buy[0]) and self.apo_buy[0] #or self.pattern_buy())
        condition2 = (self.cmo_buy[0] or self.adx_buy[0]) and self.mfi_buy[0] and self.dma_osc_buy[0]
        condition3 = self.high_volatility[0] and (self.ad_osc_buy1[0] or self.volume_filter[0]) and self.hmo_buy1[0] #   # 
        condition4 = (self.stoch_buy1[0] or self.stoch_buy1[-1])
        check_buy_condition = (condition1 and condition2 and condition3 and condition4)
        return check_buy_condition

    def check_sell_condition(self):        
        condition1 = (self.mas_sell[0] or self.pivot.sell() or self.mama_sell[0] or self.sma_sell[0]) and self.apo_sell[0]# or self.pattern_sell())
        condition2 = (self.cmo_sell[0] or self.adx_sell[0]) and self.mfi_sell[0] and self.dma_osc_sell[0]
        condition3 = self.high_volatility[0] and (self.ad_osc_sell1[0] or self.volume_filter[0]) and self.hmo_sell1[0]
        condition4 = (self.stoch_sell1[0] or self.stoch_sell1[-1])
        check_sell_condition = (condition1 and condition2 and condition3 and condition4)
        return check_sell_condition

    def check_stop_buy_condition(self):        
        condition1 = (self.pattern_stopBuy1[0] or self.pattern_sell1[0] or self.mas_stop_buy[0] or self.mama_sell[0]) #  or self.pivot_stop_buy())
        condition2 = self.mfi_sell[0] and (self.ad_osc_sell1[0] or self.volume_filter[0])
        condition3 = ((self.dmi.DIminus[-1] < self.dmi.DIminus[0]) and self.adx_growing()) or \
                      (self.stoch_sell1[0] or self.stoch_sell1[-1])
        check_stop_buy_condition = (condition1 and condition2 and condition3) or self.pivot.sell() or self.sma_sell[0]
        return check_stop_buy_condition and not self.check_buy_condition()
    
    def check_stop_sell_condition(self):
        condition1 = (self.pattern_stopSell1[0] or self.pattern_buy1[0] or self.mas_stop_sell[0] or self.mama_buy[0]) #  or self.pivot_stop_sell())
        condition2 = self.mfi_buy[0] and (self.ad_osc_sell1[0] or self.volume_filter[0])
        condition3 = ((self.dmi.DIplus[0] > self.dmi.DIplus[-1]) and self.adx_growing()) or \
                      (self.stoch_buy1[0] or self.stoch_buy1[-1])
        check_stop_sell_condition = (condition1 and condition2 and condition3) or self.pivot.buy() or self.sma_buy[0]
        return check_stop_sell_condition and not self.check_sell_condition()

    # def start(self):
    #     self.daily_value = self.broker.getvalue(self.data1)  # Store the initial portfolio value
    #     self.start_value = self.broker.getvalue(self.data1)  # Store the initial portfolio value
    
    # def notify_data1(self, data1, status, *args, **kwargs):
    #     dn = data1._name
    #     dt = dtime.datetime.now()
    #     msg= 'Data1 Status: {}'.format(data1._getstatusname(status))
    #     print(dt,dn,msg)
    #     # asyncio.run(self.notifier.send_message(f"{dt.isoformat(sep=' ')}\n"
    #     #                                        f"{dn}\n"
    #     #                                        f"{msg}"))
    #     if data1._getstatusname(status) == 'LIVE':            
    #         self.live_data1 = True
    #         # print(f"{dt} {dn} Data1 is live.")                       
            
    #     else:
    #         self.live_data1 = False
    #         # print(f"{dt} {dn} Data1 is not live.")

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        
        commission = abs(abs(trade.pnlcomm) - abs(trade.pnl))
        
        self.log('OPERATION PROFIT: GROSS:  %.2f, NET:  %.2f, COMM:  %.2f' %
                 (trade.pnl, trade.pnlcomm, commission))
        
        self.past_trades.append(trade.pnlcomm)
    
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            self.order = order
            return

        # Check if an order has been completed
        if order.status in [order.Completed]:
            if order.isbuy():                
                self.current_position_size += order.executed.size
                self.long_position = True
                self.long_entry_price = order.executed.price
                self.log(f"BUY EXECUTED, Price: {order.executed.price:.4f}, Cost: {order.executed.value:.2f}, Comm {order.executed.comm:.2f}")

            elif order.issell():                
                self.current_position_size -= order.executed.size
                self.long_position = False
                self.short_entry_price = order.executed.price
                self.log(f"SELL EXECUTED, Price: {order.executed.price:.4f}, Cost: {order.executed.value:.2f}, Comm {order.executed.comm:.2f}")

            if not self.position:  # if position is closed
                closed_size = self.current_position_size
                self.current_position_size = 0
                # self.in_position = False
                if self.long_position:
                    profit_loss = (self.short_entry_price - order.executed.price) * closed_size
                    self.log(f"Closed SHORT position, Price: {order.executed.price:.4f},\t ----- PnL = {profit_loss:.2f} $") 
                                     
                else:  # long position closed
                    profit_loss = (order.executed.price - self.long_entry_price) * closed_size
                    self.log(f"Closed LONG position, Price: {order.executed.price:.4f},\t ----- PnL = {profit_loss:.2f} $")
                # self.long_position = None                    
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
        # print(self.data1close[0], self.data1.close[0])

        # Define your timezone
        kiev_tz = pytz.timezone('Europe/Kiev')

        
        dt_naive = self.data0.datetime.datetime() # Get the current datetime       
        dt_aware = kiev_tz.localize(dt_naive)  # Convert it to a timezone aware datetime object
        current_time = bt.num2date(self.data0.datetime[0]).replace(tzinfo=pytz.utc) # .astimezone(tz=kiev_tz) # get the timestamp of the current data10 point
        local_time = dtime.datetime.now(tz=kiev_tz)
        time_diff = local_time - dtime.timedelta(minutes=10)
        status = 'HIST' if current_time < time_diff else 'LIVE'
        
        # Get the current portfolio value 
        value = self.broker.getvalue()
        cash = self.broker.getcash()

        self.in_position = self.broker.getposition(data=self.data1).size 
        if self.in_position != 0:            
            self.pos = 'Long' if self.in_position > 0 else 'Short'            
        else: 
            self.pos = 'None'

        prediction = self.a2c_pred.prediction[0]
        print(f'Prediction: {prediction}')        
        # self.log(f'O: {self.data1.open[0]:.4f}, H: {self.data1.high[0]:.4f}, L: {self.data1.low[0]:.4f}, C: {self.data1.close[0]:.4f}, Volume: {self.data1.volume[0]:.0f}    {status}')
        # self.log(f'  \t Portfolio value:  {value:.2f} $,  Position:  {self.pos}')
        # self.log('**************************'*2)

        if self.data.haslivedata:
            print('**************************'*3)
            # print()
            for data in self.datas:
                if len(data) > 0:                    
                    print('{}| {} {} | O: {:.4f} H: {:.4f} L: {:.4f} C: {:.4f} V: {:.0f} | {:.2f} | {} '.format(data.datetime.datetime(),
                                                                                        data._name, status, data.open[0], data.high[0], data.low[0], data.close[0], data.volume[0], value, self.pos))
            # print()
            

        # if self.r1[0] != self.r1[-1]:          
        #     print(f"{bt.num2date(self.data11.datetime[0])}, HAclose: {self.ha.lines.ha_close[0]:.4f} ====,       R2: {self.r2[0]:.4f},   R1: {self.r1[0]:.4f},    P: {self.pp[0]:.4f},    S1: {self.s1[0]:.4f},    S2: {self.s2[0]:.4f}")
        
        
        price = self.data1[0]
        if price == 0:           
            return         
        
        # Calculate the size for the trade using the Kelly Criterion
        wins = [trade for trade in self.past_trades if trade > 0]  # Wins are trades with positive profit
        losses = [trade for trade in self.past_trades if trade < 0]  # Losses are trades with negative profit
        average_win = sum(wins) / len(wins) if wins else 0.0
        average_loss = abs(sum(losses)) / abs(len(losses)) if losses else 0.0
        win_rate = len(wins) / len(self.past_trades) if self.past_trades else 0.0
        win_loss_ratio1 = average_win / average_loss if average_loss != 0 else 1.0
        # win_rate = len(wins) / len(self.past_trades) if self.past_trades else 0.0
        # win_loss_ratio2 = abs(sum(wins)) / abs(sum(losses)) if losses else float('inf')
        win_loss_ratio = win_loss_ratio1 #+ win_loss_ratio2) / 2        
        if len(self.past_trades) == self.params.num_past_trades:                   
            kelly_coef = (win_rate - ((1 - win_rate) / win_loss_ratio)) if win_loss_ratio != 0.0 else 0.5
        else:
            kelly_coef = 0.5
        if kelly_coef < 0:
            kelly_coef = 0.0
        self.log(f" Kelly Criterion: {kelly_coef}")

        if self.order:
            print('Order is pending...')
            return
        
        # if self.in_position == 0 and \
        if not self.position and (self.check_buy_condition() or self.check_sell_condition()):                    
            if self.check_buy_condition():
                price = self.data1.high[0]                 
                if self.params.use_kelly:
                    trade_amount1 = self.params.trade_coef / 2
                    free_money = cash * (trade_amount1 + (trade_amount1 * kelly_coef))
                else:                        
                    free_money = cash * self.params.trade_coef

                size = self.broker.getcommissioninfo(data=self.data1).getsize(price=price, cash=free_money) #* (kelly_coef)
                self.order = self.buy(data=self.data1, size=size, exectype=bt.Order.Market)
                print()
                print("-" * 50)                
                print(f"{bt.num2date(self.data1.datetime[0], tz=kiev_tz)}\t ---- LONG ---- size = {size:.2f} at price = {price:.4f} //// ----- value: {value:.2f} $")                
                if self.params.use_kelly:               
                    print(f"win rate: {win_rate:.2f}, win loss ratio: {win_loss_ratio:.2f},\t Kelly coef: {kelly_coef:.2f}")
                

            elif self.check_sell_condition():
                price = self.data1.low[0]                
                if self.params.use_kelly:
                    trade_amount1 = self.params.trade_coef / 2
                    free_money = cash * (trade_amount1 + (trade_amount1 * kelly_coef))
                else:
                    free_money = cash * self.params.trade_coef
                size = self.broker.getcommissioninfo(data=self.data1).getsize(price=price, cash=free_money) #* (kelly_coef)
                self.order = self.sell(data=self.data1, size=size, exectype=bt.Order.Market)
                print()
                print("-" * 50)
                print(f"{bt.num2date(self.data1.datetime[0], tz=kiev_tz)}\t ---- SHORT ---- size = {size:.2f} at price = {price:.4f} //// ----- Value: {value:.2f} $")
                
                if self.params.use_kelly:
                    print(f"win rate: {win_rate:.2f}, win\loss ratio: {win_loss_ratio:.2f},\t Kelly coef: {kelly_coef:.2f}")

        if self.position:        
            if ((self.in_position > 0) and (self.check_sell_condition() or self.check_stop_buy_condition())):            
                self.order = self.close(data=self.data1)            
                msg = f"{bt.num2date(self.data1.datetime[0], tz=kiev_tz)}  CLOSED     ---- LONG ----  ////  \t  CASH: {cash:.2f} $"
                print(msg)
                            
            elif ((self.in_position < 0) and (self.check_buy_condition() or self.check_stop_sell_condition())):            
                self.order = self.close(data=self.data1)            
                msg = f"{bt.num2date(self.data1.datetime[0], tz=kiev_tz)}  CLOSED     ---- SHORT ----  ////  \t  CASH: {cash:.2f} $"
                print(msg)                        
        
            current_price = self.data1.close[0]
            if self.params.sl_percent != 0:
                sl_value = (self.params.sl_percent / 100)
                if ((self.in_position > 0) and  (current_price <= (self.long_entry_price * (1 - sl_value)))):                
                    self.order = self.close(data=self.data1)
                    self.log(f"Price: {current_price:.4f}, SL %: {self.params.sl_percent:.2f} \t !!! STOP-LOSS !!!")
                elif ((self.in_position < 0) and  (current_price >= (self.short_entry_price * (1 + sl_value)))):
                    self.order = self.close(data=self.data1)
                    self.log(f"Price: {current_price:.4f}, SL %: {self.params.sl_percent:.2f} \t !!! STOP-LOSS !!!")                
                                                            
            # print("-" * 50)
            # self.log('DrawDown: %.2f' % self.stats.drawdown.drawdown[-1])
            # self.log('MaxDrawDown: %.2f' % self.stats.drawdown.maxdrawdown[-1])

        self.values.append(self.broker.getvalue())


    
    
