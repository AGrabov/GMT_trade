# heikin_ashi.py

# import backtrader as bt
# import talib


# # Heikin Ashi Indicator
# class HeikinAshi(bt.Indicator):
#     lines = ('ha_open', 'ha_high', 'ha_low', 'ha_close')

#     def __init__(self):
#         if len(self.data) < 2:
#             raise ValueError("Not enough data to compute Heikin Ashi indicators")

#         self.lines.ha_close = (self.data.close + self.data.open + self.data.high + self.data.low) / 4
#         self.lines.ha_open = (self.data.open(-1) + self.data.close(-1)) / 2
#         self.lines.ha_high = bt.Max(self.data.high, self.lines.ha_open, self.lines.ha_close)
#         self.lines.ha_low = bt.Min(self.data.low, self.lines.ha_open, self.lines.ha_close)