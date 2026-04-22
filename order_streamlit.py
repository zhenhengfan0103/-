# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 10:44:20 2026

@author: user
"""

# 載入必要套件
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.dates as mdates
import numpy as np
import time
import streamlit as st


# 下單部位管理物件
class Record():
    def __init__(self):   ## 建構子
        # 儲存績效
        self.Profit = []         # 每筆完整交易的總損益(單位: 每股價差 或 指數點差 的總數量版本)
        self.Profit_rate = []    # 每筆完整交易的報酬率
        # 未平倉
        self.OpenInterestQty = 0
        self.OpenInterest = []   # [side, product, order_time, order_price, qty]
        # 交易紀錄總計
        self.TradeRecord = []    # [B/S, product, entry_time, entry_price, exit_time, exit_price, qty]

    # 進場紀錄
    def Order(self, BS, Product, OrderTime, OrderPrice, OrderQty):
        if OrderQty <= 0:
            return

        qty = int(OrderQty)

        if BS == 'B' or BS == 'Buy':
            self.OpenInterest.append([1, Product, OrderTime, OrderPrice, qty])
            self.OpenInterestQty += qty

        elif BS == 'S' or BS == 'Sell':
            self.OpenInterest.append([-1, Product, OrderTime, OrderPrice, qty])
            self.OpenInterestQty -= qty

    # 出場紀錄(買賣別需與進場相反，多單進場則空單出場)
    def Cover(self, BS, Product, CoverTime, CoverPrice, CoverQty):
        if CoverQty <= 0:
            return

        remain_qty = int(CoverQty)

        # 平多單
        if BS == 'S' or BS == 'Sell':
            while remain_qty > 0:
                long_positions = [x for x in self.OpenInterest if x[0] == 1]
                if len(long_positions) == 0:
                    print('尚無進場')
                    return

                pos = long_positions[0]
                pos_qty = pos[4]
                close_qty = min(remain_qty, pos_qty)

                entry_price = pos[3]
                profit_unit = CoverPrice - entry_price
                profit_total = profit_unit * close_qty
                profit_rate = profit_unit / entry_price if abs(entry_price) > 1e-12 else 0

                # 新增交易紀錄
                self.TradeRecord.append([
                    'B',
                    pos[1],
                    pos[2],
                    pos[3],
                    CoverTime,
                    CoverPrice,
                    close_qty
                ])

                # 紀錄績效
                self.Profit.append(profit_total)
                self.Profit_rate.append(profit_rate)

                # 更新未平倉
                if close_qty == pos_qty:
                    self.OpenInterest.remove(pos)
                else:
                    pos[4] -= close_qty

                self.OpenInterestQty -= close_qty
                remain_qty -= close_qty

        # 平空單
        elif BS == 'B' or BS == 'Buy':
            while remain_qty > 0:
                short_positions = [x for x in self.OpenInterest if x[0] == -1]
                if len(short_positions) == 0:
                    print('尚無進場')
                    return

                pos = short_positions[0]
                pos_qty = pos[4]
                close_qty = min(remain_qty, pos_qty)

                entry_price = pos[3]
                profit_unit = entry_price - CoverPrice
                profit_total = profit_unit * close_qty
                profit_rate = profit_unit / entry_price if abs(entry_price) > 1e-12 else 0

                # 新增交易紀錄
                self.TradeRecord.append([
                    'S',
                    pos[1],
                    pos[2],
                    pos[3],
                    CoverTime,
                    CoverPrice,
                    close_qty
                ])

                # 紀錄績效
                self.Profit.append(profit_total)
                self.Profit_rate.append(profit_rate)

                # 更新未平倉
                if close_qty == pos_qty:
                    self.OpenInterest.remove(pos)
                else:
                    pos[4] -= close_qty

                self.OpenInterestQty += close_qty
                remain_qty -= close_qty

    # 取得當前未平倉量
    def GetOpenInterest(self):
        return self.OpenInterestQty

    # 取得交易紀錄
    def GetTradeRecord(self):
        return self.TradeRecord

    # 取得交易盈虧清單
    def GetProfit(self):
        return self.Profit

    # 取得交易投資報酬率清單
    def GetProfitRate(self):
        return self.Profit_rate

    # 取得交易總盈虧
    def GetTotalProfit(self):
        if len(self.Profit) > 0:
            return sum(self.Profit)
        else:
            return 0

    # 取得交易次數
    def GetTotalNumber(self):
        if len(self.Profit) > 0:
            return len(self.Profit)
        else:
            return 0

    # 取得平均交易盈虧(每次)
    def GetAverageProfit(self):
        if len(self.Profit) > 0:
            return sum(self.Profit) / len(self.Profit)
        else:
            return 0

    # 取得交易平均投資報酬率
    def GetAverageProfitRate(self):
        if len(self.Profit_rate) > 0:
            return sum(self.Profit_rate) / len(self.Profit_rate)
        else:
            return 0

    # 取得勝率
    def GetWinRate(self):
        WinProfit = [i for i in self.Profit if i > 0]
        if len(self.Profit) > 0:
            return len(WinProfit) / len(self.Profit)
        else:
            return 0

    # 最大連續虧損
    def GetAccLoss(self):
        if len(self.Profit) > 0:
            AccLoss = 0
            MaxAccLoss = 0
            for p in self.Profit:
                if p <= 0:
                    AccLoss += p
                    if AccLoss < MaxAccLoss:
                        MaxAccLoss = AccLoss
                else:
                    AccLoss = 0
            return MaxAccLoss
        else:
            return 0

    # 最大累計盈虧回落(MDD)
    def GetMDD(self):
        if len(self.Profit) > 0:
            MDD, Capital, MaxCapital = 0, 0, 0
            for p in self.Profit:
                Capital += p
                MaxCapital = max(MaxCapital, Capital)
                DD = MaxCapital - Capital
                MDD = max(MDD, DD)
            return MDD
        else:
            return 0

    # 最大累計投資報酬率回落(MDD_rate)
    def GetMDD_rate(self):
        if len(self.Profit_rate) > 0:
            MDD_rate, Capital_rate, MaxCapital_rate = 0, 0, 0
            for p in self.Profit_rate:
                Capital_rate += p
                MaxCapital_rate = max(MaxCapital_rate, Capital_rate)
                DD_rate = MaxCapital_rate - Capital_rate
                MDD_rate = max(MDD_rate, DD_rate)
            return MDD_rate
        else:
            return 0

    # 平均獲利(只看獲利的)
    def GetAverEarn(self):
        if len(self.Profit) > 0:
            WinProfit = [i for i in self.Profit if i > 0]
            if len(WinProfit) > 0:
                return sum(WinProfit) / len(WinProfit)
            else:
                return 0
        else:
            return 0

    # 平均虧損(只看虧損的)
    def GetAverLoss(self):
        if len(self.Profit) > 0:
            FailProfit = [i for i in self.Profit if i < 0]
            if len(FailProfit) > 0:
                return sum(FailProfit) / len(FailProfit)
            else:
                return 0
        else:
            return 0

    # 累計盈虧
    def GetCumulativeProfit(self):
        if len(self.Profit) > 0:
            TotalProfit = [0]
            for i in self.Profit:
                TotalProfit.append(TotalProfit[-1] + i)
            return TotalProfit
        else:
            return [0]

    # 累計投資報酬率
    def GetCumulativeProfit_rate(self):
        if len(self.Profit_rate) > 0:
            TotalProfit_rate = [0]
            for i in self.Profit_rate:
                TotalProfit_rate.append(TotalProfit_rate[-1] + i)
            return TotalProfit_rate
        else:
            return [0]

    ## 產出交易績效圖(累計盈虧)
    def GeneratorProfitChart(self, choice='stock', StrategyName='Strategy'):
        matplotlib.rcParams['font.family'] = 'Noto Sans CJK JP'
        matplotlib.rcParams['axes.unicode_minus'] = False

        plt.figure()

        TotalProfit = self.GetCumulativeProfit()

        if choice == 'stock':
            TotalProfit_re = [i * 1000 for i in TotalProfit]
            plt.plot(TotalProfit_re[1:], '-', marker='o', linewidth=1)
        if choice == 'future1':
            TotalProfit_re = [i * 200 for i in TotalProfit]
            plt.plot(TotalProfit_re[1:], '-', marker='o', linewidth=1)
        if choice == 'future2':
            TotalProfit_re = [i * 50 for i in TotalProfit]
            plt.plot(TotalProfit_re[1:], '-', marker='o', linewidth=1)

        plt.title('累計盈虧(元)')
        plt.xlabel('交易編號')
        plt.ylabel('累計盈虧(元)')

        length = len(TotalProfit)
        new_ticks = range(1, length + 1)
        plt.xticks(ticks=range(length), labels=new_ticks)

        st.pyplot(plt)
        plt.close()

    ## 產出交易績效圖(累計投資報酬率)
    def GeneratorProfit_rateChart(self, StrategyName='Strategy'):
        matplotlib.rcParams['font.family'] = 'Noto Sans CJK JP'
        matplotlib.rcParams['axes.unicode_minus'] = False

        plt.figure()

        TotalProfit_rate = self.GetCumulativeProfit_rate()
        plt.plot(TotalProfit_rate[1:], '-', marker='o', linewidth=1)

        plt.title('累計投資報酬率')
        plt.xlabel('交易編號')
        plt.ylabel('累計投資報酬率')

        length = len(TotalProfit_rate)
        new_ticks = range(1, length + 1)
        plt.xticks(ticks=range(length), labels=new_ticks)

        st.pyplot(plt)
        plt.close()