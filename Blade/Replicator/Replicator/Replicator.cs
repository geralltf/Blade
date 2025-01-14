﻿using System;
using System.Collections.Generic;
using System.Linq;
using cAlgo.API;
using cAlgo.API.Indicators;
using cAlgo.API.Internals;
using cAlgo.Indicators;

namespace cAlgo.Robots
{
    [Robot(TimeZone = TimeZones.UTC, AccessRights = AccessRights.None)]
    public class ReplicatorV1 : Robot
    {
        [Parameter("Source")]
        public DataSeries Source { get; set; }
        [Parameter("BandPeriods", DefaultValue = 14)]
        public int BandPeriod { get; set; }
        [Parameter("Std", DefaultValue = 1.8)]
        public double std { get; set; }
        [Parameter("MAType")]
        public MovingAverageType MAType { get; set; }
        [Parameter("Initial Volume Percent", DefaultValue = 1, MinValue = 0)]
        public double InitialVolumePercent { get; set; }
        [Parameter("Stop Loss", DefaultValue = 0)]
        public int StopLoss { get; set; }
        [Parameter("Take Profit", DefaultValue = 25)]
        public int TakeProfit { get; set; }

        [Parameter("First order volume", Group = "Basic Setup", DefaultValue = 1000, MinValue = 1, Step = 1)]
        public double FirstVolume { get; set; }

        [Parameter("Max spread allowed to open position", Group = "Basic Setup", DefaultValue = 3.0)]
        public double MaxSpread { get; set; }

        [Parameter("Pip step", Group = "Basic Setup", DefaultValue = 10, MinValue = 1)]
        public int PipStep { get; set; }

        [Parameter("Maximum open buy position?", Group = "Basic Setup", DefaultValue = 8, MinValue = 0)]
        public int MaxOpenBuy { get; set; }

        [Parameter("Maximum open Sell position?", Group = "Basic Setup", DefaultValue = 8, MinValue = 0)]
        public int MaxOpenSell { get; set; }

        [Parameter("Volume exponent", Group = "Advanced Setup", DefaultValue = 1.0, MinValue = -100, MaxValue = 100, Step = 0.1)]
        public double VolumeExponent { get; set; }

        //[Parameter("Target profit for each group of trade", Group = "Basic Setup", DefaultValue = 3, MinValue = 1)]
        //public int AverageTakeProfit { get; set; }
        [Parameter("Apply take profit", Group = "Basic Setup", DefaultValue = false)]
        public bool ApplyTakeProfit { get; set; }

        [Parameter("Power Factor", Group = "Custom", DefaultValue = 1.0, MinValue = -100, MaxValue = 100.0, Step = 0.1)]
        public double PowerFactor { get; set; }

        [Parameter("Power Offset", Group = "Custom", DefaultValue = 0.0, MinValue = -1000, MaxValue = 1000.0, Step = 0.1)]
        public double PowerOffset { get; set; }

        [Parameter("Equity Multiplier", Group = "Custom", DefaultValue = 1000.0, MinValue = -1000, MaxValue = 10000.0, Step = 0.1)]
        public double EquityMultiplier { get; set; }

        [Parameter("Percent Increase Threshold", Group = "Custom", DefaultValue = 1000.0, MinValue = -100, MaxValue = 10000.0, Step = 0.1)]
        public double PercentIncreaseThreshold { get; set; }

        [Parameter("Risk Percentage", Group = "Custom", DefaultValue = 100, MinValue = 0.01, MaxValue = 100, Step = 0.01)]
        public double RiskPercent { get; set; }

        [Parameter("Apply Lot Multiplier", Group = "Custom", DefaultValue = true)]
        public bool ApplyLotMultiplier { get; set; }

        [Parameter("Apply Random Walk", Group = "Custom", DefaultValue = false)]
        public bool ApplyRandomWalk { get; set; }

        [Parameter("Apply Bollinger Bands", Group = "Custom", DefaultValue = true)]
        public bool ApplyBollingerBands { get; set; }

        private DateTime LastBuyTradeTime;
        private DateTime LastSellTradeTime;
        private BollingerBands boll;
        private string ThiscBotLabel;
        private static int IDx = 0;
        private Random prng = new Random();

        protected override void OnStart()
        {
            ThiscBotLabel = this.GetType().Name + "-" + (prng.Next(0, 18000)).ToString() + "-" + (++IDx).ToString();

            boll = Indicators.BollingerBands(Source, BandPeriod, std, MAType);

            if (FirstVolume != (FirstVolume = Symbol.NormalizeVolumeInUnits(FirstVolume)))
            {
                Print("Volume entered incorrectly, volume has been changed to ", FirstVolume);
            }
        }

        protected override void OnTick()
        {
            SmartGrid();

            if (Symbol.Spread / Symbol.PipSize <= MaxSpread)
            {
                if (ApplyRandomWalk)
                {
                    RandomWalk();
                }
            }
        }

        protected override void OnBar()
        {
            // Conditions check before process trade
            if (Symbol.Spread / Symbol.PipSize <= MaxSpread)
            {
                if (ApplyBollingerBands)
                {
                    BollingerBands();
                }
            }
        }

        void BollingerBands()
        {
            if (Bars.Count() > 2)
            {
                if (Bars.Last(2).Close > boll.Top.Last(2))
                {
                    if (Bars.Last(1).Close > boll.Top.Last(1))
                    {
                        //RandomWalk(); // Buy
                        DoBuy();
                    }
                    else
                    {
                        //RandomWalk(); // Sell
                        DoSell();
                    }
                }
                else if (Bars.Last(2).Close < boll.Bottom.Last(2))
                {
                    if (Bars.Last(1).Close < boll.Bottom.Last(1))
                    {
                        //RandomWalk(); // Sell
                        DoSell();
                    }
                    else
                    {
                        //RandomWalk(); // Buy
                        DoBuy();
                    }
                }
            }
        }

        void RandomWalk()
        {
            if (GetRandomTradeType() == TradeType.Buy)
            {
                // HEADS
                DoBuy();
            }
            else
            {
                // TAILS
                DoSell();
            }
        }

        void DoBuy()
        {
            var buyPositionLst = Positions.Where(x => x.TradeType == TradeType.Buy && x.SymbolName == SymbolName && x.Label == ThiscBotLabel);
            int numBuyPositions = buyPositionLst.Count();

            if (numBuyPositions < MaxOpenBuy)
            {
                //ProcessBuy();
                ProcessOrder(TradeType.Buy);
            }
        }

        void DoSell()
        {
            var sellPositionLst = Positions.Where(x => x.TradeType == TradeType.Sell && x.SymbolName == SymbolName && x.Label == ThiscBotLabel);
            int numSellPositions = sellPositionLst.Count();

            if (numSellPositions < MaxOpenSell)
            {
                //ProcessSell();
                ProcessOrder(TradeType.Sell);
            }
        }

        protected override void OnStop()
        {
            // Put your deinitialization logic here
        }

        TradeType GetRandomTradeType()
        {
            return prng.Next(0, 2) == 0 ? TradeType.Buy : TradeType.Sell;
        }

        private void ProcessOrder(TradeType tradeType)
        {
            var positionLst = Positions.Where(x => x.TradeType == tradeType && x.SymbolName == SymbolName && x.Label == ThiscBotLabel);
            int numPositions = positionLst.Count();

            double? stopLossPips = null;
            double? takeProfitPips = null;

            if (StopLoss != 0)
            {
                stopLossPips = StopLoss;
            }

            if (TakeProfit != 0 && ApplyTakeProfit)
            {
                takeProfitPips = TakeProfit;
            }

            bool buyProfit = (tradeType == TradeType.Buy) && Bars.ClosePrices.Last(1) > Bars.ClosePrices.Last(2);
            bool sellCheap = (tradeType == TradeType.Sell) && Bars.ClosePrices.Last(2) > Bars.ClosePrices.Last(1);

            if (numPositions == 0 && (buyProfit || sellCheap))
            {
                //ExecuteMarketOrder(tradeType, SymbolName, FirstVolume, ThiscBotLabel, stopLossPips, takeProfitPips);
                ExecuteMarketOrder(tradeType, SymbolName, ExponentialVolume(tradeType, FirstVolume), ThiscBotLabel, stopLossPips, takeProfitPips);

                if (tradeType == TradeType.Buy)
                {
                    LastBuyTradeTime = Bars.OpenTimes.Last(0);
                }
                else if (tradeType == TradeType.Sell)
                {
                    LastSellTradeTime = Bars.OpenTimes.Last(0);
                }
            }
            if (numPositions > 0)
            {
                DateTime lastTradeTime = (tradeType == TradeType.Buy) ? LastBuyTradeTime : LastSellTradeTime;

                bool executePipStep = false;
                if (tradeType == TradeType.Buy)
                {
                    if ((Symbol.Ask < positionLst.Min(x => x.EntryPrice) - PipStep * Symbol.PipSize) && lastTradeTime != Bars.OpenTimes.Last(0))
                    {
                        executePipStep = true;
                    }

                }
                else
                {
                    if ((Symbol.Bid > positionLst.Max(x => x.EntryPrice) + PipStep * Symbol.PipSize) && lastTradeTime != Bars.OpenTimes.Last(0))
                    {
                        executePipStep = true;
                    }
                }

                if (executePipStep)
                {
                    ExecuteMarketOrder(tradeType, SymbolName, ExponentialVolume(tradeType, CalculateVolume(tradeType)), ThiscBotLabel, stopLossPips, takeProfitPips);
                    //ExecuteMarketOrder(tradeType, SymbolName, CalculateVolume(tradeType), ThiscBotLabel, stopLossPips, takeProfitPips);

                    if (tradeType == TradeType.Buy)
                    {
                        LastBuyTradeTime = Bars.OpenTimes.Last(0);
                    }
                    else if (tradeType == TradeType.Sell)
                    {
                        LastSellTradeTime = Bars.OpenTimes.Last(0);
                    }
                }
            }
        }

        private void ProcessBuy()
        {
            double? stopLossPips = null;
            double? takeProfitPips = null;

            if (StopLoss != 0)
            {
                stopLossPips = StopLoss;
            }

            if (TakeProfit != 0 && ApplyTakeProfit)
            {
                takeProfitPips = TakeProfit;
            }

            if (Positions.Count(x => x.TradeType == TradeType.Buy && x.SymbolName == SymbolName && x.Label == ThiscBotLabel) == 0 && MarketSeries.Close.Last(1) > MarketSeries.Close.Last(2))
            {
                ExecuteMarketOrder(TradeType.Buy, SymbolName, FirstVolume, ThiscBotLabel, stopLossPips, takeProfitPips);
                LastBuyTradeTime = MarketSeries.OpenTime.Last(0);
            }
            if (Positions.Count(x => x.TradeType == TradeType.Buy && x.SymbolName == SymbolName && x.Label == ThiscBotLabel) > 0)
            {
                if (Symbol.Ask < (Positions.Where(x => x.TradeType == TradeType.Buy && x.SymbolName == SymbolName && x.Label == ThiscBotLabel).Min(x => x.EntryPrice) - PipStep * Symbol.PipSize) && LastBuyTradeTime != MarketSeries.OpenTime.Last(0))
                {
                    ExecuteMarketOrder(TradeType.Buy, SymbolName, ExponentialVolume(TradeType.Buy, CalculateVolume(TradeType.Buy)), ThiscBotLabel, stopLossPips, takeProfitPips);
                    LastBuyTradeTime = MarketSeries.OpenTime.Last(0);
                }
            }
        }

        private void ProcessSell()
        {
            double? stopLossPips = null;
            double? takeProfitPips = null;

            if (StopLoss != 0)
            {
                stopLossPips = StopLoss;
            }

            if (TakeProfit != 0 && ApplyTakeProfit)
            {
                takeProfitPips = TakeProfit;
            }

            if (Positions.Count(x => x.TradeType == TradeType.Sell && x.SymbolName == SymbolName && x.Label == ThiscBotLabel) == 0 && MarketSeries.Close.Last(2) > MarketSeries.Close.Last(1))
            {
                ExecuteMarketOrder(TradeType.Sell, SymbolName, FirstVolume, ThiscBotLabel, stopLossPips, takeProfitPips);
                LastSellTradeTime = MarketSeries.OpenTime.Last(0);
            }
            if (Positions.Count(x => x.TradeType == TradeType.Sell && x.SymbolName == SymbolName && x.Label == ThiscBotLabel) > 0)
            {
                if (Symbol.Bid > (Positions.Where(x => x.TradeType == TradeType.Sell && x.SymbolName == SymbolName && x.Label == ThiscBotLabel).Max(x => x.EntryPrice) + PipStep * Symbol.PipSize) && LastSellTradeTime != MarketSeries.OpenTime.Last(0))
                {
                    ExecuteMarketOrder(TradeType.Sell, SymbolName, ExponentialVolume(TradeType.Sell, CalculateVolume(TradeType.Sell)), ThiscBotLabel, stopLossPips, takeProfitPips);
                    LastSellTradeTime = MarketSeries.OpenTime.Last(0);
                }
            }
        }

        private void SmartGrid()
        {
            double targetProfit = FirstVolume * TakeProfit * Symbol.PipSize;

            var buyPositionLst = Positions.Where(x => x.TradeType == TradeType.Buy && x.SymbolName == SymbolName && x.Label == ThiscBotLabel);
            var sellPositionLst = Positions.Where(x => x.TradeType == TradeType.Sell && x.SymbolName == SymbolName && x.Label == ThiscBotLabel);

            int numBuyPositions = buyPositionLst.Count();
            int numSellPositions = sellPositionLst.Count();

            // Close all buy positions if all buy positions' target profit is met
            if (numBuyPositions > 0)
            {
                if (buyPositionLst.Average(x => x.NetProfit) >= targetProfit)
                {
                    CloseTrades(TradeType.Buy);
                }
            }
            // Close all sell positions if all sell positions' target profit is met
            if (numSellPositions > 0)
            {
                if (sellPositionLst.Average(x => x.NetProfit) >= targetProfit)
                {
                    CloseTrades(TradeType.Sell);
                }
            }
        }

        private void CloseTrades(TradeType tradeType)
        {
            // Close all positions for the specified trade type.
            foreach (var position in Positions)
            {
                if (position.TradeType == tradeType && position.SymbolName == SymbolName && position.Label == ThiscBotLabel)
                {
                    ClosePosition(position);
                }
            }
        }

        private void CloseAllTrades()
        {
            // Close all buy and sell positions.
            foreach (var position in Positions)
            {
                if (position.SymbolName == SymbolName && position.Label == ThiscBotLabel)
                {
                    ClosePosition(position);
                }
            }
        }

        private double ExponentialVolume(TradeType tradeType, double volume)
        {
            int numPositions = Positions.Count(x => x.TradeType == tradeType && x.SymbolName == SymbolName && x.Label == ThiscBotLabel);

            double risk = RiskPercent / 100.0;

            double lotMultiplier = 0;

            if (ApplyLotMultiplier)
            {
                lotMultiplier = (long)Math.Ceiling((Account.Equity / PercentIncreaseThreshold) * risk) * EquityMultiplier;

                if (lotMultiplier < 0)
                {
                    lotMultiplier = 0;
                }
            }

            //double v = Account.Equity * risk / (Symbol.PipValue);

            volume = (lotMultiplier + FirstVolume) * Math.Pow(VolumeExponent, PowerOffset + (PowerFactor * numPositions));

            return Symbol.NormalizeVolumeInUnits(volume);

        }

        private double CalculateVolume(TradeType tradeType)
        {
            int numPositions = Positions.Count(x => x.TradeType == tradeType && x.SymbolName == SymbolName && x.Label == ThiscBotLabel);

            return Symbol.NormalizeVolumeInUnits(FirstVolume * Math.Pow(VolumeExponent, numPositions));
        }
    }
}
