﻿//To use Math
using cAlgo.API;
using System;
using System.Collections.Generic;
//To use Positions.Count() method
using System.Linq;

namespace cAlgo.Robots
{
    [Robot(TimeZone = TimeZones.UTC, AccessRights = AccessRights.None)]
    public class BladeCustomV1 : Robot
    {
        #region User Defined Parameters

        #region Blade parameters
        [Parameter("Maximum open buy position?", Group = "Basic Setup", DefaultValue = 8, MinValue = 0)]
        public int MaxOpenBuy { get; set; }

        [Parameter("Maximum open Sell position?", Group = "Basic Setup", DefaultValue = 8, MinValue = 0)]
        public int MaxOpenSell { get; set; }

        [Parameter("Pip step", Group = "Basic Setup", DefaultValue = 10, MinValue = 1)]
        public int PipStep { get; set; }

        [Parameter("Stop loss pips", Group = "Basic Setup", DefaultValue = 0, MinValue = 0, Step = 10)]
        public double StopLoss { get; set; }

        [Parameter("First order volume", Group = "Basic Setup", DefaultValue = 1000, MinValue = 1, Step = 1)]
        public double FirstVolume { get; set; }

        [Parameter("Max spread allowed to open position", Group = "Basic Setup", DefaultValue = 3.0)]
        public double MaxSpread { get; set; }

        [Parameter("Target profit for each group of trade", Group = "Basic Setup", DefaultValue = 3)]
        public int AverageTakeProfit { get; set; }

        [Parameter("Apply take profit", Group = "Basic Setup", DefaultValue = false)]
        public bool ApplyTakeProfit { get; set; }

        [Parameter("Apply stop loss", Group = "Basic Setup", DefaultValue = false)]
        public bool ApplyStopLoss { get; set; }

        [Parameter("Debug flag, set to No on real account to avoid closing all positions when stoping this cBot", Group = "Advanced Setup", DefaultValue = false)]
        public bool IfCloseAllPositionsOnStop { get; set; }

        [Parameter("Volume exponent", Group = "Advanced Setup", DefaultValue = 1.0, MinValue = -100, MaxValue = 100, Step = 0.1)]
        public double VolumeExponent { get; set; }

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


        [Parameter("Cryptographic PRNG", Group = "Custom", DefaultValue = false)]
        public bool CryptographicPRNG { get; set; }

        [Parameter("Apply Lot Multiplier", Group = "Custom", DefaultValue = true)]
        public bool ApplyLotMultiplier { get; set; }

        #endregion

        private string ThiscBotLabel;
        private DateTime LastBuyTradeTime;
        private DateTime LastSellTradeTime;

        #endregion

        #region Fields

        private static int IDx = 0;
        private Random prng = new Random();
        private System.Security.Cryptography.RNGCryptoServiceProvider prngProvider = new System.Security.Cryptography.RNGCryptoServiceProvider();

        #endregion

        #region cTrader Events

        /// <summary>
        /// cBot initialization. OnStart is called when the agent starts.
        /// </summary>
        protected override void OnStart()
        {

            //Print(Symbol.PipSize);

            // Set position label to cBot name
            //ThiscBotLabel = this.GetType().Name + "-" + (++IDx).ToString();
            ThiscBotLabel = this.GetType().Name + "-" + (prng.Next(0, 18000)).ToString() + "-" + (++IDx).ToString();
            //ThiscBotLabel = this.GetType().Name;

            // Normalize volume in case a wrong volume was entered
            if (FirstVolume != (FirstVolume = Symbol.NormalizeVolumeInUnits(FirstVolume)))
            {
                Print("Volume entered incorrectly, volume has been changed to ", FirstVolume);
            }
        }


        /// <summary>
        /// Error handling. OnError is called when the agent encounters an error.
        /// </summary>
        /// <param name="error"></param>
        protected override void OnError(Error error)
        {
            Print("Error occured, error code: ", error.Code);
        }

        /// <summary>
        /// OnTick is called everytime the price changes for the symbol.
        /// General agent sensing of what to do is performed.
        /// </summary>
        protected override void OnTick()
        {
            var buyPositionLst = Positions.Where(x => x.TradeType == TradeType.Buy && x.SymbolName == SymbolName && x.Label == ThiscBotLabel);
            var sellPositionLst = Positions.Where(x => x.TradeType == TradeType.Sell && x.SymbolName == SymbolName && x.Label == ThiscBotLabel);

            int numBuyPositions = buyPositionLst.Count();
            int numSellPositions = sellPositionLst.Count();

            SmartGrid(buyPositionLst, sellPositionLst, numBuyPositions, numSellPositions);

            // Conditions check before process trade
            if (Symbol.Spread / Symbol.PipSize <= MaxSpread)
            {
                RandomWalk(numBuyPositions, numSellPositions);
            }

            if (!this.IsBacktesting)
            {
                DisplayStatusOnChart();
            }
        }

        /// <summary>
        /// OnStop is called when the agent stops.
        /// </summary>
        protected override void OnStop()
        {
            Chart.RemoveAllObjects();

            // Close all open positions opened by this cBot on stop.
            if (this.IsBacktesting || IfCloseAllPositionsOnStop)
            {
                CloseAllTrades();
            }
        }

        protected override void OnBar()
        {
            if (this.IsBacktesting)
            {
                DisplayPositions();
            }
        }
        #endregion

        #region Helper Methods

        private void SmartGrid(IEnumerable<Position> buyPositionLst, IEnumerable<Position> sellPositionLst, int numBuyPositions, int numSellPositions)
        {
            // Take Profit implementation:
            double targetProfit = FirstVolume * AverageTakeProfit * Symbol.PipSize;

            // Close all winning buy positions if all buy positions' target profit is met
            if (numBuyPositions > 0)
            {
                if (buyPositionLst.Average(x => x.NetProfit) >= targetProfit)
                {
                    CloseTrades(TradeType.Buy);
                }
            }
            // Close all winning sell positions if all sell positions' target profit is met
            if (numSellPositions > 0)
            {
                if (sellPositionLst.Average(x => x.NetProfit) >= targetProfit)
                {
                    CloseTrades(TradeType.Sell);
                }
            }

            // Stop Loss implementation:
            //double targetLoss = FirstVolume * StopLoss * Symbol.PipSize;
            //double targetLoss = FirstVolume * StopLoss;
            //double targetLoss = FirstVolume * StopLoss * Symbol.PipSize;
            //double targetLoss = FirstVolume * StopLoss;

            //// Close all losing buy positions if all buy positions' target loss is met
            //if (numBuyPositions > 0)
            //{
            //    var loss = GetAverageNetLoss(TradeType.Buy);
            //    if (-1*loss >= targetLoss)
            //    {
            //        CloseTrades(TradeType.Buy);
            //    }
            //}
            //// Close all losing sell positions if all sell positions' target loss is met
            //if (numSellPositions > 0)
            //{
            //    var loss = GetAverageNetLoss(TradeType.Sell);
            //    if (-1 * loss >= targetLoss)
            //    {
            //        CloseTrades(TradeType.Sell);
            //    }
            //}


        }

        double GetAverageNetLoss(TradeType tradeType)
        {
            double maxLossNet = 0;
            double maxProfitNet = 0;
            double actualNet = 0;

            double maxLossGross = 0;
            double maxProfitGross = 0;
            double actualGross = 0;

            double toMaxProfitPips = 0;
            double toMaxProfitPipsAvg = 0;
            double ProfitVolume = 0;
            double actualPips = 0;
            double actualPipsAvg = 0;
            double actualVolume = 0;
            double toMaxLossPips = 0;
            double toMaxLossPipsAvg = 0;
            double LossVolume = 0;
            int missingStopLoss = 0;
            int missingTakeProfit = 0;

            int numPositions = 0;

            foreach (var p in Positions)
            {
                if (p.TradeType == tradeType)
                {
                    numPositions++;
                    var s = MarketData.GetSymbol(p.SymbolCode);
                    API.Internals.Symbol mainPair = null;
                    mainPair = MarketData.GetSymbol(p.SymbolCode);

                    actualGross += p.GrossProfit;
                    actualNet += p.NetProfit;
                    actualPips += p.Pips;
                    actualPipsAvg += p.Pips * p.VolumeInUnits;
                    actualVolume += p.VolumeInUnits;

                    double stopLossPips = 0, takeProfitPips = 0;
                    double sign = p.TradeType == TradeType.Buy ? 1 : -1;

                    if (p.StopLoss != null)
                    {
                        stopLossPips = sign * ((double)(p.EntryPrice - p.StopLoss)) / s.PipSize;
                        toMaxLossPips += stopLossPips + p.Pips;
                        toMaxLossPipsAvg += (stopLossPips + p.Pips) * p.VolumeInUnits;
                        LossVolume += p.VolumeInUnits;
                        //var loss = (double)(sign * (p.VolumeInUnits - p.VolumeInUnits * p.EntryPrice / p.StopLoss));
                        //if (mainPair != null)
                        //    loss /= mainPair.Ask;
                        //maxLossGross += loss;
                        //maxLossNet += loss + p.Commissions + p.Swap;
                    }
                    else
                    {
                        missingStopLoss++;
                    }
                    double sl = (p.StopLoss == null ? 1 : p.StopLoss.Value);
                    var loss = (double)(sign * (p.VolumeInUnits - p.VolumeInUnits * p.EntryPrice / sl));
                    if (mainPair != null)
                        loss /= mainPair.Ask;

                    maxLossGross += loss;
                    maxLossNet += loss + p.Commissions + p.Swap;

                    if (p.TakeProfit != null)
                    {
                        takeProfitPips = sign * ((double)(p.TakeProfit - p.EntryPrice)) / s.PipSize;
                        toMaxProfitPips += takeProfitPips - p.Pips;
                        toMaxProfitPipsAvg += (takeProfitPips - p.Pips) * p.VolumeInUnits;
                        ProfitVolume += p.VolumeInUnits;
                        var profit = (double)(sign * (p.VolumeInUnits - p.VolumeInUnits * p.EntryPrice / p.TakeProfit));
                        if (mainPair != null)
                            profit /= mainPair.Ask;
                        maxProfitGross += profit;
                        maxProfitNet += profit + p.Commissions + p.Swap;

                    }
                    else
                    {
                        missingTakeProfit++;
                    }
                }
            }

            double averageNetLoss = maxLossNet / numPositions;

            if (averageNetLoss > 0)
            {
                Print("net loss");
            }

            return averageNetLoss;
        }

        private void RandomWalk(int numBuyPositions, int numSellPositions)
        {
            bool roll;

            if (CryptographicPRNG)
            {
                var byteArray = new byte[1];
                prngProvider.GetBytes(byteArray);

                roll = byteArray[0] >= 128;
            }
            else
            {
                roll = prng.Next(0, 2) == 0;
            }

            if (roll)
            {
                // HEADS
                if (numBuyPositions < MaxOpenBuy)
                {
                    ProcessOrder(TradeType.Buy);
                    //ProcessBuy();
                }
            }
            else
            {
                // TAILS
                if (numSellPositions < MaxOpenSell)
                {
                    ProcessOrder(TradeType.Sell);
                    //ProcessSell();
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

        private void ProcessOrder(TradeType tradeType)
        {
            var positionLst = Positions.Where(x => x.TradeType == tradeType && x.SymbolName == SymbolName && x.Label == ThiscBotLabel);
            int numPositions = positionLst.Count();

            double? stopLossPips = null;
            double? takeProfitPips = null;

            if (StopLoss != 0 && ApplyStopLoss)
            {
                stopLossPips = StopLoss;
            }

            if (AverageTakeProfit != 0 && ApplyTakeProfit)
            {
                takeProfitPips = AverageTakeProfit;
            }

            bool buyProfit = (tradeType == TradeType.Buy) && Bars.ClosePrices.Last(1) > Bars.ClosePrices.Last(2);
            bool sellCheap = (tradeType == TradeType.Sell) && Bars.ClosePrices.Last(2) > Bars.ClosePrices.Last(1);

            if (numPositions == 0 && (buyProfit || sellCheap))
            {
                //ExecuteMarketOrder(tradeType, SymbolName, FirstVolume, ThiscBotLabel, stopLossPips, takeProfitPips);
                ExecuteMarketOrder(tradeType, SymbolName, ExponentialVolume(tradeType, FirstVolume), ThiscBotLabel, stopLossPips, takeProfitPips);

                if (tradeType == TradeType.Buy)
                {
                    LastBuyTradeTime = MarketSeries.OpenTime.Last(0);
                }
                else if (tradeType == TradeType.Sell)
                {
                    LastSellTradeTime = MarketSeries.OpenTime.Last(0);
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
                        LastBuyTradeTime = MarketSeries.OpenTime.Last(0);
                    }
                    else if (tradeType == TradeType.Sell)
                    {
                        LastSellTradeTime = MarketSeries.OpenTime.Last(0);
                    }
                }
            }
        }

        private void ProcessBuy()
        {
            double? stopLossPips = null;
            double? takeProfitPips = null;

            if (StopLoss != 0 && ApplyStopLoss)
            {
                stopLossPips = StopLoss;
            }

            if (AverageTakeProfit != 0 && ApplyTakeProfit)
            {
                takeProfitPips = AverageTakeProfit;
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

            if (StopLoss != 0 && ApplyStopLoss)
            {
                stopLossPips = StopLoss;
            }

            if (AverageTakeProfit != 0 && ApplyTakeProfit)
            {
                takeProfitPips = AverageTakeProfit;
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

        private void DisplayStatusOnChart()
        {
            var buyPositionLst = Positions.Where(x => x.TradeType == TradeType.Buy && x.SymbolName == SymbolName && x.Label == ThiscBotLabel);
            var sellPositionLst = Positions.Where(x => x.TradeType == TradeType.Sell && x.SymbolName == SymbolName && x.Label == ThiscBotLabel);

            int numBuyPositions = buyPositionLst.Count();
            int numSellPositions = sellPositionLst.Count();

            if (numBuyPositions > 1)
            {
                double y = buyPositionLst.Average(x => x.EntryPrice);

                Chart.DrawHorizontalLine("bpoint", y, Color.Yellow, 2, LineStyle.Dots);
            }
            else
            {
                Chart.RemoveObject("bpoint");
            }
            if (numSellPositions > 1)
            {
                double z = sellPositionLst.Average(x => x.EntryPrice);

                Chart.DrawHorizontalLine("spoint", z, Color.HotPink, 2, LineStyle.Dots);
            }
            else
            {
                Chart.RemoveObject("spoint");
            }
            Chart.DrawStaticText("pan", GenerateStatusText(), VerticalAlignment.Top, HorizontalAlignment.Left, Color.Tomato);
        }

        int indexFrame = 0;
        int indexStep = 0;
        bool cleanupPos = false;

        void DisplayPositions()
        {
            double offset = 50 * Symbol.PipSize;
            string bullet = "\u25CF";
            string diamond = "\u2666";

            if (indexFrame >= 10000)
            {
                for (int i = indexStep; i < indexFrame; i++)
                {
                    Chart.RemoveObject("Position " + i.ToString());
                }
                cleanupPos = false;
                indexStep = 0;
                indexFrame = 0;
            }

            if (Positions.Any())
            {
                foreach (var pos in Positions.Where(x => x.SymbolName == SymbolName && x.Label == ThiscBotLabel))
                {
                    if(pos.TradeType == TradeType.Buy)
                    {
                        Chart.DrawText("Position " + (++indexFrame).ToString(), diamond, pos.EntryTime, pos.EntryPrice + offset, Color.LightGreen);
                    }
                    else
                    {
                        Chart.DrawText("Position " + (++indexFrame).ToString(), bullet, pos.EntryTime, pos.EntryPrice + offset, Color.Red);
                    }
                    
                }
                    
                //Chart.DrawText("Position " + (indexFrame).ToString(), bullet, x1, close, Color.Fuchsia);
            }

        }
        private string GenerateStatusText()
        {
            var statusText = "";
            var buyPositions = "";
            var sellPositions = "";
            var spread = "";
            var buyDistance = "";
            var sellDistance = "";

            var buyPositionLst = Positions.Where(x => x.TradeType == TradeType.Buy && x.SymbolName == SymbolName && x.Label == ThiscBotLabel);
            var sellPositionLst = Positions.Where(x => x.TradeType == TradeType.Sell && x.SymbolName == SymbolName && x.Label == ThiscBotLabel);

            int numBuyPositions = buyPositionLst.Count();
            int numSellPositions = sellPositionLst.Count();

            spread = "\nSpread = " + Math.Round(Symbol.Spread / Symbol.PipSize, 1);
            buyPositions = "\nBuy Positions = " + numBuyPositions;
            sellPositions = "\nSell Positions = " + numSellPositions;

            if (numBuyPositions > 0)
            {
                var averageBuyFromCurrent = Math.Round((buyPositionLst.Average(x => x.EntryPrice) - Symbol.Bid) / Symbol.PipSize, 1);
                buyDistance = "\nBuy Target Away = " + averageBuyFromCurrent;
            }
            if (numSellPositions > 0)
            {
                var averageSellFromCurrent = Math.Round((Symbol.Ask - sellPositionLst.Average(x => x.EntryPrice)) / Symbol.PipSize, 1);
                sellDistance = "\nSell Target Away = " + averageSellFromCurrent;
            }
            if (Symbol.Spread / Symbol.PipSize > MaxSpread)
            {
                statusText = "MAX SPREAD EXCEED";
            }
            else
            {
                statusText = ThiscBotLabel + buyPositions + spread + sellPositions + buyDistance + sellDistance;
            }
            return (statusText);
        }

        #endregion
    }
}
