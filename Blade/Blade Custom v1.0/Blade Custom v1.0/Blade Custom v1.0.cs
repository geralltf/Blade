//To use Math
using System;
//To use Positions.Count() method
using System.Linq;
using cAlgo.API;

namespace cAlgo.Robots
{
    [Robot(TimeZone = TimeZones.UTC, AccessRights = AccessRights.None)]
    public class BladeCustomV1 : Robot
    {
        #region User Defined Parameters

        #region Smart Grid parameters
        [Parameter("Maximum open buy position?", Group = "Basic Setup", DefaultValue = 8, MinValue = 0)]
        public int MaxOpenBuy { get; set; }

        [Parameter("Maximum open Sell position?", Group = "Basic Setup", DefaultValue = 8, MinValue = 0)]
        public int MaxOpenSell { get; set; }

        [Parameter("Pip step", Group = "Basic Setup", DefaultValue = 10, MinValue = 1)]
        public int PipStep { get; set; }

        [Parameter("Stop loss pips", Group = "Basic Setup", DefaultValue = 100, MinValue = 10, Step = 10)]
        public double StopLossPips { get; set; }

        [Parameter("First order volume", Group = "Basic Setup", DefaultValue = 1000, MinValue = 1, Step = 1)]
        public double FirstVolume { get; set; }

        [Parameter("Max spread allowed to open position", Group = "Basic Setup", DefaultValue = 3.0)]
        public double MaxSpread { get; set; }

        [Parameter("Target profit for each group of trade", Group = "Basic Setup", DefaultValue = 3, MinValue = 1)]
        public int AverageTakeProfit { get; set; }

        [Parameter("Debug flag, set to No on real account to avoid closing all positions when stoping this cBot", Group = "Advanced Setup", DefaultValue = false)]
        public bool IfCloseAllPositionsOnStop { get; set; }

        [Parameter("Volume exponent", Group = "Advanced Setup", DefaultValue = 1.0, MinValue = 0.1, MaxValue = 5.0)]
        public double VolumeExponent { get; set; }
        #endregion

        #region Martingale parameters

        [Parameter("Take Profit", Group = "Protection", DefaultValue = 40)]
        public int TakeProfit { get; set; }
        #endregion

        private string ThiscBotLabel;
        private DateTime LastBuyTradeTime;
        private DateTime LastSellTradeTime;

        #endregion

        #region cTrader Events

        private static int IDx = 0;
        private static Random prng = new Random();

        /// <summary>
        /// cBot initialization. OnStart is called when the agent starts.
        /// </summary>
        protected override void OnStart()
        {
            // Set position label to cBot name
            //ThiscBotLabel = this.GetType().Name + "-" + (++IDx).ToString();
            ThiscBotLabel = this.GetType().Name +"-" + (prng.Next(0, 18000)).ToString() + "-" + (++IDx).ToString();
            //ThiscBotLabel = this.GetType().Name;

            // Normalize volume in case a wrong volume was entered
            if (FirstVolume != (FirstVolume = Symbol.NormalizeVolumeInUnits(FirstVolume)))
            {
                Print("Volume entered incorrectly, volume has been changed to ", FirstVolume);
            }

            //Positions.Closed += OnPositionsClosed;
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

            // Close all buy positions if all buy positions' target profit is met
            if (numBuyPositions > 0)
            {
                if (buyPositionLst.Average(x => x.NetProfit) >= FirstVolume * AverageTakeProfit * Symbol.PipSize)
                {
                    foreach (var position in Positions)
                    {
                        if (position.TradeType == TradeType.Buy && position.SymbolName == SymbolName && position.Label == ThiscBotLabel)
                        {
                            ClosePosition(position);
                        }
                    }
                }
            }
            // Close all sell positions if all sell positions' target profit is met
            if (numSellPositions > 0)
            {
                if (sellPositionLst.Average(x => x.NetProfit) >= FirstVolume * AverageTakeProfit * Symbol.PipSize)
                {
                    foreach (var position in Positions)
                    {
                        if (position.TradeType == TradeType.Sell && position.SymbolName == SymbolName && position.Label == ThiscBotLabel)
                        {
                            ClosePosition(position);
                        }
                    }
                }
            }
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

        private void OnPositionsClosed(PositionClosedEventArgs args)
        {
            Print("Closed");
            var position = args.Position;

            if (position.Label != "Martingale" || position.SymbolName != SymbolName)
                return;

            if (position.GrossProfit > 0)
            {
                ExecuteMarketOrder(GetRandomTradeType(), position.SymbolName, FirstVolume, position.Label);
            }
            else
            {
                ExecuteMarketOrder(position.TradeType, position.SymbolName, position.Quantity * 2, position.Label);
            }
        }

        private TradeType GetRandomTradeType()
        {
            return prng.Next(2) == 0 ? TradeType.Buy : TradeType.Sell;
        }

        void RandomWalk(int numBuyPositions, int numSellPositions)
        {
            if (prng.Next(0, 2) == 0)
            {
                if (numBuyPositions < MaxOpenBuy)
                {
                    ProcessBuy();
                }
            }
            else
            {
                if (numSellPositions < MaxOpenSell)
                {
                    ProcessSell();
                }
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
                foreach (var position in Positions)
                {
                    if (position.SymbolName == SymbolName && position.Label == ThiscBotLabel)
                    {
                        ClosePosition(position);
                    }
                }
            }
        }

        #endregion

        #region Helper Methods

        private void ProcessBuy()
        {
            var buyPositionLst = Positions.Where(x => x.TradeType == TradeType.Buy && x.SymbolName == SymbolName && x.Label == ThiscBotLabel);
            int numBuyPositions = buyPositionLst.Count();

            if (numBuyPositions == 0 && MarketSeries.Close.Last(1) > MarketSeries.Close.Last(2))
            {
                ExecuteMarketOrder(TradeType.Buy, SymbolName, FirstVolume, ThiscBotLabel);

                LastBuyTradeTime = MarketSeries.OpenTime.Last(0);
            }
            if (numBuyPositions > 0)
            {
                if (Symbol.Ask < (buyPositionLst.Min(x => x.EntryPrice) - PipStep * Symbol.PipSize) && LastBuyTradeTime != MarketSeries.OpenTime.Last(0))
                {
                    ExecuteMarketOrder(TradeType.Buy, SymbolName, CalculateVolume(TradeType.Buy), ThiscBotLabel);

                    LastBuyTradeTime = MarketSeries.OpenTime.Last(0);
                }
            }
        }

        private void ProcessSell()
        {
            var sellPositionLst = Positions.Where(x => x.TradeType == TradeType.Sell && x.SymbolName == SymbolName && x.Label == ThiscBotLabel);
            int numSellPositions = sellPositionLst.Count();

            if (numSellPositions == 0 && MarketSeries.Close.Last(2) > MarketSeries.Close.Last(1))
            {
                ExecuteMarketOrder(TradeType.Sell, SymbolName, FirstVolume, ThiscBotLabel);

                LastSellTradeTime = MarketSeries.OpenTime.Last(0);
            }
            if (numSellPositions > 0)
            {
                if (Symbol.Bid > (sellPositionLst.Max(x => x.EntryPrice) + PipStep * Symbol.PipSize) && LastSellTradeTime != MarketSeries.OpenTime.Last(0))
                {
                    ExecuteMarketOrder(TradeType.Sell, SymbolName, CalculateVolume(TradeType.Sell), ThiscBotLabel);

                    LastSellTradeTime = MarketSeries.OpenTime.Last(0);
                }
            }
        }

        private double CalculateVolume(TradeType tradeType)
        {
            int numPositions = Positions.Count(x => x.TradeType == tradeType && x.SymbolName == SymbolName && x.Label == ThiscBotLabel);
            //var volumeInUnits = Symbol.QuantityToVolumeInUnits(FirstVolume);

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
