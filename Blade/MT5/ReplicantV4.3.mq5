//+------------------------------------------------------------------+
//|                                                      ProjectName |
//|                                      Copyright 2020, CompanyName |
//|                                       http://www.companyname.net |
//+------------------------------------------------------------------+
#property copyright   "Copyright 2020, Metamorphic."
#property link        "https://jumpinalake.com"
#property version     "4.3"
#property description "Expert Advisor"
#property strict

#include <Trade\PositionInfo.mqh> CPositionInfo     m_position;
#include <Trade\Trade.mqh> CTrade trade;

//************************************************************************************************/
//*                                                                                              */
//************************************************************************************************/
input string a = "$$$$$$$$$$$$$$ Basic Settings $$$$$$$$$$$$$$";
input double      TakeProfit                 = 25;             // Take Profit (in pips)
input int         MagicNumber                = 666;            // Advisor Magic Number
input int         Slippage                   = 30;             // Slippage (in pips)
input int         MaxOpenBuy                 = 15;             // Maximum number of open buy positions
input int         MaxOpenSell                = 15;             // Maximum number of open sell positions
input double      PipStep                    = 0;              // Pip step

input string b = "$$$$$$$$$$$$$$ Bollinger Band Settings $$$$$$$$$$$$$$";
input int         BBPeriod                   =  200;           // Bolinger band period
input double      BBDeviation                =  2;             // Bollinger band deviation
input int         BBShift                    =  0;             // Bollinger band shift

input string c = "$$$$$$$$$$$$$$ Advanced Setup $$$$$$$$$$$$$$";
input double      FirstVolume                = 0.01;           // First order volume scalar
input double      VolumeExponent             = 1.0;            // Volume exponent
input double      PowerFactor                = 1.0;            // Power factor
input double      PowerOffset                = 0.0;            // Power offset
input double      PercentIncreaseThreshold   = 100000;         // Percent increase threshold (Free margin denominator)
input double      RiskPercent                = 10.0;           // Risk percent (Risk managment)
input string d = "$$$$$$$$$$$$$$ Strategy $$$$$$$$$$$$$$";
input bool        ApplyBollingerBands        = true;           // Apply bollinger bands strategy
input bool        ApplyRandomWalk            = false;          // Apply random walk
string algoTitle = "Replicant";

datetime LastBuyTradeTime;
datetime LastSellTradeTime;

enum BL_TRADE_TYPE
  {
   BL_BUY,
   BL_SELL
  };

//---
//************************************************************************************************/
//*                                                                                              */
//************************************************************************************************/
int OnInit()
  {
   Comment("");
// Use current time as seed for random generator
   MathSrand(GetTickCount()); //srand(time(0));
   trade.LogLevel(LOG_LEVEL_ERRORS);
   trade.SetExpertMagicNumber(MagicNumber);
   trade.SetDeviationInPoints(Slippage);
   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double GetPipSize()
  {
     return Point();
   //return Point()*(Digits%2==1 ? 10 : 1); // for forex only
//return 0.0001;
//return 1;
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void GetNumOpenPositions(int& numOpenBuy, int& numOpenSell)
  {
   for(int i=PositionsTotal()-1; i>=0; i--)
     {
      if(m_position.SelectByIndex(i))
        {
         if(m_position.Symbol()==Symbol())
           {
            if(m_position.Magic()==MagicNumber)
              {
               if(m_position.PositionType()==POSITION_TYPE_BUY)
                 {
                  numOpenBuy ++;
                 }
               if(m_position.PositionType()==POSITION_TYPE_SELL)
                 {
                  numOpenSell++;
                 }
              }
           }
        }
     }
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int GetCountOpenPositions(BL_TRADE_TYPE tradeType)
  {
   int count = 0;
   for(int i=PositionsTotal()-1; i>=0; i--)
     {
      if(m_position.SelectByIndex(i))
        {
         if(m_position.Symbol()==Symbol())
           {
            if(m_position.Magic()==MagicNumber)
              {
               if(m_position.PositionType()==POSITION_TYPE_BUY)
                 {
                  count ++;
                 }
               if(m_position.PositionType()==POSITION_TYPE_SELL)
                 {
                  count++;
                 }
              }
           }
        }
     }
   return count;
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double CalculateVolume(BL_TRADE_TYPE tradeType)
  {
   int countPositions = 0;
   int numOpenSell;
   int numOpenBuy;

   GetNumOpenPositions(numOpenBuy, numOpenSell);

   if(tradeType == BL_BUY)
     {
      countPositions = numOpenBuy;
     }
   else
      if(tradeType == BL_SELL)
        {
         countPositions = numOpenSell;
        }

   double risk = RiskPercent / 100.0;

   double freeMargin = AccountInfoDouble(ACCOUNT_FREEMARGIN);
   
   double lotMultiplier = (freeMargin / PercentIncreaseThreshold) * risk;

   double volume = ((lotMultiplier + FirstVolume) * MathPow(VolumeExponent, PowerOffset + (PowerFactor * countPositions)));

   return volume;
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int GetRandom(int max, int min)
  {
   return MathRand()%((max + 1) - min) + min;
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int GetRandomTradeType()
  {
   if(GetRandom(1,0) == 0)
     {
      return BL_SELL;
     }
   else
     {
      return BL_BUY;
     }
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool ExecuteOrder(BL_TRADE_TYPE tradeType, double volume, ulong deviation, double stopLoss, double takeProfit, ulong magicNumber)
  {
   MqlTradeRequest mrequest;                             // Will be used for trade requests
   MqlTradeResult mresult;                               // Will be used for results of trade requests

   ZeroMemory(mrequest);
   ZeroMemory(mresult);

   double Ask = SymbolInfoDouble(_Symbol,SYMBOL_ASK);    // Ask price
   double Bid = SymbolInfoDouble(_Symbol,SYMBOL_BID);    // Bid price

   double price = NormalizeDouble((tradeType == BL_BUY ? Ask : Bid),_Digits);

   ENUM_ORDER_TYPE orderType = (tradeType == BL_BUY ? ORDER_TYPE_BUY : ORDER_TYPE_SELL);

   if(!PositionSelect(_Symbol))
     {
      mrequest.action = TRADE_ACTION_DEAL;               // Immediate order execution
      mrequest.price = price;                            // Lastest Ask price
      mrequest.sl = stopLoss;                            // Stop Loss
      mrequest.tp = takeProfit;                          // Take Profit
      mrequest.symbol = _Symbol;                         // Symbol
      mrequest.volume = volume;                          // Number of lots to trade
      mrequest.magic = magicNumber;                      // Magic Number
      mrequest.type = orderType;
      mrequest.type_filling = ORDER_FILLING_FOK;         // Order execution type
      mrequest.deviation=deviation;                      // Deviation from current price

      return OrderSend(mrequest,mresult);                // Send order
     }
   return false;
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int ExecuteMarketOrder(BL_TRADE_TYPE tradeType)
  {
   ExecuteOrder(tradeType, NormalizeDouble(CalculateVolume(BL_BUY), 2), Slippage, 0,0, MagicNumber);

   return -1;
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double GetMinEntryPrice(BL_TRADE_TYPE tradeType)
  {
//var positionLst = Positions.Where(x => x.TradeType == tradeType && x.SymbolName == SymbolName && x.Label == ThiscBotLabel);
// return positionLst.Min(x => x.EntryPrice);
   double minEntryPrice = DBL_MAX;
   double price;
   
   for(int i=PositionsTotal()-1; i>=0; i--)
   {
     if(m_position.SelectByIndex(i))
     {
      if(m_position.Symbol()==Symbol())
        {
         if(m_position.Magic()==MagicNumber)
           {
            if(m_position.PositionType()==POSITION_TYPE_BUY)
              {
               price = m_position.PriceOpen();
               if(price < minEntryPrice)
               {
                  minEntryPrice = price;
               }
              }
            if(m_position.PositionType()==POSITION_TYPE_SELL)
              {
               price = m_position.PriceOpen();
               if(price < minEntryPrice)
               {
                  minEntryPrice = price;
               }
              }
           }
        }
     }
   }
//Print("minEntryPrice=", minEntryPrice);
   return minEntryPrice;
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double GetMaxEntryPrice(BL_TRADE_TYPE tradeType)
  {
//var positionLst = Positions.Where(x => x.TradeType == tradeType && x.SymbolName == SymbolName && x.Label == ThiscBotLabel);
// return positionLst.Max(x => x.EntryPrice);
   double maxEntryPrice = 0;
   double price;

   for(int i=PositionsTotal()-1; i>=0; i--)
   {
     if(m_position.SelectByIndex(i))
     {
      if(m_position.Symbol()==Symbol())
        {
         if(m_position.Magic()==MagicNumber)
           {
            if(m_position.PositionType()==POSITION_TYPE_BUY)
              {
               price = m_position.PriceOpen();
               if(price > maxEntryPrice)
               {
                  maxEntryPrice = price;
               }
              }
            if(m_position.PositionType()==POSITION_TYPE_SELL)
              {
               price = m_position.PriceOpen();
               if(price > maxEntryPrice)
               {
                  maxEntryPrice = price;
               }
              }
           }
        }
     }
   }
//Print("maxEntryPrice=", maxEntryPrice);
   return maxEntryPrice;
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
datetime GetTimeLastOrder(int ticketId)
  {
   //bool result = false;
   //result = OrderSelect(ticketId);
   m_position.SelectByTicket(ticketId);
   return m_position.Time();
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void ProcessOrder(BL_TRADE_TYPE tradeType)
  {

   double Ask = SymbolInfoDouble(_Symbol,SYMBOL_ASK);
   double Bid = SymbolInfoDouble(_Symbol,SYMBOL_BID);

//ExecuteMarketOrder(tradeType);return;

   int numPositions = GetCountOpenPositions(tradeType);

///*

   double CLOSE1 = iClose(_Symbol,PERIOD_CURRENT,1);
   double CLOSE2 = iClose(_Symbol,PERIOD_CURRENT,2);

   bool buyProfit = (tradeType == BL_BUY) && (CLOSE1 > CLOSE2);
   bool sellCheap = (tradeType == BL_SELL) && (CLOSE2 > CLOSE1);

   if(numPositions == 0 && (buyProfit || sellCheap)) //MarketSeries.Close.Last(1) > MarketSeries.Close.Last(2))
      //if (numPositions == 0)
     {
      //Print("TEST1");
      int ticketId = ExecuteMarketOrder(tradeType);
      //datetime now = GetTimeLastOrder(ticketId);
      //OrderSelect((OrdersTotal() -1 ), SELECT_BY_POS);
      //datetime now = GetTimeLastOrder(OrderTicket());
      //datetime now = Time[0];
      datetime now = iTime(Symbol(), PERIOD_CURRENT, 0);

      if(tradeType == BL_BUY)
        {
         LastBuyTradeTime = now;//MarketSeries.OpenTime.Last(0);
        }
      else
         if(tradeType == BL_SELL)
           {
            LastSellTradeTime = now;//MarketSeries.OpenTime.Last(0);
           }
     }
   if(numPositions > 0)  // */
     {
      //Print("TEST2");
      /*
       double lastbuyopenprice,lastsellopenprice;int i;
          lastbuyopenprice=0;
         lastsellopenprice=0;
         if(OrdersTotal()>0){
            for(i=0;i<=OrdersTotal();i++){
               bool result = OrderSelect(i,SELECT_BY_POS,MODE_TRADES);
               if(OrderSymbol()==Symbol() && OrderMagicNumber()==MagicNumber && OrderCloseTime()==0){
                  if(OrderType()==OP_BUY){
                     lastbuyopenprice=OrderOpenPrice();
                  }
                  if(OrderType()==OP_SELL){
                     lastsellopenprice=OrderOpenPrice();
                  }
               }
            }
         }

       double lastopenprice = tradeType == BL_BUY ? lastbuyopenprice : lastsellopenprice;
       */


      datetime lastTradeTime = (tradeType == BL_BUY) ? LastBuyTradeTime : LastSellTradeTime;

      //OrderSelect((OrdersTotal() -1 ), SELECT_BY_POS);
      //datetime timeOfLastOrder = GetTimeLastOrder(OrderTicket());

      double price = tradeType == BL_BUY ? Ask : Bid;

      datetime now = iTime(Symbol(),PERIOD_CURRENT,0); // Time[0]

      bool executePipStep = false;
      if(tradeType == BL_BUY)
        {
         if(Ask < (GetMinEntryPrice(tradeType) - PipStep * GetPipSize()) && lastTradeTime != now)
           {
            executePipStep = true;
           }
        }
      else
        {
         if(Bid > (GetMaxEntryPrice(tradeType) + PipStep * GetPipSize()) && lastTradeTime != now)
           {
            executePipStep = true;
           }
        }

      // * GetPipSize()
      //if (price < (GetMinEntryPrice(tradeType) - PipStep * GetPipSize()))// && lastTradeTime != timeOfLastOrder)
      //if (price < (GetMinEntryPrice(tradeType) - PipStep * GetPipSize()))// && lastTradeTime != Time[0]) // && lastTradeTime != MarketSeries.OpenTime.Last(0))
      //if(price >= GetMinEntryPrice(tradeType) + PipStep * GetPipSize() && lastTradeTime != Time[0])
      //if(lastopenprice !=0 && Ask >= lastopenprice+PipStep*GetPipSize() && lastTradeTime != Time[0])
      //if(lastopenprice !=0 && Bid <= lastopenprice-PipStep*GetPipSize() && lastTradeTime != Time[0])
      if(executePipStep)
        {
         //Print("TEST3");
         int ticketId = ExecuteMarketOrder(tradeType);
         //datetime now = GetTimeLastOrder(ticketId);

         //OrderSelect((OrdersTotal() -1 ), SELECT_BY_POS);
         //datetime now = GetTimeLastOrder(OrderTicket());
         //datetime now = iTime(Symbol(),PERIOD_CURRENT,0); //datetime now = Time[0];


         if(tradeType == BL_BUY)
           {
            LastBuyTradeTime = now;//MarketSeries.OpenTime.Last(0);
           }
         else
            if(tradeType == BL_SELL)
              {
               LastSellTradeTime = now;//MarketSeries.OpenTime.Last(0);
              }
        }
     }
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void DoBuy()
  {
   int numOpenSell;
   int numOpenBuy;

   GetNumOpenPositions(numOpenBuy, numOpenSell);

   if(numOpenBuy < MaxOpenBuy)
     {
      ProcessOrder(BL_BUY);
     }
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void DoSell()
  {
   int numOpenSell;
   int numOpenBuy;

   GetNumOpenPositions(numOpenBuy, numOpenSell);

   if(numOpenSell < MaxOpenSell)
     {
      ProcessOrder(BL_SELL);
     }
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CloseTrades(BL_TRADE_TYPE tradeType)
  {
    for(int i=PositionsTotal()-1; i>=0; i--)
    {
      if(m_position.SelectByIndex(i))
        {
         if(m_position.Symbol()==Symbol())
           {
            if(m_position.Magic()==MagicNumber)
              {
               if(m_position.PositionType()==POSITION_TYPE_BUY && tradeType == BL_BUY)
                 {
                  trade.PositionClose(m_position.Ticket(), Slippage);
                 }
               if(m_position.PositionType()==POSITION_TYPE_SELL && tradeType == BL_SELL)
                 {
                  trade.PositionClose(m_position.Ticket(), Slippage);
                 }
              }
           }
        }
     }
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void SmartGrid()
  {
   double targetProfit = FirstVolume * TakeProfit * GetPipSize() * 1000000 ; // * 10000000
//double targetProfit = FirstVolume * TakeProfit * GetPipSize();// * Point(); // Symbol.PipSize
//double targetProfit = FirstVolume * TakeProfit * Point();
//double targetProfit = FirstVolume * TakeProfit;

   int numOpenBuy = 0;
   int numOpenSell = 0;
   double buyNetProfit = 0;
   double sellNetProfit = 0;
   double averageBuyNetProfit;
   double averageSellNetProfit;

    for(int i=PositionsTotal()-1; i>=0; i--)
    {
      if(m_position.SelectByIndex(i))
        {
         if(m_position.Symbol()==Symbol())
           {
            if(m_position.Magic()==MagicNumber)
              {
               if(m_position.PositionType()==POSITION_TYPE_BUY)
                 {
                  numOpenBuy ++;
                  buyNetProfit += m_position.Profit() + m_position.Commission() + m_position.Swap();
                 }
               if(m_position.PositionType()==POSITION_TYPE_SELL)
                 {
                  numOpenSell++;
                  sellNetProfit += m_position.Profit() + m_position.Commission() + m_position.Swap();
                 }
              }
           }
        }
     }

   if(numOpenBuy > 0)
     {

      //targetProfit = FirstVolume * Ask + TakeProfit * GetPipSize();
      averageBuyNetProfit = buyNetProfit / numOpenBuy;

      //Print("averageBuyNetProfit: ", averageBuyNetProfit);
      //Print("targetProfit: ", targetProfit);

      if(averageBuyNetProfit >= targetProfit)
        {
         Print("Closing BUY trades");
         CloseTrades(BL_BUY);
        }
     }

   if(numOpenSell > 0)
     {

      //targetProfit = FirstVolume * Bid - TakeProfit * GetPipSize();
      averageSellNetProfit = sellNetProfit / numOpenSell;

      //Print("averageSellNetProfit: ", averageSellNetProfit);
      //Print("targetProfit: ", targetProfit);
      //if(averageSellNetProfit >0)
      //{
      //Print("GTZERO");
      //Print("averageSellNetProfit: ", averageSellNetProfit);
      //Print("targetProfit: ", targetProfit);
      //}
      if(averageSellNetProfit >= targetProfit)
        {
         Print("Closing SELL trades");
         CloseTrades(BL_SELL);
        }
     }
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void BollingerBands()
  {
   int period = PERIOD_CURRENT;
//double BB_UPPER   = iBands(Symbol(),period,BBPeriod ,BBDeviation ,BBShift ,PRICE_CLOSE ,MODE_UPPER ,0);
//double BB_SMA     = iBands(Symbol(),period ,BBPeriod ,BBDeviation ,BBShift ,PRICE_CLOSE ,MODE_SMA   ,0);
//double BB_LOWER   = iBands(Symbol(),period ,BBPeriod ,BBDeviation ,BBShift ,PRICE_CLOSE ,MODE_LOWER ,0);
   //double BB_UPPER1   = iBands(Symbol(),period,BBPeriod,BBShift,BBDeviation,PRICE_CLOSE,MODE_UPPER,1);       // Top band
   //double BB_UPPER2   = iBands(Symbol(),period,BBPeriod,BBShift,BBDeviation,PRICE_CLOSE,MODE_UPPER,2);
   //double BB_LOWER1   = iBands(Symbol(),period,BBPeriod,BBShift,BBDeviation,PRICE_CLOSE,MODE_LOWER,1);       // Bottom band
   //double BB_LOWER2   = iBands(Symbol(),period,BBPeriod,BBShift,BBDeviation,PRICE_CLOSE,MODE_LOWER,2);

   int BolBandsHandle = iBands(Symbol(),PERIOD_CURRENT,BBPeriod,BBShift,BBDeviation,PRICE_CLOSE);

   if(BolBandsHandle<0)
     {
      Print("Error in creation of bollinger bands indicator - error: ",GetLastError());
      return;
     }
     
   double BBUp[],BBLow[],BBMidle[];   // dynamic arrays for numerical values of Bollinger Bands
   ArraySetAsSeries(BBUp,true);
   ArraySetAsSeries(BBLow,true);
   ArraySetAsSeries(BBMidle,true);

   if(CopyBuffer(BolBandsHandle,0,0,3,BBMidle)<0 || CopyBuffer(BolBandsHandle,1,0,3,BBUp)<0 || CopyBuffer(BolBandsHandle,2,0,3,BBLow)<0)
     {
      Print("Error copying Bollinger Bands indicator Buffers - error:",GetLastError());
      return;
     }

   double BB_UPPER1 = BBUp[1];
   double BB_UPPER2 = BBUp[2];
   double BB_LOWER1 = BBLow[1];
   double BB_LOWER2 = BBLow[2];
   
   double CLOSE1 = iClose(_Symbol,PERIOD_CURRENT,1);
   double CLOSE2 = iClose(_Symbol,PERIOD_CURRENT,2);
   
   if(Bars(Symbol(),PERIOD_CURRENT) > 2)
     {
      if(CLOSE2 > BB_UPPER2) //boll.Top.Last(2))
        {
         if(CLOSE1 > BB_UPPER1) //boll.Top.Last(1))
           {
            DoBuy();
           }
         else
           {
            DoSell();
           }
        }
      else
         if(CLOSE2 < BB_LOWER2) //boll.Bottom.Last(2))
           {
            if(CLOSE1 < BB_LOWER1) //boll.Bottom.Last(1))
              {
               DoSell();
              }
            else
              {
               DoBuy();
              }
           }
     }
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void RandomWalk()
  {
//Print("Test 123: ", rand());
//Print("Test: ", rand() % 10);
//Print("Test: ", GetRandom(10,0));
//Print("Test: ", GetRandom(1,0));

   if(GetRandomTradeType() == BL_BUY)
     {
      DoBuy();
     }
   else
     {
      DoSell();
     }
  }
//************************************************************************************************/
//*                                                                                              */
//************************************************************************************************/
void OnTick()
  {

//*************************************************************//
   SmartGrid();

   if(ApplyBollingerBands)
     {
      BollingerBands();
     }
   if(ApplyRandomWalk)
     {
      RandomWalk();
     }

//*************************************************************//

  }
//************************************************************************************************/
//*                                                                                              */
//************************************************************************************************/
void OnDeinit(const int reason)
  {

  }
//************************************************************************************************/
//+------------------------------------------------------------------+
