#property copyright   "Copyright 2020, Metamorphic."
#property link        "https://jumpinalake.com"
#property version     "4.5"
#property description "Expert Advisor"
#property strict

//************************************************************************************************/
//*                                                                                              */
//************************************************************************************************/
input string a = "$$$$$$$$$$$$$$ Basic Settings $$$$$$$$$$$$$$";
input double      TakeProfit                 = 5;             // Take Profit (in pips)
input int         MagicNumber                = 666;            // Advisor Magic Number
input int         Slippage                   = 5;             // Slippage (in pips)
input int         MaxOpenBuy                 = 35;             // Maximum number of open buy positions
input int         MaxOpenSell                = 35;             // Maximum number of open sell positions
input double      PipStep                    = 175;              // Pip step

input string b = "$$$$$$$$$$$$$$ Bollinger Band Settings $$$$$$$$$$$$$$";
input int         BBPeriod                   =  75;           // Bolinger band period
input double      BBDeviation                =  2;             // Bollinger band deviation (stddev)
input int         BBShift                    =  0;             // Bollinger band shift

input string c = "$$$$$$$$$$$$$$ Advanced Setup $$$$$$$$$$$$$$";
input double      FirstVolume                = 0.01;           // First order volume scalar
input double      VolumeExponent             = 1.3;            // Volume exponent
input double      PowerFactor                = 1.0;            // Power factor
input double      PowerOffset                = 0.0;            // Power offset
input double      PercentIncreaseThreshold   = 100000;         // Percent increase threshold (Free margin denominator)
input double      RiskPercent                = 75.0;           // Risk percent (Risk managment)
input string d = "$$$$$$$$$$$$$$ Strategy $$$$$$$$$$$$$$";
input bool        ApplyBollingerBands        = true;           // Apply bollinger bands strategy
input bool        ApplyRandomWalk            = false;          // Apply random walk
input bool        AutoQuitOnCompleteClose    = false;          // Auto quit on complete close: when buys and sells are closed off in profit

string algoTitle = "Replicant";
bool continueBuyTrading = true;
bool continueSellTrading = true;
bool firstOrder = true;

datetime LastBuyTradeTime;
datetime LastSellTradeTime;
double lastBuyPrice;
double lastSellPrice;

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
   return(INIT_SUCCEEDED);
  }
  
double GetPipSize()
{
   //return Point();
   return Point()*(Digits%2==1 ? 10 : 1); // for forex only
   //return 0.0001;
   //return 1;
}

void GetNumOpenPositions(int& numOpenBuy, int& numOpenSell)
{
   numOpenBuy = 0;
   numOpenSell = 0;
   
   //for(int i=0;i<OrdersTotal();i++)
   for(int i=OrdersTotal()-1; i>=0; i--)
   {
       if(OrderSelect(i, SELECT_BY_POS, MODE_TRADES))
       {
          if(OrderMagicNumber() != MagicNumber) break;
       
          if(OrderType()==OP_BUY  && OrderSymbol() == Symbol())
          {
            numOpenBuy ++;
          }
          if(OrderType()==OP_SELL && OrderSymbol() == Symbol())
          {
            numOpenSell++;
          }
       }
   }
}

int GetCountOpenPositions(BL_TRADE_TYPE tradeType)
{
   int count = 0;
   //for(int i=0;i<OrdersTotal();i++)
   for(int i=OrdersTotal()-1; i>=0; i--)
   {
       if(OrderSelect(i, SELECT_BY_POS, MODE_TRADES))
       {
          if(OrderMagicNumber() != MagicNumber) break;
       
          if(OrderType()==OP_BUY  && OrderSymbol() == Symbol())
          {
            if(tradeType == BL_BUY)
            {
               count ++;
            }
          }
          if(OrderType()==OP_SELL && OrderSymbol() == Symbol())
          {
            if(tradeType == BL_SELL)
            {
               count ++;
            }
          }
       }
   }
   return count;
}

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
   else if (tradeType == BL_SELL)
   {
      countPositions = numOpenSell;
   }
   
   double risk = RiskPercent / 100.0;
   
   double lotMultiplier = (AccountFreeMargin() / PercentIncreaseThreshold) * risk;
   
   double volume = ((lotMultiplier + FirstVolume) * MathPow(VolumeExponent, PowerOffset + (PowerFactor * countPositions)));
   
   return volume;
}

int GetRandom(int max, int min)
{
   return MathRand()%((max + 1) - min) + min;
}

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

double ExecuteMarketOrder(BL_TRADE_TYPE tradeType)
{
   int ticketId = -1;
   double price = 0;
   
   if(tradeType == BL_BUY && continueBuyTrading)
   {
      price = NormalizeDouble(Ask, Digits());
      ticketId = OrderSend(Symbol(), OP_BUY, NormalizeDouble(CalculateVolume(BL_BUY), 2), price, Slippage, 0, 0, algoTitle, MagicNumber, 0, clrGreen);
      //ticketId = OrderSend(Symbol(), OP_BUY, NormalizeDouble(CalculateVolume(BL_BUY), 2), NormalizeDouble(Ask, Digits()), Slippage, 0, Ask + TakeProfit * GetPipSize(), algoTitle, MagicNumber, 0, clrGreen);
      if(ticketId < 0)
         Print("OrderSend error #", GetLastError());   
   }
   else if(tradeType == BL_SELL && continueSellTrading)
   {
      price = NormalizeDouble(Bid, Digits());
      ticketId = OrderSend(Symbol(), OP_SELL, NormalizeDouble(CalculateVolume(BL_SELL), 2), price, Slippage, 0, 0, algoTitle, MagicNumber, 0, clrRed);
      //ticketId = OrderSend(Symbol(), OP_SELL, NormalizeDouble(CalculateVolume(BL_SELL), 2), NormalizeDouble(Bid, Digits()), Slippage, 0, Bid - TakeProfit * GetPipSize(), algoTitle, MagicNumber, 0, clrRed);
      if(ticketId < 0)
         Print("OrderSend error #", GetLastError());
   }
   
   return price;
}

double GetMinEntryPrice(BL_TRADE_TYPE tradeType)
{
   //var positionLst = Positions.Where(x => x.TradeType == tradeType && x.SymbolName == SymbolName && x.Label == ThiscBotLabel);
   // return positionLst.Min(x => x.EntryPrice);
   double minEntryPrice = DBL_MAX;
   double price;
   
   //for(int i=0;i<OrdersTotal();i++)
   for(int i=OrdersTotal()-1; i>=0; i--)
   {
       if(OrderSelect(i, SELECT_BY_POS, MODE_TRADES))
       {
          if(OrderMagicNumber() != MagicNumber) break;
       
          if(OrderType()==OP_BUY  && OrderSymbol() == Symbol())
          {
            if(tradeType == BL_BUY)
            {
               price = OrderOpenPrice();
               if(price < minEntryPrice)
               {
                  minEntryPrice = price;
               }
            }
          }
          if(OrderType()==OP_SELL && OrderSymbol() == Symbol())
          {
            if(tradeType == BL_SELL)
            {
               price = OrderOpenPrice();
               if(price < minEntryPrice)
               {
                  minEntryPrice = price;
               }
            }
          }
       }
   }
   //Print("minEntryPrice=", minEntryPrice);
   return minEntryPrice;
}

double GetMaxEntryPrice(BL_TRADE_TYPE tradeType)
{
   //var positionLst = Positions.Where(x => x.TradeType == tradeType && x.SymbolName == SymbolName && x.Label == ThiscBotLabel);
   // return positionLst.Max(x => x.EntryPrice);
   double maxEntryPrice = 0;
   double price;
   
   //for(int i=0;i<OrdersTotal();i++)
   for(int i=OrdersTotal()-1; i>=0; i--)
   {
       if(OrderSelect(i, SELECT_BY_POS, MODE_TRADES))
       {
          if(OrderMagicNumber() != MagicNumber) break;
       
          if(OrderType()==OP_BUY  && OrderSymbol() == Symbol())
          {
            if(tradeType == BL_BUY)
            {
               price = OrderOpenPrice();
               if(price > maxEntryPrice)
               {
                  maxEntryPrice = price;
               }
            }
          }
          if(OrderType()==OP_SELL && OrderSymbol() == Symbol())
          {
            if(tradeType == BL_SELL)
            {
               price = OrderOpenPrice();
               if(price > maxEntryPrice)
               {
                  maxEntryPrice = price;
               }
            }
          }
       }
   }
   //Print("maxEntryPrice=", maxEntryPrice);
   return maxEntryPrice;
}

datetime GetTimeLastOrder(int ticketId)
{
   bool result = false;
   result = OrderSelect(ticketId,SELECT_BY_TICKET);
   return OrderOpenTime();
   //return OrderCloseTime();
}  

double GetLastOpenPrice(BL_TRADE_TYPE tradeType)
{
  double lastbuyopenprice = 0,lastsellopenprice = 0;
  
  lastbuyopenprice=0;
  lastsellopenprice=0;
  if(OrdersTotal()>0)
  {
   for(int i=OrdersTotal()-1;i>=0;i--)
   //for(int i=0;i<=OrdersTotal();i++)
   {
      bool result = OrderSelect(i,SELECT_BY_POS,MODE_TRADES);
      if(OrderSymbol()==Symbol() && OrderMagicNumber()==MagicNumber && OrderCloseTime()==0)
      {
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
  return lastopenprice;
}

void ProcessOrder(BL_TRADE_TYPE tradeType)
{

//ExecuteMarketOrder(tradeType);return;

   int numPositions = GetCountOpenPositions(tradeType);

///*

   bool buyProfit = (tradeType == BL_BUY) && (Close[1] > Close[2]);
   bool sellCheap = (tradeType == BL_SELL) && (Close[2] > Close[1]);

   datetime now = Time[0];

   if (numPositions == 0 && (buyProfit || sellCheap))//MarketSeries.Close.Last(1) > MarketSeries.Close.Last(2))
   //if (numPositions == 0)
   {
      //Print("TEST1");
       double price = ExecuteMarketOrder(tradeType);
       
       firstOrder = false;
       //datetime now = GetTimeLastOrder(ticketId);
       //OrderSelect((OrdersTotal() -1 ), SELECT_BY_POS);
       //datetime now = GetTimeLastOrder(OrderTicket());
       
       if (tradeType == BL_BUY)
       {
           LastBuyTradeTime = now;//MarketSeries.OpenTime.Last(0);
           lastBuyPrice = price;
       }
       else if (tradeType == BL_SELL)
       {
           LastSellTradeTime = now;//MarketSeries.OpenTime.Last(0);
           lastSellPrice = price;
       }
   }
   if (numPositions > 0) // */
   {
      //Print("TEST2");
       
       double lastopenprice = GetLastOpenPrice(tradeType);
       lastopenprice = (tradeType == BL_BUY) ? lastBuyPrice : lastSellPrice;
       
       datetime lastTradeTime = (tradeType == BL_BUY) ? LastBuyTradeTime : LastSellTradeTime;

       //OrderSelect((OrdersTotal() -1 ), SELECT_BY_POS);
       //datetime timeOfLastOrder = GetTimeLastOrder(OrderTicket());

       //double price = tradeType == BL_BUY ? Ask : Bid;
       
       bool executePipStep = false;
       
       if(tradeType == BL_BUY)
        {
         //if(Ask < (GetMinEntryPrice(tradeType) - PipStep * GetPipSize())) // && lastTradeTime != now)
         //if(lastopenprice != 0 && Ask >= lastopenprice+PipStep*GetPipSize())// && lastTradeTime != now)
         //if(Ask >= GetMinEntryPrice(tradeType)+PipStep*GetPipSize())// && lastTradeTime != now)
         if(lastopenprice != 0 && Ask < (lastopenprice - PipStep * GetPipSize()) && lastTradeTime != now)
           {
            executePipStep = true;
           }
        }
      else
        {
         //if(Bid > (GetMaxEntryPrice(tradeType) + PipStep * GetPipSize()))// && lastTradeTime != now)
         //if(lastopenprice != 0 && Bid <= lastopenprice-PipStep*GetPipSize())// && lastTradeTime != now)
         //if(Bid <= GetMaxEntryPrice(tradeType)-PipStep*GetPipSize())// && lastTradeTime != now)
         if(lastopenprice != 0 && Bid > (lastopenprice + PipStep * GetPipSize()) && lastTradeTime != now)
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
          double price = ExecuteMarketOrder(tradeType);
          //datetime now = GetTimeLastOrder(ticketId);
          
          firstOrder = false;
          
          //OrderSelect((OrdersTotal() -1 ), SELECT_BY_POS);
          //datetime now = GetTimeLastOrder(OrderTicket());
          
          if (tradeType == BL_BUY)
          {
              LastBuyTradeTime = now;//MarketSeries.OpenTime.Last(0);
              lastBuyPrice = price;
          }
          else if (tradeType == BL_SELL)
          {
              LastSellTradeTime = now;//MarketSeries.OpenTime.Last(0);
              lastSellPrice = price;
          }
       }
   }
}

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

void CloseTrades(BL_TRADE_TYPE tradeType)
{
   bool result=false;
   
   //for(int i=0;i<OrdersTotal();i++)
   for(int i=OrdersTotal()-1; i>=0; i--)
   {
       if(OrderSelect(i, SELECT_BY_POS, MODE_TRADES))
       {
          if(OrderMagicNumber() != MagicNumber) break;
       
          if(OrderType()==OP_BUY  && OrderSymbol() == Symbol())
          {
            if(tradeType == BL_BUY)
            {
               result = OrderClose(OrderTicket(), OrderLots(), Bid, Slippage, clrYellow); // Ask
            }
          }
          if(OrderType()==OP_SELL && OrderSymbol() == Symbol())
          {
            if(tradeType == BL_SELL)
            {
               result = OrderClose(OrderTicket(), OrderLots(), Ask, Slippage, clrYellow); // Bid
            }
          }
       }
   }
}

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
   
   //for(int i=0;i<OrdersTotal();i++)
   for(int i=OrdersTotal()-1; i>=0; i--)
   {
       if(OrderSelect(i, SELECT_BY_POS, MODE_TRADES))
       {
          if(OrderMagicNumber() != MagicNumber) break;
       
          if(OrderType()==OP_BUY  && OrderSymbol() == Symbol())
          {
            numOpenBuy ++;
            buyNetProfit += OrderProfit() + OrderCommission() + OrderSwap();
          }
          if(OrderType()==OP_SELL && OrderSymbol() == Symbol())
          {
            numOpenSell++;
            sellNetProfit += OrderProfit() + OrderCommission() + OrderSwap();
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
        
void BollingerBands()
{
   int period = PERIOD_CURRENT;
   //double BB_UPPER   = iBands(Symbol(),period,BBPeriod ,BBDeviation ,BBShift ,PRICE_CLOSE ,MODE_UPPER ,0);
   //double BB_SMA     = iBands(Symbol(),period ,BBPeriod ,BBDeviation ,BBShift ,PRICE_CLOSE ,MODE_SMA   ,0);
   //double BB_LOWER   = iBands(Symbol(),period ,BBPeriod ,BBDeviation ,BBShift ,PRICE_CLOSE ,MODE_LOWER ,0);
   double BB_UPPER1   = iBands(Symbol(),period ,BBPeriod ,BBDeviation ,BBShift ,PRICE_CLOSE ,MODE_UPPER ,1); // Top band
   double BB_UPPER2   = iBands(Symbol(),period ,BBPeriod ,BBDeviation ,BBShift ,PRICE_CLOSE ,MODE_UPPER ,2);
   double BB_LOWER1   = iBands(Symbol(),period ,BBPeriod ,BBDeviation ,BBShift ,PRICE_CLOSE ,MODE_LOWER ,1); // Bottom band
   double BB_LOWER2   = iBands(Symbol(),period ,BBPeriod ,BBDeviation ,BBShift ,PRICE_CLOSE ,MODE_LOWER ,2);
   
   if(Bars > 2)
   {
      if (Close[2] > BB_UPPER2)//boll.Top.Last(2))
       {
           if (Close[1] > BB_UPPER1)//boll.Top.Last(1))
           {
               DoBuy();
           }
           else
           {
               DoSell();
           }
       }
       else if (Close[2] < BB_LOWER2)//boll.Bottom.Last(2))
       {
           if (Close[1] < BB_LOWER1)//boll.Bottom.Last(1))
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

void AutoQuitOnClose()
{
   if(AutoQuitOnCompleteClose && !firstOrder)
   {
      if(GetCountOpenPositions(BL_BUY) == 0)
      {
         continueBuyTrading = false;
      }
      
      if(GetCountOpenPositions(BL_SELL) == 0)
      {
         continueSellTrading = false;
      }
   }
}

//************************************************************************************************/
//*                                                                                              */
//************************************************************************************************/
void OnTick()
  {

//*************************************************************//
   SmartGrid();
   
   AutoQuitOnClose();
   
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