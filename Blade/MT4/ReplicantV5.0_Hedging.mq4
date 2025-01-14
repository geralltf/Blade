#property copyright   "Copyright 2020, Metamorphic."
#property link        "https://jumpinalake.com"
#property version     "5.0"
#property description "Expert Advisor"
#property strict

//************************************************************************************************/
//*                                                                                              */
//************************************************************************************************/
input string a = "$$$$$$$$$$$$$$ Basic Settings $$$$$$$$$$$$$$";
input double      TakeProfit                 = 5;              // Take Profit (in pips)
input double      StopLoss                   = 1000;           // Stop Loss (in pips)
input bool        StopLossEnabled            = false;          // Enable silent Stop Loss
input int         MagicNumber                = 666;            // Advisor Magic Number
input int         Slippage                   = 5;              // Slippage (in pips)
input int         MaxOpenBuy                 = 35;             // Maximum number of open buy positions
input int         MaxOpenSell                = 35;             // Maximum number of open sell positions
//input double      PipStep                    = 175;            // Pip step (random walk)
//input double      PipStepBB                  = 175;            // Pip step (bolinger)
input string PipStepsRW="175";
input string PipStepsBB="175";
input double      PipScalar                  = 1.0;            // Pip scalar

input string b = "$$$$$$$$$$$$$$ Bollinger Band Settings $$$$$$$$$$$$$$";
input int         BBPeriod                   =  75;            // Bolinger band period
input double      BBDeviation                =  2;             // Bollinger band deviation (stddev)
input int         BBShift                    =  0;             // Bollinger band shift

input string c = "$$$$$$$$$$$$$$ Advanced Setup $$$$$$$$$$$$$$";
input double      FirstVolume                = 0.01;           // First order volume scalar
input double      VolumeExponent             = 1.3;            // Volume exponent
input double      PowerFactor                = 1.0;            // Power factor
input double      PowerOffset                = 0.0;            // Power offset
input double      PercentIncreaseThreshold   = 100000;         // Percent increase threshold (Free margin denominator)
input double      PercentIncreaseThreshold100 = 1000000;       // Percent increase threshold (Free margin used above $100K)
input double      RiskPercent                = 75.0;           // Risk percent (Risk managment)

input string f = "$$$$$$$$$$$$$$ Anti Drawdown $$$$$$$$$$$$$$";
input double      DrawdownPercent            = 50;            // Acceptable drawdown percentage
input double      WaitAfterDrawdownPips      = 1;             // Wait duration (in pips) after drawdown

input string e = "$$$$$$$$$$$$$$ Strategy $$$$$$$$$$$$$$";
input bool        ApplyBollingerBands        = true;           // Apply bollinger bands strategy
input bool        ApplyRandomWalk            = false;          // Apply random walk
input bool        ApplyHedging               = false;          // Apply hedging strategy
input bool        ApplyAntiDrawdown          = false;          // Apply anti-drawdown
input bool        AutoQuitOnCompleteClose    = false;          // Auto quit on complete close: when buys and sells are closed off in profit

string algoTitle = "Replicant";
bool continueBuyTrading = true;
bool continueSellTrading = true;
bool firstOrder = true;

string separator = ",";
int countPipstepsRW;
int countPipstepsBB;
double parsedPipStepsRW[100]; 
double parsedPipStepsBB[100];

datetime LastBuyTradeTime;
datetime LastSellTradeTime;
double lastBuyPrice;
double lastSellPrice;

bool IsNewBar;
double startprice = 0;
bool waiting = false;

enum BL_TRADE_TYPE
{
   BL_BUY,
   BL_SELL
};


void ParsePipsteps(string pipSteps, int& countPipsteps, double& out[])
{
   ushort u_sep = StringGetCharacter(separator,0); // The code of the separator character 
   string result[]; // An array to get strings 
   
   int k = StringSplit(pipSteps,u_sep,result); 
   
   countPipsteps = k;
   
   if(k > 0) 
   { 
      for(int i=0; i < k; i++) 
      { 
         out[i] = StrToDouble(result[i]);
         
         Print("pip step: ",i," = ",out[i]); 
      } 
   }
}

//---
//************************************************************************************************/
//*                                                                                              */
//************************************************************************************************/
int OnInit()
  {
   Comment("");
   // Use current time as seed for random generator 
   MathSrand(GetTickCount()); //srand(time(0));
   
   ParsePipsteps(PipStepsRW, countPipstepsRW, parsedPipStepsRW);
   ParsePipsteps(PipStepsBB, countPipstepsBB, parsedPipStepsBB);
    
   double lot_min  = MarketInfo(Symbol(),MODE_MINLOT);
   double lot_max  = MarketInfo(Symbol(),MODE_MAXLOT);
   Print("[Helpful info] Lot min: ", lot_min, " Lot max: ", lot_max);
    
   return(INIT_SUCCEEDED);
  }

double PointValue()
{
   double tickSize    = MarketInfo(Symbol(), MODE_TICKSIZE);
   double tickValue   = MarketInfo(Symbol(), MODE_TICKVALUE);
   double point       = MarketInfo(Symbol(), MODE_POINT);
   double ticksPerPoint = tickSize / point;
   double pointValue = tickValue / ticksPerPoint;
   
   return pointValue;
}

bool IsThisANewCandle()
{
   static datetime bartime=0;     //-- last time you've seen a new bar
   datetime dt=iTime(_Symbol,_Period,0);
   if(dt==bartime)
      return false;               //-- no new bar
   bartime=dt;                    //-- a new bar opened, great
   return true;
}

// Function to Determine Tick Point Value in Account Currency
double dblTickValue()
{
   return( MarketInfo(Symbol(), MODE_TICKVALUE ) );
}
        

// Function to Determine Pip Point Value in Account Currency
double dblPipValue()
{
   double dblCalcPipValue = dblTickValue();
   double digits = MarketInfo(Symbol(), MODE_DIGITS);
   
   if(digits == 3 || digits == 5)
   {
      dblCalcPipValue *= 10;
   }
   
   return dblCalcPipValue;
}

double GetPipSize()
{
   //return Point();
   //double CurrentPoint = MarketInfo(Symbol(), MODE_POINT);
   //Print(PointValue()," ", Point()," ", CurrentPoint, " ", dblPipValue());
   
   //return PointValue()*(Digits%2==1 ? 10 : 1) * PipScalar; // for forex only
   return Point()*(Digits%2==1 ? 10 : 1) * PipScalar; // for forex only
   //return 0.0001;
   //return 1;
}

double GetFirstOrderVolume()
{
   double lot_min  = MarketInfo(Symbol(),MODE_MINLOT);
   double lot_max  = MarketInfo(Symbol(),MODE_MAXLOT);
   
   double FirstOrderVolume = FirstVolume;
   if(FirstOrderVolume < lot_min)
   {
      FirstOrderVolume=lot_min;
   }
   if(FirstOrderVolume > lot_max)
   {
      FirstOrderVolume=lot_max;
   }
   
   return FirstOrderVolume;
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
   double lot_min  = MarketInfo(Symbol(),MODE_MINLOT);
   double lot_max  = MarketInfo(Symbol(),MODE_MAXLOT);
   double lot_step = MarketInfo(Symbol(),MODE_LOTSTEP);
   double contract = MarketInfo(Symbol(),MODE_LOTSIZE);

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
   
   double accountMargin = AccountFreeMargin();
   double lotMultiplier = 0;
   
   if(accountMargin > PercentIncreaseThreshold)
   {
      //lotMultiplier = ((accountMargin) / PercentIncreaseThreshold100)* risk *AccountLeverage()/contract;
      lotMultiplier = ((accountMargin) / PercentIncreaseThreshold100) * risk;
      //lotMultiplier = ((accountMargin * risk) / PercentIncreaseThreshold100);
   }
   else{
      //lotMultiplier = ((accountMargin) / PercentIncreaseThreshold) * risk *AccountLeverage()/contract;
      lotMultiplier = ((accountMargin) / PercentIncreaseThreshold) * risk;
      //lotMultiplier = ((accountMargin * risk) / PercentIncreaseThreshold);
   }

   //Print("Account free margin = ",AccountFreeMargin());
   
   /*
   double pointValue = PointValue();
   double riskAmount = accountMargin * risk;
   //double riskPoints = riskAmount / (pointValue * volume);
   double riskPoints = 50;
   double riskLots = riskAmount / (pointValue * riskPoints);
   
   
   double OneLotMargin = MarketInfo(Symbol(),MODE_MARGINREQUIRED);
   double lotMM = (accountMargin/OneLotMargin)*risk;
   double LotStep = MarketInfo(Symbol(),MODE_LOTSTEP);
   lotMM = NormalizeDouble(lotMM/LotStep,0)*LotStep;
   //lotMultiplier = lotMM / 100000;
   
   double lotStep = (accountMargin / 10000);
   //lotMultiplier = lotMultiplier * lotStep;
   
   Print(lotMultiplier);
   */
   
   double FirstOrderVolume = GetFirstOrderVolume();
   
   double volume = ((lotMultiplier + FirstOrderVolume) * MathPow(VolumeExponent, PowerOffset + (PowerFactor * countPositions)));
   
   //volume = NormalizeDouble(volume, 2);
   if(volume < lot_min) volume=lot_min;
   if(volume > lot_max) volume=lot_max;
   
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
      for(int p=0; p < countPipstepsRW;p++)
      {
         double pipStep = parsedPipStepsRW[p];
         
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
            if(lastopenprice != 0 && Ask < (lastopenprice - pipStep * GetPipSize()) && lastTradeTime != now)
              {
               executePipStep = true;
              }
           }
         else
           {
            if(lastopenprice != 0 && Bid > (lastopenprice + pipStep * GetPipSize()) && lastTradeTime != now)
              {
               executePipStep = true;
              }
           }
         
          if(executePipStep)
          {
             double price = ExecuteMarketOrder(tradeType);
             
             if(ApplyHedging)
             {
                int numOpenSell;
                int numOpenBuy;
               
                GetNumOpenPositions(numOpenBuy, numOpenSell);
             
                if (tradeType == BL_BUY && (numOpenSell < MaxOpenSell))
                {
                  ExecuteMarketOrder(BL_SELL);
                }
                else if (tradeType == BL_SELL && (numOpenBuy < MaxOpenBuy))
                {
                  ExecuteMarketOrder(BL_BUY);
                }
             }
             
             firstOrder = false;
            
             
             if (tradeType == BL_BUY)
             {
                 LastBuyTradeTime = now;
                 lastBuyPrice = price;
             }
             else if (tradeType == BL_SELL)
             {
                 LastSellTradeTime = now;
                 lastSellPrice = price;
             }
          }
      }
   }
}

void ProcessOrder_BB(BL_TRADE_TYPE tradeType)
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
      for(int p=0; p < countPipstepsBB;p++)
      {
         double pipStep = parsedPipStepsBB[p];
 
          
          double lastopenprice = GetLastOpenPrice(tradeType);
          lastopenprice = (tradeType == BL_BUY) ? lastBuyPrice : lastSellPrice;
          
          datetime lastTradeTime = (tradeType == BL_BUY) ? LastBuyTradeTime : LastSellTradeTime;
  
          
          bool executePipStep = false;
          
          if(tradeType == BL_BUY)
           {
            if(lastopenprice != 0 && Ask < (lastopenprice - pipStep * GetPipSize()) && lastTradeTime != now)
              {
               executePipStep = true;
              }
           }
         else
           {
            if(lastopenprice != 0 && Bid > (lastopenprice + pipStep * GetPipSize()) && lastTradeTime != now)
              {
               executePipStep = true;
              }
           }
          
          if(executePipStep)
          {
            //Print("TEST3");
             double price = ExecuteMarketOrder(tradeType);
             
             if(ApplyHedging)
             {
                int numOpenSell;
                int numOpenBuy;
               
                GetNumOpenPositions(numOpenBuy, numOpenSell);
             
                if (tradeType == BL_BUY && (numOpenSell < MaxOpenSell))
                {
                  ExecuteMarketOrder(BL_SELL);
                }
                else if (tradeType == BL_SELL && (numOpenBuy < MaxOpenBuy))
                {
                  ExecuteMarketOrder(BL_BUY);
                }
             }
             
             firstOrder = false;
             
             if (tradeType == BL_BUY)
             {
                 LastBuyTradeTime = now;
                 lastBuyPrice = price;
             }
             else if (tradeType == BL_SELL)
             {
                 LastSellTradeTime = now;
                 lastSellPrice = price;
             }
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

void DoBuy_BB()
{
   int numOpenSell;
   int numOpenBuy;
   
   GetNumOpenPositions(numOpenBuy, numOpenSell);

   if(numOpenBuy < MaxOpenBuy)
   {
      ProcessOrder_BB(BL_BUY);
   }
}

void DoSell_BB()
{
   int numOpenSell;
   int numOpenBuy;
   
   GetNumOpenPositions(numOpenBuy, numOpenSell);
   
   if(numOpenSell < MaxOpenSell)
   {
      ProcessOrder_BB(BL_SELL);
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
   double FirstOrderVolume = GetFirstOrderVolume();

   double targetProfit = FirstOrderVolume * TakeProfit * GetPipSize() * 1000000 ; // * 10000000
   //double targetProfit = FirstOrderVolume * TakeProfit * GetPipSize();// * Point(); // Symbol.PipSize
   //double targetProfit = FirstOrderVolume * TakeProfit * Point();
   //double targetProfit = FirstOrderVolume * TakeProfit;
   
   double targetLoss = FirstOrderVolume * StopLoss * GetPipSize() * 1000000;
   
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
      
      //targetProfit = FirstOrderVolume * Ask + TakeProfit * GetPipSize();
      averageBuyNetProfit = buyNetProfit / numOpenBuy;
      
      //Print("averageBuyNetProfit: ", averageBuyNetProfit);
      //Print("targetProfit: ", targetProfit);
      
      if(averageBuyNetProfit >= targetProfit)
      {
         Print("Closing BUY trades - AT PROFIT");
         CloseTrades(BL_BUY);
      }
      if(StopLossEnabled && averageBuyNetProfit <= -targetLoss)
      {
         Print("Closing BUY trades - AT LOSS");
         CloseTrades(BL_BUY);
      }
   }
   
   if(numOpenSell > 0)
   {
      
      //targetProfit = FirstOrderVolume * Bid - TakeProfit * GetPipSize();
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
         Print("Closing SELL trades - AT PROFIT");
         CloseTrades(BL_SELL);
      }
      if(StopLossEnabled && averageSellNetProfit <= -targetLoss)
      {
         Print("Closing SELL trades - AT LOSS");
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
               DoBuy_BB();
           }
           else
           {
               DoSell_BB();
           }
       }
       else if (Close[2] < BB_LOWER2)//boll.Bottom.Last(2))
       {
           if (Close[1] < BB_LOWER1)//boll.Bottom.Last(1))
           {
               DoSell_BB();
           }
           else
           {
               DoBuy_BB();
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

void CloseLosingTrades()
{
   bool result=false;
   double profit;
   
   //for(int i=0;i<OrdersTotal();i++)
   for(int i=OrdersTotal()-1; i>=0; i--)
   {
       if(OrderSelect(i, SELECT_BY_POS, MODE_TRADES))
       {
          if(OrderMagicNumber() != MagicNumber) break;
       
          if(OrderType()==OP_BUY  && OrderSymbol() == Symbol())
          {
            profit = OrderProfit() + OrderCommission() + OrderSwap();
          
            if(profit < 0)
            {
               result = OrderClose(OrderTicket(), OrderLots(), Bid, Slippage, clrYellow);
            }
            
          }
          if(OrderType()==OP_SELL && OrderSymbol() == Symbol())
          {
            profit = OrderProfit() + OrderCommission() + OrderSwap();
          
            if(profit < 0)
            {
               result = OrderClose(OrderTicket(), OrderLots(), Ask, Slippage, clrYellow);
            }
          }
       }
   }
}

bool DrawdownMonitor()
{
   if(!ApplyAntiDrawdown) return true;
   
   double equityPercent = (1 - AccountEquity() / AccountBalance()) * 100;
   double drawdownPercent = NormalizeDouble(DrawdownPercent,2);
   bool result = true;
   
   if(equityPercent > drawdownPercent && !waiting)
   {
      //Print("Drawdown too much, closing orders");
      //CloseLosingTrades();
      waiting = true;
      
      Print("DDDDDDDDDDDDDDDDDDDDDDD Drawdown too much, waiting ", WaitAfterDrawdownPips, " pips for better opportunity");
   }
   if(equityPercent > drawdownPercent)
   {
      result = false;
   }
   
   if(waiting)
   {      
      //double current_ask = MarketInfo(Symbol(),MODE_ASK);
      //double current_bid = MarketInfo(Symbol(),MODE_BID);
      double current_ask = Ask;
      double current_bid = Bid;
      
      double waitPoints = WaitAfterDrawdownPips * GetPipSize(); // Point() GetPipSize()
      
      if(Ask >= Ask + waitPoints)
      {
         Print("######################### Price has moved up ",WaitAfterDrawdownPips," pips to ask: ",current_ask," from ", current_ask + waitPoints);
         
         waiting = false;
         
         result = true;
      }
      else if(Bid <= Bid - waitPoints)
      {
         Print("######################### Price has moved up ",WaitAfterDrawdownPips," pips to bid: ",current_bid," from ", current_bid - waitPoints);
         
         waiting = false;
         
         result = true;
      }
   }
   return result;
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
   IsNewBar=IsThisANewCandle();
   
//*************************************************************//
   if(IsNewBar && DrawdownMonitor())
   {
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