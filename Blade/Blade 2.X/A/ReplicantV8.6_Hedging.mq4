#property copyright   "Copyright 2020, Metamorphic."
#property link        "https://jumpinalake.com"
#property version     "8.7"
#property description "Expert Advisor"
#property strict

#include <stdlib.mqh>

enum BL_TRADE_TYPE
{
   BL_BUY,
   BL_SELL
};

enum BL_VOLUME
{
   BL_BASIC_VOLUME = 0,
   BL_NORMALISED_VOLUME = 1
};

enum BL_EXP
{
   BL_LINEAR = 0,
   BL_EXPONENTIAL = 1,
   BL_SMALL_EXPONENTIAL = 2,
};

//************************************************************************************************/
//*                                                                                              */
//************************************************************************************************/
//input string a = "$$$$$$$$$$$$$$ Basic Settings $$$$$$$$$$$$$$";
input double      TakeProfit                 = 0.75;              // Take Profit (in pips)
input double      StopLoss                   = 1000;           // Stop Loss (in pips)
input bool        StopLossEnabled            = false;          // Enable silent Stop Loss

input bool        UseTightStop               = false;
input bool        UseTrailing                = false; 
input double      ScalpPips                  = 3;
input double      TrailingAct                = 6;    
input double      TrailingStep               = 3; 

input int         MagicNumberS                = 666;            // Advisor Magic Number Sell
input int         MagicNumberB                = 666;            // Advisor Magic Number Buy
input int         Slippage                   = 3;              // Slippage (in pips)
//input double      MaxSpread                  = 3;
input int         MaxOpenBuy                 = 35;             // Maximum number of open buy positions
input int         MaxOpenSell                = 35;             // Maximum number of open sell positions
//input double      PipStep                    = 175;            // Pip step (random walk)
//input double      PipStepBB                  = 175;            // Pip step (bolinger)
//input string      PipStepsRW                 = "235";
//input string      PipStepsBB                 = "35";
input double      PipStepsRW                 = 235;
input double      PipStepsBB                 = 175;
input double      PipStepsSTO                = 1;
input double      PipScalar                  = 1.0;            // Pip scalar

//input string b = "$$$$$$$$$$$$$$ Bollinger Band Settings $$$$$$$$$$$$$$";
input int         BBPeriod                   =  35;            // Bolinger band period
input double      BBDeviation                =  2;             // Bollinger band deviation (stddev)
input int         BBShift                    =  0;             // Bollinger band shift
input ENUM_APPLIED_PRICE BBPriceMode         = PRICE_CLOSE;    // Bollinger band price mode
input int         MaxConsecutiveBuys         = 35;             // Max consecutive number of buy positions
input int         MaxConsecutiveSells        = 35;             // Max consecutive number of sell positions

//input string c = "$$$$$$$$$$$$$$ Advanced Setup $$$$$$$$$$$$$$";
input double      FirstVolume                = 0.1;            // First order volume scalar
input double      VolumeExponent             = 1.7;            // Volume exponent
input double      NumGroups                  = 1;              // Number of groups
input BL_VOLUME   VolumeFlags                = BL_BASIC_VOLUME; // Normalise volume lot sizes
input double      PowerFactor                = 1.0;            // Power factor
input double      PowerOffset                = 0.0;            // Power offset
input double      PercentIncreaseThreshold   = 1000000;         // Percent increase threshold (Free margin denominator)
input double      PercentIncreaseThreshold100 = 1000000;       // Percent increase threshold (Free margin used above $100K)
input double      RiskPercent                = 125.0;           // Risk percent (Risk managment)
//input double      RiskPercentB               = 100.0;           // Risk percent B (Risk managment)
input int         InitialTrades              = 50;             // Initial number of trades for initial lower risk
input double      InitialVolumeExponent      = 1.7;            // Initial volume exponent
input double      InitialNumGroups           = 3;              // Initial number of groups for initial lower risk
input double      DrawdownNumGroups          = 3;              // Drawdown number of groups boost
input int         EquityInterval             = 3000;          // Equity increase threshold
input double      EquityExponent             = 1.05;            // Equity exponent
input double      RiskRatio                  = 1.0;            // Risk ratio
input int         InitialNumTradesLinear     = 0;              // Initial linearity number of trades
input double      LotIncreasePercent         = 25.0;           // Increase lot size by percentage
input double      HedgeLotScalar             = 1.7;            // Hedge lot scalar
input double      MaxExp                     = 100.0;          // Max Exp
input double      MaxLot                     = 100;             // Max Lot

//input string f = "$$$$$$$$$$$$$$ Anti Drawdown $$$$$$$$$$$$$$";
input double      DrawdownPercent            = 25;             // Acceptable drawdown percentage %
input double      WaitAfterDrawdownPips      = 1;              // Wait duration (in pips) after drawdown
input bool        ApplyFastWait              = true;          // Fast recovery from drawdown.

input int         KPeriod                    =  50; // KPeriod
input int         Slowing                    =   7; // Slowing
input int         DPeriod                    =   7; // DPeriod
input ENUM_MA_METHOD MAMethod                =   3; // MAMethod
input ENUM_STO_PRICE PriceField              =   1; // PriceField
input int         overBought                 = 10;  // overBought short
input int         overSold                   = 5;  // overSold short

input int         FastK                      = 8;   // FastK
input int         SlowK                      = 3;   // SlowK
input int         SlowD                      = 3;   // SlowD

input int         MA_FASTMA                     = 7;
input int         MA_SLOWMA                     = 25;
input ENUM_MA_METHOD       MA_MA_METHOD         = MODE_EMA;
input ENUM_APPLIED_PRICE   MA_PRICE_METHOD      = PRICE_CLOSE;

input int         WP_willperiod              = 50;
input double      WP_upperband               = -30;
input double      WP_lowerband               = -15;

input int         FI_forceu                  = 1;
//input int                  FI_ExtForcePeriod             = 13;
input ENUM_MA_METHOD       FI_ExtForceMAMethod           = 2;
input ENUM_APPLIED_PRICE   FI_ExtForceAppliedPrice      = 5;

input int         RSI_rsiu                   = 50;
input double      RSI_lowerband              = 25;
input double      RSI_upperband              = 70;

input int         MACD_fastmacd              = 25;
input int         MACD_slowmacd              = 100;
input int         MACD_signalmacd            = 50;

//input string e = "$$$$$$$$$$$$$$ Strategy $$$$$$$$$$$$$$";
input bool        ApplyBollingerBands        = false;           // Apply bollinger bands strategy
input bool        ApplyRandomWalk            = false;          // Apply random walk
input bool        ApplyStochasticDiNapoli    = false;          // Apply Stochastic
input bool        ApplyStochasticDiNapoli_v1 = false;          // Apply StochasticDiNapoli_v1
input bool        ApplyStochasticDiNapoli_v2 = false;          // Apply StochasticDiNapoli_v2
input bool        ApplyMACrossoverBol        = false;          // Apply MA Crossover
input bool        ApplyWillPeriod            = true;          // Apply Will Percentage Range strategy
input bool        ApplyForceIndex            = false;          // Apply Force Index strategy
input bool        ApplyRSI                   = true;          // Apply RSI strategy
input bool        ApplyMACD                  = true;          // Apply MACD strategy
input bool        ApplyHedging               = false;           // Apply hedging strategy
input bool        AlwaysHedge                = false;          // ALWAYS Apply hedging strategy
input bool        AlwaysProperHedge          = false;          // Apply proper hedging strategy (remember last volume)
input bool        AlwaysProperHedge2         = false;          // Apply 2nd proper hedging strategy (remember last order volume)
input bool        ApplyAntiDrawdown          = false;           // Apply anti-drawdown
input BL_EXP      ApplyExponential           = BL_LINEAR;      // Apply exponential or linear lot sizes
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

datetime LastBuyTradeTime2;
datetime LastSellTradeTime2;
double lastBuyPrice2;
double lastSellPrice2;

datetime LastBuyTradeTime3;
datetime LastSellTradeTime3;
double lastBuyPrice3;
double lastSellPrice3;

double lastPriceAsk = 0;
double lastPriceBid = 0;

int countConsecutiveBuys = 0;
int countConsecutiveSells = 0;

int numTrades = 0;

BL_TRADE_TYPE lastOrderTradeType = BL_BUY;

bool IsNewBar;
double startprice = 0;
bool waiting = false;
bool isDrawdown = false;

double startingEquity;

double TrailPrice;

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
   
   parsedPipStepsRW[0] = PipStepsRW;
   parsedPipStepsBB[0] = PipStepsBB;
   countPipstepsRW = 1;
   countPipstepsBB = 1;
   
   //ParsePipsteps(PipStepsRW, countPipstepsRW, parsedPipStepsRW);
   //ParsePipsteps(PipStepsBB, countPipstepsBB, parsedPipStepsBB);
    
   double lot_min  = MarketInfo(Symbol(),MODE_MINLOT);
   double lot_max  = MarketInfo(Symbol(),MODE_MAXLOT);
   Print("[Helpful info] Lot min: ", lot_min, " Lot max: ", lot_max);
   
   ShowOpenTrades();
   
   startingEquity = AccountEquity();
       
   return(INIT_SUCCEEDED);
  }

double PointValue()
{
   double tickSize    = MarketInfo(Symbol(), MODE_TICKSIZE);
   double tickValue   = MarketInfo(Symbol(), MODE_TICKVALUE);
   double point       = MarketInfo(Symbol(), MODE_POINT);
   double digits      = MarketInfo(Symbol(), MODE_DIGITS);
   double ticksPerPoint = tickSize / point;
   double pointValue = tickValue / ticksPerPoint;
   
   return pointValue;
}

datetime NewCandleTime = TimeCurrent();

bool IsThisANewCandle()
{
   if (NewCandleTime == iTime(Symbol(), _Period, 0)) 
   {
      return false;
   }
   else
   {
      NewCandleTime = iTime(Symbol(), _Period, 0);
      return true;
   }
   /*
   static datetime bartime=0;     //-- last time you've seen a new bar
   datetime dt=iTime(_Symbol,_Period,0);
   if(dt==bartime)
      return false;               //-- no new bar
   bartime=dt;                    //-- a new bar opened, great
   return true; */
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

double GetPipScalar()
{
   if(Digits==1)
   {
      return 0.0001; // for CFDs
   }
   else if(Digits==2)
   {
      return 0.01; // for JPY
   }
   return 1.0;
}

double GetPipSize()
{
   double digits      = MarketInfo(Symbol(), MODE_DIGITS);
   //Print("digits: ", digits);
   
   //if(digits==5 || digits==3)
   /*if(Digits==1)
   {
      //Print("DIGITS: ", digits);
      return Point() * PipScalar * 0.0001; // for CFDs
   }
   else if(Digits==2)
   {
      //Print("DIGITS JPY: ", digits);
      return Point() * PipScalar * 0.01; // for JPY
   }
   else*/ 
   if(Digits%2==1)
   //if(Digits==3 || Digits == 5)
   {
      //Print("DIGITS: ", digits);
      return Point() * 10 * PipScalar; // for forex only
   }
   else 
   {
      return Point() * 1 * PipScalar; // for forex only
   }
   //return Point();
   //double CurrentPoint = MarketInfo(Symbol(), MODE_POINT);
   //Print(PointValue()," ", Point()," ", CurrentPoint, " ", dblPipValue());
   
   //return PointValue()*(Digits%2==1 ? 10 : 1) * PipScalar; // for forex only
   //return Point()*(Digits%2==1 ? 10 : 1) * PipScalar; // for forex only
}

double GetFirstOrderVolume()
{
   double lot_min  = MarketInfo(Symbol(),MODE_MINLOT);
   double lot_max  = MarketInfo(Symbol(),MODE_MAXLOT);
   
   double FirstOrderVolume = FirstVolume;
   
   if(FirstOrderVolume == 0)
   {
      FirstOrderVolume = lot_min;
   }
   
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

int GetSlippage()
{
   int slippage = Slippage;
   
   if(Slippage <= 0)
   {
      slippage = (int)MarketInfo(OrderSymbol(), MODE_SPREAD);
   }
   
   return slippage;
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
          if(OrderType()==OP_BUY  && OrderSymbol() == Symbol() && OrderMagicNumber() == MagicNumberB)
          {
            numOpenBuy ++;
          }
          if(OrderType()==OP_SELL && OrderSymbol() == Symbol() && OrderMagicNumber() == MagicNumberS)
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
          if(OrderType()==OP_BUY  && OrderSymbol() == Symbol() && OrderMagicNumber() == MagicNumberB)
          {
            if(tradeType == BL_BUY)
            {
               count ++;
            }
          }
          if(OrderType()==OP_SELL && OrderSymbol() == Symbol() && OrderMagicNumber() == MagicNumberS)
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

int GetCountWinningOpenPositions(BL_TRADE_TYPE tradeType)
{
   int count = 0;
   //for(int i=0;i<OrdersTotal();i++)
   for(int i=OrdersTotal()-1; i>=0; i--)
   {
       if(OrderSelect(i, SELECT_BY_POS, MODE_TRADES))
       {       
          double profit = OrderProfit() + OrderCommission() + OrderSwap();
          
          if(profit > 0)
          {
             if(OrderType()==OP_BUY  && OrderSymbol() == Symbol() && OrderMagicNumber() == MagicNumberB)
             {
               if(tradeType == BL_BUY)
               {
                  count ++;
               }
             }
             if(OrderType()==OP_SELL && OrderSymbol() == Symbol() && OrderMagicNumber() == MagicNumberS)
             {
               if(tradeType == BL_SELL)
               {
                  count ++;
               }
             }
          }
       }
   }
   return count;
}

double GetAverageNetProfit(BL_TRADE_TYPE tradeType)
{
   int count = 0;
   double netProfit = 0;
   //for(int i=0;i<OrdersTotal();i++)
   for(int i=OrdersTotal()-1; i>=0; i--)
   {
       if(OrderSelect(i, SELECT_BY_POS, MODE_TRADES))
       {      
          double profit = OrderProfit() + OrderCommission() + OrderSwap();

          if(OrderType()==OP_BUY  && OrderSymbol() == Symbol() && OrderMagicNumber() == MagicNumberB)
          {
            if(tradeType == BL_BUY)
            {
               netProfit += profit;
               count ++;
            }
          }
          if(OrderType()==OP_SELL && OrderSymbol() == Symbol() && OrderMagicNumber() == MagicNumberS)
          {
            if(tradeType == BL_SELL)
            {
               netProfit += profit;
               count ++;
            }
          }
          
       }
   }
   if(count >0)
   {
      return (netProfit / (double)count);
   }
   return 0;
}

double NormalizeVolume(const double volume)
{
   double volumeStep=SymbolInfoDouble(_Symbol,SYMBOL_VOLUME_STEP);
   double volumeMin=SymbolInfoDouble(_Symbol,SYMBOL_VOLUME_MIN);
   double volumeMax=SymbolInfoDouble(_Symbol,SYMBOL_VOLUME_MAX);
   double volumeNormalized=int(volume / volumeStep) * volumeStep;
   return(volumeNormalized<volumeMin?0.0:(volumeNormalized>volumeMax?volumeMax:volumeNormalized));
}

double CalculateVolume(BL_TRADE_TYPE tradeType, bool hedgingPosition)
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
   
   
   return ExponentialVolume(tradeType, hedgingPosition, countPositions);
}

double ExponentialVolume(BL_TRADE_TYPE tradeType, bool hedgingPosition, int countPreviousPositions)
{
   //return existingVolume * HedgeLotScalar;
   
   double lot_min  = MarketInfo(Symbol(),MODE_MINLOT);
   double lot_max  = MarketInfo(Symbol(),MODE_MAXLOT);
   double lot_step = MarketInfo(Symbol(),MODE_LOTSTEP);
   double contract = MarketInfo(Symbol(),MODE_LOTSIZE);

   int countPositions = countPreviousPositions; // Keep previous volume
   
   double risk = RiskPercent / 100.0;
   //double riskB = RiskPercentB / 100.0;
   
   double  ab  = AccountBalance();
   //risk = MathSqrt(RiskMultiplier * ab)/ab; // ~const rate.
   //risk = MathSqrt(RiskMultiplier * ab * MathPow( 1 - risk, OrdersTotal() ));
   
   double numGroups;
   
   if(numTrades < InitialTrades)
   {
      numGroups = InitialNumGroups;
   }
   else if(isDrawdown)
   {
      numGroups = DrawdownNumGroups;
   }
   else
   {
      numGroups = NumGroups;
   }
   
   //double accountMargin = AccountFreeMargin();
   double accountMargin = AccountFreeMargin();
   accountMargin = AccountEquity();
   //accountMargin = AccountBalance();
   
   double lotMultiplier = 0;
   
   if(accountMargin > PercentIncreaseThreshold)
   {
      lotMultiplier = ((accountMargin) / PercentIncreaseThreshold100) * risk;
   }
   else{
      lotMultiplier = ((accountMargin) / PercentIncreaseThreshold) * risk;
   }
   
   double FirstOrderVolume = GetFirstOrderVolume();
   if(hedgingPosition)
   {
      FirstOrderVolume *= HedgeLotScalar;
   }
   
   double ae;
   
   if(AccountEquity() > startingEquity)
   {  
      //double ee = (AccountEquity() * (LotIncreasePercent / 100.0));
      ae = (((AccountEquity()) / (EquityInterval))) * RiskRatio;
   }
   else 
   {
       ae = 1.0;
   }
   if(ae <= 0)
   {
      ae = 1;
   }

   ae = MathPow(EquityExponent, ae);
   

   
   if(isDrawdown)
   {
      ae = 1;
   }
   
   if(ae >= MaxExp)
   {
      ae = MaxExp;
   }

   double volume = FirstOrderVolume;
   double volumeExp = VolumeExponent;
   
   if(numTrades < InitialTrades)
   {
      volumeExp = InitialVolumeExponent;
   }
   
   
   if(ApplyExponential == BL_LINEAR)
   {
      volume =  ((lotMultiplier + FirstOrderVolume) * MathPow(volumeExp, PowerOffset + ((PowerFactor * countPositions) / numGroups)));
   }
   else if (ApplyExponential == BL_EXPONENTIAL)
   {
      volume =  ae * ((lotMultiplier + FirstOrderVolume) * MathPow(volumeExp, PowerOffset + ((PowerFactor * (countPositions)) / numGroups)));
   }
   else if (ApplyExponential == BL_SMALL_EXPONENTIAL)
   {
      volume =  ae * FirstOrderVolume * MathPow(volumeExp, PowerOffset + ((PowerFactor * (countPositions)) / numGroups));   
      //volume =  ae * ((((FirstOrderVolume * ((1+countPositions)/ numGroups))))) ;
   }
   
   if(numTrades <= InitialNumTradesLinear)
   {
      // Linear
      volume =  ((lotMultiplier + FirstOrderVolume) * MathPow(volumeExp, PowerOffset + ((PowerFactor * countPositions) / numGroups)));
   }
   else
   {
      double ee = (volume * (LotIncreasePercent / 100.0));
      
      //if(isDrawdown)
      //{
         volume += ee;   
      //}
   }
   //volume =  ((lotMultiplier + FirstOrderVolume) * MathPow(VolumeExponent, PowerOffset + ((PowerFactor * countPositions) / numGroups)));
   //volume =  ae * ((lotMultiplier + FirstOrderVolume) * MathPow(VolumeExponent, PowerOffset + ((PowerFactor * countPositions) / numGroups)));
  
   if(volume >= MaxLot)
   {
      volume = MaxLot;
   }
   
   //volume = NormalizeDouble(volume, 2);
   if(volume < lot_min) volume=lot_min;
   if(volume > lot_max) volume=lot_max;
   
   if(VolumeFlags == BL_NORMALISED_VOLUME)
   {
      volume = NormalizeVolume(volume);
   }
   //Print(volume);
   
   volume = NormalizeDouble(volume, Digits());
  
   
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

 double EntryLoss(BL_TRADE_TYPE tradeType, double lots) 
 {
   // This method calculates the profit or loss of a position for the current currency
   double result;
   //double priceChange = MathAbs(entry - exit);
   double pointSize = MarketInfo(Symbol(), MODE_POINT);
   double tickValue = MarketInfo(Symbol(), MODE_TICKVALUE);
   double price = GetPrice(tradeType);

   //price = MathAbs(Open[1] - Close[0]);
   
   result = price * lots * (1 / pointSize) * tickValue;
  
   
   //Print("###### Entry Cost: ", result, " lots: ", lots, " price: ", price);
 
   
   return result;
 }

double GetPrice(BL_TRADE_TYPE tradeType)
{
   double price = 0;
   
   if(tradeType == BL_BUY)
   {
      price = MarketInfo(Symbol(), MODE_ASK);
      //price = NormalizeDouble(Ask, Digits());
   }
   else if(tradeType == BL_SELL)
   {
      price = MarketInfo(Symbol(), MODE_BID);
      //price = NormalizeDouble(Bid, Digits());
   }
   
   return price;
}

double GetClosingPrice(BL_TRADE_TYPE tradeType)
{
   double price = 0;
   
   if(tradeType == BL_BUY)
   {
      price = MarketInfo(Symbol(), MODE_BID);
      //price = NormalizeDouble(Bid, Digits());
   }
   else if(tradeType == BL_SELL)
   {
      price = MarketInfo(Symbol(), MODE_ASK);
      //price = NormalizeDouble(Ask, Digits());
   }
   
   return price;
}
double previousBuyVolume = 0;
double previousSellVolume = 0;
int previousBuyPositions = -1;
int previousSellPositions = -1;

double ExecuteMarketOrder(BL_TRADE_TYPE tradeType, double& volumeOut, double volumeIn, bool first)
{
   int ticketId = -1;
   double price = 0;
   double priceOrig = 0;
   double volume = 0;
   int nextPositions = 0;
   
   bool hedging = ApplyHedging; 
   
   if(AlwaysHedge && first)
   {
      hedging = true;
   }
   else if(AlwaysHedge && !first)
   {
      hedging = true;
   }
   if(ApplyHedging && !AlwaysHedge && first)
   {
      hedging = false;
   }
   if(ApplyHedging && !AlwaysHedge && !first)
   {
      hedging = true;
   }
   if(ApplyHedging && AlwaysProperHedge2)
   {
      hedging = true;
   }
   
   if(tradeType == BL_BUY && continueBuyTrading)
   {
      price = GetPrice(BL_BUY);
      priceOrig = price;
      
      if(volumeIn > 0 && AlwaysProperHedge)
      {
         volume = volumeIn;
      }
      else
      {
         volume = CalculateVolume(BL_BUY, false);
      }
      volume = CalculateVolume(BL_BUY, false);
      
      if(ApplyHedging && AlwaysProperHedge2 && previousSellPositions > -1)
      {
         //volume = ExponentialVolume(BL_BUY, false, previousSellPositions);
         //previousSellPositions = -1; 
      }
      
      if(ApplyHedging && AlwaysProperHedge2 && numTrades % 2 == 0 && previousSellPositions > -1)
      {
         volume = ExponentialVolume(BL_SELL, false, previousSellPositions);
      }
      
      //previousBuyVolume = volume;
      //previousBuyPositions = GetCountOpenPositions(BL_BUY);
      
      //nextPositions = previousSellPositions;
      
      ticketId = OrderSend(Symbol(), OP_BUY, volume, price, GetSlippage(), 0, 0, algoTitle, MagicNumberB, 0, clrGreen);

      if(ticketId < 0)
      {
         Print("OrderSend error #", GetLastError(), " ", ErrorDescription(GetLastError()));
      }
      
      if(hedging)
      {
         price = GetPrice(BL_SELL);
         
         if(!AlwaysProperHedge)
         {
            volume = CalculateVolume(BL_SELL, true);
         }
         
         if(ApplyHedging && previousBuyPositions > -1 && AlwaysProperHedge2)
         {
            //volume = ExponentialVolume(BL_SELL, true, previousBuyPositions);
            //previousBuyPositions = -1; 
         }
         else if(previousBuyPositions == -1 && AlwaysProperHedge2)
         {
            //volume = CalculateVolume(BL_SELL, true);
         }
         
         if(ApplyHedging && AlwaysProperHedge2 && numTrades % 2 == 0 && previousBuyPositions > -1)
         {
            volume = ExponentialVolume(BL_BUY, true, previousBuyPositions);
         }
         
         //volume = ExponentialVolume(BL_SELL, true, nextPositions);
         
         ticketId = OrderSend(Symbol(), OP_SELL, volume, price, GetSlippage(), 0, 0, algoTitle + " hedging", MagicNumberS, 0, clrRed);
   
         if(ticketId < 0)
         {
            Print("OrderSend error #", GetLastError(), " ", ErrorDescription(GetLastError()));
         }
      }
      
      previousBuyPositions = GetCountOpenPositions(BL_BUY) - 1;
   }
   else if(tradeType == BL_SELL && continueSellTrading)
   {
      price = GetPrice(BL_SELL);
      priceOrig = price;
      
      if(volumeIn > 0 && AlwaysProperHedge)
      {
         volume = volumeIn;
      }
      else
      {
         volume = CalculateVolume(BL_SELL, false);
      }
      volume = CalculateVolume(BL_SELL, false);
      
      if(ApplyHedging && AlwaysProperHedge2 && previousBuyPositions > -1)
      {
         //volume = ExponentialVolume(BL_BUY, false, previousBuyPositions);
         //previousBuyPositions = -1;
      }
      
      if(ApplyHedging && AlwaysProperHedge2 && numTrades % 2 == 0 && previousBuyPositions > -1)
      {
         volume = ExponentialVolume(BL_BUY, false, previousBuyPositions);
      }
      
      //previousSellVolume = volume;
      //previousSellPositions = GetCountOpenPositions(BL_SELL);
      
      //nextPositions = previousBuyPositions;
      
      ticketId = OrderSend(Symbol(), OP_SELL, volume, price, GetSlippage(), 0, 0, algoTitle, MagicNumberS, 0, clrRed);
      
      if(ticketId < 0)
      {
         Print("OrderSend error #", GetLastError());
      }
      
      if(hedging)
      {
         price = GetPrice(BL_BUY);
         if(!AlwaysProperHedge)
         {
            volume = CalculateVolume(BL_BUY, true);
         }
         
         
         if(previousSellPositions > -1 && AlwaysProperHedge2)
         {
            //volume = ExponentialVolume(BL_BUY, true, previousSellPositions);
            //previousBuyPositions = -1;
         }
         else if(previousSellPositions == -1 && AlwaysProperHedge2)
         {
            //volume = CalculateVolume(BL_BUY, true);
         }
         
         if(ApplyHedging && AlwaysProperHedge2 && numTrades % 2 == 0 && previousSellPositions > -1)
         {
            volume = ExponentialVolume(BL_SELL, true, previousSellPositions);
         }
         
         //volume = ExponentialVolume(BL_BUY, true, nextPositions);
         
         ticketId = OrderSend(Symbol(), OP_BUY, volume, price, GetSlippage(), 0, 0, algoTitle + " hedging", MagicNumberB, 0, clrGreen);
   
         if(ticketId < 0)
         {
            Print("OrderSend error #", GetLastError());
         }
      }
      previousSellPositions = GetCountOpenPositions(BL_SELL) - 1;
   }
   
   lastOrderTradeType = tradeType;
   numTrades ++;
   volumeOut = volume;
   
   return priceOrig;
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
          if(OrderType()==OP_BUY  && OrderSymbol() == Symbol() && OrderMagicNumber() == MagicNumberB)
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
          if(OrderType()==OP_SELL && OrderSymbol() == Symbol() && OrderMagicNumber() == MagicNumberS)
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
          if(OrderType()==OP_BUY  && OrderSymbol() == Symbol() && OrderMagicNumber() == MagicNumberB)
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
          if(OrderType()==OP_SELL && OrderSymbol() == Symbol() && OrderMagicNumber() == MagicNumberS)
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
      if(OrderSymbol()==Symbol() && OrderCloseTime()==0)
      {
         if(OrderType()==OP_BUY && OrderMagicNumber() == MagicNumberB){
            lastbuyopenprice=OrderOpenPrice();
         }
         if(OrderType()==OP_SELL && OrderMagicNumber() == MagicNumberS){
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
   double volume = 0;
//ExecuteMarketOrder(tradeType);return;

   int numPositions = GetCountOpenPositions(tradeType);

///*

   bool buyProfit = (tradeType == BL_BUY) && (Close[1] > Close[2]);
   bool sellCheap = (tradeType == BL_SELL) && (Close[2] > Close[1]);

   //bool buyProfit = (tradeType == BL_BUY) && (Close[0] > Close[1]);
   //bool sellCheap = (tradeType == BL_SELL) && (Close[1] > Close[0]);

   datetime now = Time[0];

   if (numPositions == 0 && (buyProfit || sellCheap))//MarketSeries.Close.Last(1) > MarketSeries.Close.Last(2))
   //if (numPositions == 0)
   {
      //Print("TEST1");
       double price = ExecuteMarketOrder(tradeType, volume, 0, true);
       
       firstOrder = false;
       //datetime now = GetTimeLastOrder(ticketId);
       //OrderSelect((OrdersTotal() -1 ), SELECT_BY_POS);
       //datetime now = GetTimeLastOrder(OrderTicket());
       
       if (tradeType == BL_BUY)
       {
           LastBuyTradeTime2 = now;//MarketSeries.OpenTime.Last(0);
           lastBuyPrice2 = price;
       }
       else if (tradeType == BL_SELL)
       {
           LastSellTradeTime2 = now;//MarketSeries.OpenTime.Last(0);
           lastSellPrice2 = price;
       }
   }
   if (numPositions > 0) // */
   {  
      for(int p=0; p < countPipstepsRW;p++)
      {
         double pipStep = parsedPipStepsRW[p];
         
         //Print("TEST2");
          
          double lastopenprice = GetLastOpenPrice(tradeType);
          lastopenprice = (tradeType == BL_BUY) ? lastBuyPrice2 : lastSellPrice2;
          
          datetime lastTradeTime = (tradeType == BL_BUY) ? LastBuyTradeTime2 : LastSellTradeTime2;
   
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
             double price = ExecuteMarketOrder(tradeType, volume, 0, false);
             
             if (tradeType == BL_BUY)
             {
                 LastBuyTradeTime2 = now;
                 lastBuyPrice2 = price;
             }
             else if (tradeType == BL_SELL)
             {
                 LastSellTradeTime2 = now;
                 lastSellPrice2 = price;
             }
            
             firstOrder = false;
          }
      }
   }
}

void ProcessOrder_STO(BL_TRADE_TYPE tradeType)
{
   double volume = 0;
//ExecuteMarketOrder(tradeType);return;

   int numPositions = GetCountOpenPositions(tradeType);

///*

   bool buyProfit = (tradeType == BL_BUY) && (Close[1] > Close[2]);
   bool sellCheap = (tradeType == BL_SELL) && (Close[2] > Close[1]);

   //bool buyProfit = (tradeType == BL_BUY) && (Close[0] > Close[1]);
   //bool sellCheap = (tradeType == BL_SELL) && (Close[1] > Close[0]);

   datetime now = Time[0];

   if (numPositions == 0 && (buyProfit || sellCheap))//MarketSeries.Close.Last(1) > MarketSeries.Close.Last(2))
   //if (numPositions == 0)
   {
      //Print("TEST1");
       double price = ExecuteMarketOrder(tradeType, volume, 0, true);
       
       firstOrder = false;
       //datetime now = GetTimeLastOrder(ticketId);
       //OrderSelect((OrdersTotal() -1 ), SELECT_BY_POS);
       //datetime now = GetTimeLastOrder(OrderTicket());
       
       if (tradeType == BL_BUY)
       {
           LastBuyTradeTime2 = now;//MarketSeries.OpenTime.Last(0);
           lastBuyPrice2 = price;
       }
       else if (tradeType == BL_SELL)
       {
           LastSellTradeTime2 = now;//MarketSeries.OpenTime.Last(0);
           lastSellPrice2 = price;
       }
   }
   if (numPositions > 0) // */
   {  
      double pipStep = PipStepsSTO;
      
      //Print("TEST2");
       
       double lastopenprice = GetLastOpenPrice(tradeType);
       lastopenprice = (tradeType == BL_BUY) ? lastBuyPrice2 : lastSellPrice2;
       
       datetime lastTradeTime = (tradeType == BL_BUY) ? LastBuyTradeTime2 : LastSellTradeTime2;

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
          double price = ExecuteMarketOrder(tradeType, volume, 0, false);
          
          if (tradeType == BL_BUY)
          {
              LastBuyTradeTime2 = now;
              lastBuyPrice2 = price;
          }
          else if (tradeType == BL_SELL)
          {
              LastSellTradeTime2 = now;
              lastSellPrice2 = price;
          }
         
          firstOrder = false;
       }
      
   }
}

void ProcessOrder_STO_SELL(BL_TRADE_TYPE tradeType)
{
   double volume = 0;
//ExecuteMarketOrder(tradeType,volume,0,true);return;

   int numPositions = GetCountOpenPositions(tradeType);

///*

   bool buyProfit = (tradeType == BL_BUY) && (Close[1] > Close[2]);
   bool sellCheap = (tradeType == BL_SELL) && (Close[2] > Close[1]);

   //bool buyProfit = (tradeType == BL_BUY) && (Close[0] > Close[1]);
   //bool sellCheap = (tradeType == BL_SELL) && (Close[1] > Close[0]);

   datetime now = Time[0];

   if (numPositions == 0 && (buyProfit || sellCheap))//MarketSeries.Close.Last(1) > MarketSeries.Close.Last(2))
   //if (numPositions == 0)
   {
      //Print("TEST1");
       double price = ExecuteMarketOrder(tradeType, volume, 0, true);
       
       firstOrder = false;
       //datetime now = GetTimeLastOrder(ticketId);
       //OrderSelect((OrdersTotal() -1 ), SELECT_BY_POS);
       //datetime now = GetTimeLastOrder(OrderTicket());
       
       if (tradeType == BL_BUY)
       {
           LastBuyTradeTime3 = now;//MarketSeries.OpenTime.Last(0);
           lastBuyPrice3 = price;
       }
       else if (tradeType == BL_SELL)
       {
           LastSellTradeTime3 = now;//MarketSeries.OpenTime.Last(0);
           lastSellPrice3 = price;
       }
   }
   if (numPositions > 0) // */
   {  
      double pipStep = PipStepsSTO;
      
      //Print("TEST2");
       
       double lastopenprice = GetLastOpenPrice(tradeType);
       lastopenprice = (tradeType == BL_BUY) ? lastBuyPrice3 : lastSellPrice3;
       
       datetime lastTradeTime = (tradeType == BL_BUY) ? LastBuyTradeTime3 : LastSellTradeTime3;

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
          double price = ExecuteMarketOrder(tradeType, volume, 0, false);
          
          if (tradeType == BL_BUY)
          {
              LastBuyTradeTime3 = now;
              lastBuyPrice3 = price;
          }
          else if (tradeType == BL_SELL)
          {
              LastSellTradeTime3 = now;
              lastSellPrice3 = price;
          }
         
          firstOrder = false;
       }
      
   }
}

void ProcessOrder_STO_BUY(BL_TRADE_TYPE tradeType)
{
   double volume = 0;
//ExecuteMarketOrder(tradeType,volume,0,true);return;

   int numPositions = GetCountOpenPositions(tradeType);

///*

   bool buyProfit = (tradeType == BL_BUY) && (Close[1] > Close[2]);
   bool sellCheap = (tradeType == BL_SELL) && (Close[2] > Close[1]);

   //bool buyProfit = (tradeType == BL_BUY) && (Close[0] > Close[1]);
   //bool sellCheap = (tradeType == BL_SELL) && (Close[1] > Close[0]);

   datetime now = Time[0];

   if (numPositions == 0 && (buyProfit || sellCheap))//MarketSeries.Close.Last(1) > MarketSeries.Close.Last(2))
   //if (numPositions == 0)
   {
      //Print("TEST1");
       double price = ExecuteMarketOrder(tradeType, volume, 0, true);
       
       firstOrder = false;
       //datetime now = GetTimeLastOrder(ticketId);
       //OrderSelect((OrdersTotal() -1 ), SELECT_BY_POS);
       //datetime now = GetTimeLastOrder(OrderTicket());
       
       if (tradeType == BL_BUY)
       {
           LastBuyTradeTime2 = now;//MarketSeries.OpenTime.Last(0);
           lastBuyPrice2 = price;
       }
       else if (tradeType == BL_SELL)
       {
           LastSellTradeTime2 = now;//MarketSeries.OpenTime.Last(0);
           lastSellPrice2 = price;
       }
   }
   if (numPositions > 0) // */
   {  
      double pipStep = PipStepsSTO;
      
      //Print("TEST2");
       
       double lastopenprice = GetLastOpenPrice(tradeType);
       lastopenprice = (tradeType == BL_BUY) ? lastBuyPrice2 : lastSellPrice2;
       
       datetime lastTradeTime = (tradeType == BL_BUY) ? LastBuyTradeTime2 : LastSellTradeTime2;

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
          double price = ExecuteMarketOrder(tradeType, volume, 0, false);
          
          if (tradeType == BL_BUY)
          {
              LastBuyTradeTime2 = now;
              lastBuyPrice2 = price;
          }
          else if (tradeType == BL_SELL)
          {
              LastSellTradeTime2 = now;
              lastSellPrice2 = price;
          }
         
          firstOrder = false;
       }
      
   }
}

void ProcessOrder_BB(BL_TRADE_TYPE tradeType)
{
   double volume = 0;

//ExecuteMarketOrder(tradeType);return;

   int numPositions = GetCountOpenPositions(tradeType);

///*

   bool buyProfit = (tradeType == BL_BUY) && (Close[1] > Close[2]);
   bool sellCheap = (tradeType == BL_SELL) && (Close[2] > Close[1]);

   //bool buyProfit = (tradeType == BL_BUY) && (Close[0] > Close[1]);
   //bool sellCheap = (tradeType == BL_SELL) && (Close[1] > Close[0]);

   datetime now = Time[0];

   if (numPositions == 0 && (buyProfit || sellCheap))//MarketSeries.Close.Last(1) > MarketSeries.Close.Last(2))
   //if (numPositions == 0)
   {
      //Print("TEST1");
       double price = ExecuteMarketOrder(tradeType, volume, 0, true);
       
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
             double price = ExecuteMarketOrder(tradeType, volume, 0, false);
             
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
           
             firstOrder = false;
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

void DoBuy_STO()
{
   int numOpenSell;
   int numOpenBuy;
   
   GetNumOpenPositions(numOpenBuy, numOpenSell);

   if(numOpenBuy < MaxOpenBuy)
   {
      //ProcessOrder_STO_BUY(BL_BUY);
      ProcessOrder_STO(BL_BUY);
   }
}

void DoSell_STO()
{
   int numOpenSell;
   int numOpenBuy;
   
   GetNumOpenPositions(numOpenBuy, numOpenSell);
   
   if(numOpenSell < MaxOpenSell)
   {
      //ProcessOrder_STO_SELL(BL_SELL);
      ProcessOrder_STO(BL_SELL);
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
          if(OrderType()==OP_BUY  && OrderSymbol() == Symbol() && OrderMagicNumber() == MagicNumberB)
          {
            if(tradeType == BL_BUY)
            {
               result = OrderClose(OrderTicket(), OrderLots(), GetClosingPrice(BL_BUY), GetSlippage(), clrYellow); // Bid
            }
          }
          if(OrderType()==OP_SELL && OrderSymbol() == Symbol() && OrderMagicNumber() == MagicNumberS)
          {
            if(tradeType == BL_SELL)
            {
               result = OrderClose(OrderTicket(), OrderLots(), GetClosingPrice(BL_SELL), GetSlippage(), clrYellow); // Ask
            }
          }
       }
   }
}

void SmartGrid()
{
   double FirstOrderVolume = GetFirstOrderVolume();

   double targetProfit = FirstOrderVolume * TakeProfit * GetPipSize() * GetPipScalar() * 1000000 ; // * 10000000
   //double targetProfit = FirstOrderVolume * TakeProfit * GetPipSize();// * Point(); // Symbol.PipSize
   //double targetProfit = FirstOrderVolume * TakeProfit * Point();
   //double targetProfit = FirstOrderVolume * TakeProfit;
   
   double targetLoss = FirstOrderVolume * StopLoss * GetPipSize() * GetPipScalar() * 1000000;
   
   int numOpenBuy = 0;
   int numOpenSell = 0;
   double buyNetProfit = 0;
   double sellNetProfit = 0;
   double averageBuyNetProfit;
   double averageSellNetProfit;
   
   double netProfit = 0;
   double averageNetProfit = 0;
   int numOpenOrders = 0;
   
   bool HedgeCloseOff = false; // AlwaysHedge
   
   if(HedgeCloseOff)
   {
      //for(int i=0;i<OrdersTotal();i++)
      for(int i=OrdersTotal()-1; i>=0; i--)
      {
          if(OrderSelect(i, SELECT_BY_POS, MODE_TRADES))
          {         
             if(OrderType()==OP_BUY  && OrderSymbol() == Symbol() && OrderMagicNumber() == MagicNumberB)
             {
               numOpenOrders ++;
               netProfit += OrderProfit() + OrderCommission() + OrderSwap();
             }
             if(OrderType()==OP_SELL && OrderSymbol() == Symbol() && OrderMagicNumber() == MagicNumberS)
             {
               numOpenOrders++;
               netProfit += OrderProfit() + OrderCommission() + OrderSwap();
             }
          }
      }
      
      if(numOpenOrders > 0)
      {
         averageNetProfit = netProfit / (double)numOpenOrders;
         
         Print("numOpenOrders: ", numOpenOrders, " averageNetProfit: ", averageNetProfit);
         
         if(averageNetProfit >= targetProfit)
         {
            Print("Closing ALL trades - AT PROFIT");
            CloseTrades(BL_BUY);
            CloseTrades(BL_SELL);
         }
         if(StopLossEnabled && averageNetProfit <= -targetLoss)
         {
            Print("Closing ALL trades - AT LOSS");
            CloseTrades(BL_BUY);
            CloseTrades(BL_SELL);
         }
      }
   }
   else
   {
      //for(int i=0;i<OrdersTotal();i++)
      for(int i=OrdersTotal()-1; i>=0; i--)
      {
          if(OrderSelect(i, SELECT_BY_POS, MODE_TRADES))
          {          
             if(OrderType()==OP_BUY  && OrderSymbol() == Symbol() && OrderMagicNumber() == MagicNumberB)
             {
               numOpenBuy ++;
               buyNetProfit += OrderProfit() + OrderCommission() + OrderSwap();
             }
             if(OrderType()==OP_SELL && OrderSymbol() == Symbol() && OrderMagicNumber() == MagicNumberS)
             {
               numOpenSell++;
               sellNetProfit += OrderProfit() + OrderCommission() + OrderSwap();
             }
          }
      }
      if(numOpenBuy > 0)
      {
         
         //targetProfit = FirstOrderVolume * Ask + TakeProfit * GetPipSize();
         averageBuyNetProfit = buyNetProfit / (double)numOpenBuy;
         
         //Print("averageBuyNetProfit: ", averageBuyNetProfit);
         
         //Print("averageBuyNetProfit: ", averageBuyNetProfit, " targetProfit: ", targetProfit);
         
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
         averageSellNetProfit = sellNetProfit / (double)numOpenSell;
         
         //Print("averageSellNetProfit: ", averageSellNetProfit);
         //Print("targetProfit: ", targetProfit);
         //if(averageSellNetProfit >0)
         //{
            //Print("GTZERO");    
            //Print("averageSellNetProfit: ", averageSellNetProfit);
            //Print("targetProfit: ", targetProfit);  
         //}
         
         //Print("averageSellNetProfit: ", averageSellNetProfit, " targetProfit: ", targetProfit);
         
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
}
        
void BollingerBands()
{
   int period = PERIOD_CURRENT;
   
   //double BB_UPPER   = iBands(Symbol(),period,BBPeriod ,BBDeviation ,BBShift ,PRICE_CLOSE ,MODE_UPPER ,0);
   //double BB_SMA     = iBands(Symbol(),period ,BBPeriod ,BBDeviation ,BBShift ,PRICE_CLOSE ,MODE_SMA   ,0);
   //double BB_LOWER   = iBands(Symbol(),period ,BBPeriod ,BBDeviation ,BBShift ,PRICE_CLOSE ,MODE_LOWER ,0);
   double BB_UPPER1   = iBands(Symbol(),period ,BBPeriod ,BBDeviation ,BBShift ,BBPriceMode ,MODE_UPPER ,1); // Top band
   double BB_UPPER2   = iBands(Symbol(),period ,BBPeriod ,BBDeviation ,BBShift ,BBPriceMode ,MODE_UPPER ,2);
   double BB_LOWER1   = iBands(Symbol(),period ,BBPeriod ,BBDeviation ,BBShift ,BBPriceMode ,MODE_LOWER ,1); // Bottom band
   double BB_LOWER2   = iBands(Symbol(),period ,BBPeriod ,BBDeviation ,BBShift ,BBPriceMode ,MODE_LOWER ,2);
   
   if((countConsecutiveBuys >= MaxConsecutiveBuys) && (countConsecutiveSells >= MaxConsecutiveSells))
   {
      countConsecutiveSells = 0;
      countConsecutiveBuys = 0;
   }
   
   if(Bars > 2)
   {
      if (Close[2] > BB_UPPER2)//boll.Top.Last(2))
       {
           if (Close[1] > BB_UPPER1)//boll.Top.Last(1))
           {
               countConsecutiveSells = 0;
               
               if(countConsecutiveBuys <= MaxConsecutiveBuys)
               {
                  DoBuy_BB();
                  countConsecutiveBuys++;
               }
           }
           else
           {
               countConsecutiveBuys = 0;
               
               if(countConsecutiveSells <= MaxConsecutiveSells)
               {
                  DoSell_BB();
                  countConsecutiveSells++;
               }
           }
       }
       else if (Close[2] < BB_LOWER2)//boll.Bottom.Last(2))
       {
           if (Close[1] < BB_LOWER1)//boll.Bottom.Last(1))
           {
               countConsecutiveBuys = 0;
               
               if(countConsecutiveSells <= MaxConsecutiveSells)
               {
                  DoSell_BB();
                  countConsecutiveSells++;
               }
           }
           else
           {
               countConsecutiveSells = 0;
               
               if(countConsecutiveBuys <= MaxConsecutiveBuys)
               {
                  DoBuy_BB();
                  countConsecutiveBuys++;
               }
           }
       }
   }
}

void BollingerBands_Buy()
{
   int period = PERIOD_CURRENT;
   //double BB_UPPER   = iBands(Symbol(),period,BBPeriod ,BBDeviation ,BBShift ,PRICE_CLOSE ,MODE_UPPER ,0);
   //double BB_SMA     = iBands(Symbol(),period ,BBPeriod ,BBDeviation ,BBShift ,PRICE_CLOSE ,MODE_SMA   ,0);
   //double BB_LOWER   = iBands(Symbol(),period ,BBPeriod ,BBDeviation ,BBShift ,PRICE_CLOSE ,MODE_LOWER ,0);
   double BB_UPPER1   = iBands(Symbol(),period ,BBPeriod ,BBDeviation ,BBShift ,PRICE_CLOSE ,MODE_UPPER ,1); // Top band
   double BB_UPPER2   = iBands(Symbol(),period ,BBPeriod ,BBDeviation ,BBShift ,PRICE_CLOSE ,MODE_UPPER ,2);
   double BB_LOWER1   = iBands(Symbol(),period ,BBPeriod ,BBDeviation ,BBShift ,PRICE_CLOSE ,MODE_LOWER ,1); // Bottom band
   double BB_LOWER2   = iBands(Symbol(),period ,BBPeriod ,BBDeviation ,BBShift ,PRICE_CLOSE ,MODE_LOWER ,2);
   
   if((countConsecutiveBuys >= MaxConsecutiveBuys) && (countConsecutiveSells >= MaxConsecutiveSells))
   {
      countConsecutiveSells = 0;
      countConsecutiveBuys = 0;
   }
   
   if(Bars > 2)
   {
      if (Close[2] > BB_UPPER2)//boll.Top.Last(2))
       {
           if (Close[1] > BB_UPPER1)//boll.Top.Last(1))
           {
               countConsecutiveSells = 0;
               
               if(countConsecutiveBuys <= MaxConsecutiveBuys)
               {
                  DoBuy_BB();
                  countConsecutiveBuys++;
               }
           }
       }
       else if (Close[2] < BB_LOWER2)//boll.Bottom.Last(2))
       {
           if (!(Close[1] < BB_LOWER1))//boll.Bottom.Last(1))
           {
               countConsecutiveSells = 0;
               
               if(countConsecutiveBuys <= MaxConsecutiveBuys)
               {
                  DoBuy_BB();
                  countConsecutiveBuys++;
               }
           }
       }
   }
}

void BollingerBands_Sell()
{
   int period = PERIOD_CURRENT;
   //double BB_UPPER   = iBands(Symbol(),period,BBPeriod ,BBDeviation ,BBShift ,PRICE_CLOSE ,MODE_UPPER ,0);
   //double BB_SMA     = iBands(Symbol(),period ,BBPeriod ,BBDeviation ,BBShift ,PRICE_CLOSE ,MODE_SMA   ,0);
   //double BB_LOWER   = iBands(Symbol(),period ,BBPeriod ,BBDeviation ,BBShift ,PRICE_CLOSE ,MODE_LOWER ,0);
   double BB_UPPER1   = iBands(Symbol(),period ,BBPeriod ,BBDeviation ,BBShift ,PRICE_CLOSE ,MODE_UPPER ,1); // Top band
   double BB_UPPER2   = iBands(Symbol(),period ,BBPeriod ,BBDeviation ,BBShift ,PRICE_CLOSE ,MODE_UPPER ,2);
   double BB_LOWER1   = iBands(Symbol(),period ,BBPeriod ,BBDeviation ,BBShift ,PRICE_CLOSE ,MODE_LOWER ,1); // Bottom band
   double BB_LOWER2   = iBands(Symbol(),period ,BBPeriod ,BBDeviation ,BBShift ,PRICE_CLOSE ,MODE_LOWER ,2);
   
   if((countConsecutiveBuys >= MaxConsecutiveBuys) && (countConsecutiveSells >= MaxConsecutiveSells))
   {
      countConsecutiveSells = 0;
      countConsecutiveBuys = 0;
   }
   
   if(Bars > 2)
   {
      if (Close[2] > BB_UPPER2)//boll.Top.Last(2))
       {
           if (!(Close[1] > BB_UPPER1))//boll.Top.Last(1))
           {
               countConsecutiveBuys = 0;
               
               if(countConsecutiveSells <= MaxConsecutiveSells)
               {
                  DoSell_BB();
                  countConsecutiveSells++;
               }
           }
       }
       else if (Close[2] < BB_LOWER2)//boll.Bottom.Last(2))
       {
           if (Close[1] < BB_LOWER1)//boll.Bottom.Last(1))
           {
               countConsecutiveBuys = 0;
               
               if(countConsecutiveSells <= MaxConsecutiveSells)
               {
                  DoSell_BB();
                  countConsecutiveSells++;
               }
           }
       }
   }
}

void Stochastic()
{
   int period = PERIOD_CURRENT;
   
   //double dstoch = iCustom(Symbol(),period,"StochasticDiNapoli_v1",FastK,SlowK,SlowD,0,0);
   //double dssignal = iCustom(Symbol(),period,"StochasticDiNapoli_v1",FastK,SlowK,SlowD,1,0);
   
   //double valueIndic1 = iWPR (Symbol(), period, 8,0 );
   //double PrevValueIndic1 = iWPR (Symbol(), period, 8,1);
   
   //double valueIndic2 = iCustom (Symbol(), period, "StochasticDiNapoli_v1", FastK,SlowK,SlowD,0,0);
   //double PrevValueIndic2 = iCustom (Symbol(), period, "StochasticDiNapoli_v1", FastK,SlowK,SlowD,0,1);
   
   /*
   string    timeFrame     =  "Current time frame";
   bool      showBars      = false;
   bool      showArrows    = true;
   bool      alertsOn      = false;
   bool      alertsMessage = true;
   bool      alertsSound   = false;
   bool      alertsEmail   = false;*/
   
   //double KFull = iCustom (Symbol(), period, "color_stochastic_v1.02", KPeriod,Slowing,DPeriod,MAMethod,PriceField,overBought,overSold,timeFrame,showBars,showArrows,alertsOn,alertsMessage,alertsSound,alertsEmail,0,0);
   //double KFullPrev = iCustom (Symbol(), period, "color_stochastic_v1.02", KPeriod,Slowing,DPeriod,MAMethod,PriceField,overBought,overSold,timeFrame,showBars,showArrows,alertsOn,alertsMessage,alertsSound,alertsEmail,0,1);
   
   double KFull = 0;
   double KFullPrev = 0;
   
   if(ApplyStochasticDiNapoli_v1)
   {
      KFull = iCustom(Symbol(),period,"StochasticDiNapoli_v1",FastK,SlowK,SlowD,1,0);
      KFullPrev = iCustom(Symbol(),period,"StochasticDiNapoli_v1",FastK,SlowK,SlowD,1,1);
   }
   else if(ApplyStochasticDiNapoli_v2)
   {
      KFull = iCustom(Symbol(),period,"StochasticDiNapoli_v1",FastK,SlowK,SlowD,0,0);
      KFullPrev = iCustom(Symbol(),period,"StochasticDiNapoli_v1",FastK,SlowK,SlowD,0,1);
   }
   else
   {
      KFull = iStochastic(Symbol(),period,KPeriod,DPeriod,Slowing,MAMethod,PriceField,MODE_MAIN,0);
      KFullPrev = iStochastic(Symbol(),period,KPeriod,DPeriod,Slowing,MAMethod,PriceField,MODE_MAIN,1);
   }
   
   
   
   
   if(ApplyBollingerBands)
   {
      if (KFull > overBought && KFullPrev < overBought) 
      {
         BollingerBands_Buy();
      }
      if (KFull < overSold   && KFullPrev > overSold)
      {
         BollingerBands_Sell();
      }
   }
   else if (ApplyMACrossoverBol)
   {
      double FastMovingAverageCurr = iMA(Symbol(),period,MA_FASTMA,0,MA_MA_METHOD,MA_PRICE_METHOD,0); 
      double FastMovingAveragePrev = iMA(Symbol(),period,MA_FASTMA,0,MA_MA_METHOD,MA_PRICE_METHOD,1); 
      double SlowMovingAverageCurr = iMA(Symbol(),period,MA_SLOWMA,0,MA_MA_METHOD,MA_PRICE_METHOD,0); 
      double SlowMovingAveragePrev = iMA(Symbol(),period,MA_SLOWMA,0,MA_MA_METHOD,MA_PRICE_METHOD,1); 
   
      //Print("KFull: ", KFull, " KFullPrev: ", KFullPrev); 

      
      if (KFull > overBought && KFullPrev < overBought) 
      {
         if((SlowMovingAveragePrev > FastMovingAveragePrev) && (FastMovingAverageCurr > SlowMovingAverageCurr) && (MathAbs(FastMovingAverageCurr-SlowMovingAverageCurr)>0.00005)) 
         {
            //BollingerBands_Buy();
            DoBuy_STO();
         }
      }
      if (KFull < overSold   && KFullPrev > overSold)
      {
         if((SlowMovingAveragePrev < FastMovingAveragePrev) && (SlowMovingAverageCurr > FastMovingAverageCurr) && (MathAbs(SlowMovingAverageCurr-FastMovingAverageCurr)>0.00005)) 
         {
            //BollingerBands_Sell();
            DoSell_STO();
         }
      }
   }
   else
   {
      if (KFull > overBought && KFullPrev < overBought) 
      //if (valueIndic1> valueIndic2 && PrevValueIndic1 <PrevValueIndic2)
      {
         DoBuy_STO();
      }
      if (KFull < overSold   && KFullPrev > overSold)
      //if (valueIndic1 < valueIndic2 && PrevValueIndic1> PrevValueIndic2)
      {
         DoSell_STO();
      }
   }
   
   if(ApplyRandomWalk)
   {
      /*
      if (valueIndic1> valueIndic2 && PrevValueIndic1 <PrevValueIndic2)
      {
         DoBuy();
      }
      if (valueIndic1 <valueIndic2 && PrevValueIndic1> PrevValueIndic2)
      {
         DoSell();
      }
   
      if (valueIndic2 >= AlertLevel && PrevValueIndic2 <= AlertLevel)
      {
         Alert ("Crossing up on", Symbol (), "!!!");
      }
      if (valueIndic2 <= AlertLevel && PrevValueIndic2 >= AlertLevel)
      {
         Alert ("Crossing down on", Symbol (), "!!!");
      }
      */
   
   }
}

void WillPeriod()
{
   int period = PERIOD_CURRENT;
   
   double willcurrent=iWPR(Symbol(), period, WP_willperiod, 0);
   double willprevious=iWPR(Symbol(), period, WP_willperiod, 1);
   
   if (willprevious < WP_upperband && willcurrent >= WP_upperband)
   {
      DoBuy_STO();
   }
   
   if (willprevious > WP_lowerband && willcurrent <= WP_lowerband)
   {
      DoSell_STO();
   }
}


void ForceIndex()
{
   int period = PERIOD_CURRENT;
   
   //double forcecurrent = iForce(Symbol(), period, FI_forceu, MODE_SMA, PRICE_CLOSE, 0);
   
   double forcecurrent = Volume[0]*(
      iMA(Symbol(), period, FI_forceu,0,FI_ExtForceMAMethod,FI_ExtForceAppliedPrice,0)
      - iMA(Symbol(), period, FI_forceu,0,FI_ExtForceMAMethod,FI_ExtForceAppliedPrice,1)
   );
   
   if(forcecurrent < 0)
   {
      DoBuy_STO();
   }
   
   if(forcecurrent > 0)
   {
      DoSell_STO();
   }
}

void RSI()
{
   int period = PERIOD_CURRENT;
   double rsicurrent=iRSI(Symbol(), period, RSI_rsiu, PRICE_CLOSE, 0);
   double rsiprevious=iRSI(Symbol(), period, RSI_rsiu, PRICE_CLOSE, 1);
   
   if (rsiprevious < RSI_lowerband && rsicurrent >= RSI_lowerband)
   {
      DoBuy_STO();
   }
   
   if (rsiprevious > RSI_upperband && rsicurrent <= RSI_upperband)
   {
      DoSell_STO();
   }
}

void MACD()
{
   int period = PERIOD_CURRENT;

   double macdmaincurr=iMACD(Symbol(), period,MACD_fastmacd,MACD_slowmacd,MACD_signalmacd,0,MODE_MAIN,0);
   double macdmainprev=iMACD(Symbol(), period,MACD_fastmacd,MACD_slowmacd,MACD_signalmacd,0,MODE_MAIN,1);
   double macdsigcurr=iMACD(Symbol(), period,MACD_fastmacd,MACD_slowmacd,MACD_signalmacd,0,MODE_SIGNAL,0);
   double macdsigprev=iMACD(Symbol(), period,MACD_fastmacd,MACD_slowmacd,MACD_signalmacd, 0,MODE_SIGNAL,1);


   if (macdmainprev < 0 && macdmaincurr >= 0)
   {
      DoBuy_STO();
   }
   
   if (macdmainprev > 0 && macdmaincurr <= 0)
   {
      DoSell_STO();
   }
}

void MACrossoverBol()
{
   int period = PERIOD_CURRENT;

   double FastMovingAverageCurr = iMA(Symbol(),period,MA_FASTMA,0,MA_MA_METHOD,MA_PRICE_METHOD,0); 
   double FastMovingAveragePrev = iMA(Symbol(),period,MA_FASTMA,0,MA_MA_METHOD,MA_PRICE_METHOD,1); 
   double SlowMovingAverageCurr = iMA(Symbol(),period,MA_SLOWMA,0,MA_MA_METHOD,MA_PRICE_METHOD,0); 
   double SlowMovingAveragePrev = iMA(Symbol(),period,MA_SLOWMA,0,MA_MA_METHOD,MA_PRICE_METHOD,1); 

   //Print("KFull: ", KFull, " KFullPrev: ", KFullPrev);

   if((SlowMovingAveragePrev < FastMovingAveragePrev) && (SlowMovingAverageCurr > FastMovingAverageCurr) && (MathAbs(SlowMovingAverageCurr-FastMovingAverageCurr)>0.00005)) 
   {
      //BollingerBands_Sell();
      DoSell_STO();
   }
   
   if((SlowMovingAveragePrev > FastMovingAveragePrev) && (FastMovingAverageCurr > SlowMovingAverageCurr) && (MathAbs(FastMovingAverageCurr-SlowMovingAverageCurr)>0.00005)) 
   {
      //BollingerBands_Buy();
      DoBuy_STO();
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
          if(OrderType()==OP_BUY  && OrderSymbol() == Symbol() && OrderMagicNumber() == MagicNumberB)
          {
            profit = OrderProfit() + OrderCommission() + OrderSwap();
          
            if(profit < 0)
            {
               result = OrderClose(OrderTicket(), OrderLots(), GetClosingPrice(BL_BUY), GetSlippage(), clrYellow);
            }
            
          }
          if(OrderType()==OP_SELL && OrderSymbol() == Symbol() && OrderMagicNumber() == MagicNumberS)
          {
            profit = OrderProfit() + OrderCommission() + OrderSwap();
          
            if(profit < 0)
            {
               result = OrderClose(OrderTicket(), OrderLots(), GetClosingPrice(BL_SELL), GetSlippage(), clrYellow);
            }
          }
       }
   }
}

bool DrawdownMonitor()
{
   if(!ApplyAntiDrawdown) return true;
   
   double accountBalance = AccountBalance();
   double accountEquity = AccountEquity();
   
   double equityPercent = (1 -  accountEquity / accountBalance) * 100;
   double drawdownPercent = NormalizeDouble(DrawdownPercent,Digits());
   //double drawdownPercent = DrawdownPercent;
   
   bool result = false;
   isDrawdown = false;
   
   if(equityPercent > drawdownPercent && !waiting)
   {
      //Print("Drawdown too much, closing orders");
      //CloseLosingTrades();
      waiting = true;
      result = false;
      
      //Print("DDDDDDDDDDDDDDDDDDDDDDD Drawdown too much, waiting ", WaitAfterDrawdownPips, " pips for better opportunity. Remaining equity: ", accountEquity, " Remaining balance: ", accountBalance);
      
      //Print("BBBBBBBBBBBBBBBBBBBBBBB Executing Lockup-Hedge order ");
      //if(lastOrderTradeType == BL_BUY)
      //{
      //   ExecuteMarketOrder(BL_SELL);
      //}
      //else
      //{
      //   ExecuteMarketOrder(BL_BUY);
      //}
   }
   if(equityPercent > drawdownPercent)
   {
      result = false;
      isDrawdown = true;
   }
   else
   {
      if(ApplyFastWait)
      {
         waiting = false;
      }
   }
   
   if(waiting)
   {      
      //double current_ask = MarketInfo(Symbol(),MODE_ASK);
      //double current_bid = MarketInfo(Symbol(),MODE_BID);
      double current_ask = Ask;
      double current_bid = Bid;
      
      //double waitPoints = WaitAfterDrawdownPips * GetPipSize() * 1000000; // Point() GetPipSize()
      //double waitPoints = WaitAfterDrawdownPips * GetPipSize(); // Point() GetPipSize()
      //double waitPoints = WaitAfterDrawdownPips * Point(); // Point() GetPipSize()
      double waitPoints = WaitAfterDrawdownPips ; // * GetPipSize()
      
      //if(Ask >= Ask + waitPoints)
      if((Ask - lastPriceAsk) / GetPipSize() >= WaitAfterDrawdownPips)
      {
         //Print("######################### Price has moved up ",WaitAfterDrawdownPips," pips to ask: ",current_ask," from ", current_ask + waitPoints);
         
         waiting = false;
         
         result = true;
      }
      //else if(Bid >= Bid + waitPoints)
      else if((Bid - lastPriceBid) / GetPipSize() >= WaitAfterDrawdownPips)
      {
         //Print("######################### Price has moved up ",WaitAfterDrawdownPips," pips to bid: ",current_bid," from ", current_bid - waitPoints);
         
         waiting = false;
         
         result = true;
      }
      
      lastPriceAsk = Ask;
      lastPriceBid = Bid;
   }
   else
   {
      result = true;
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

void ShowOpenTrades()
{
   double profit;
   Print("EXISTING ORDERS TOTAL:", OrdersTotal());
   
   //for(int i=0;i<OrdersTotal();i++)
   for(int i=OrdersTotal()-1; i>=0; i--)
   {
       if(OrderSelect(i, SELECT_BY_POS, MODE_TRADES))
       {      
          if(OrderType()==OP_BUY  && OrderSymbol() == Symbol() && OrderMagicNumber() == MagicNumberB)
          {
            profit = OrderProfit() + OrderCommission() + OrderSwap();
            
            Print("@@@@@@@@@@@@@@@@@@@@@@ ORDER [BUY] index: ",i," profit: ", profit);
          }
          if(OrderType()==OP_SELL && OrderSymbol() == Symbol() && OrderMagicNumber() == MagicNumberS)
          {
            profit = OrderProfit() + OrderCommission() + OrderSwap();
            
            Print("@@@@@@@@@@@@@@@@@@@@@@ ORDER [BUY] index: ",i," profit: ", profit);
          }
       }
   }
}

/// Trailing Stop
void TrailingPositions() 
{
  for (int i=0; i<OrdersTotal(); i++) 
  {
    if (OrderSelect(i, SELECT_BY_POS, MODE_TRADES)) 
    {
      if (OrderType()==OP_SELL && OrderMagicNumber()==MagicNumberS) 
      {
         if (OrderOpenPrice()-Ask>TrailingAct*Point && TrailPrice ==0) 
         {
            TrailPrice=Ask+TrailingStep*Point;
            Print("TRAIL PRICE SET: ",TrailPrice);
            if(TrailingStep > 8)
            {
               ModifyStopLoss(TrailPrice);
            }
         }
         if (TrailPrice!=0 && Ask+TrailingStep*Point < TrailPrice  )
         {
            TrailPrice=Ask-TrailingStep*Point;
            Print("TRAIL PRICE MODIFIED: ",TrailPrice);
            if(TrailingStep > 8)
            {
               ModifyStopLoss(TrailPrice);
            }
         }
         if (TrailPrice != 0 && Ask >= TrailPrice )
         {
            CloseOrder(2);
         }
      }
      if (OrderType()==OP_BUY && OrderMagicNumber()==MagicNumberB) 
      {
         if (Bid-OrderOpenPrice() > TrailingAct*Point && TrailPrice ==0) 
         {
            TrailPrice=Bid-TrailingStep*Point;
            Print("TRAIL PRICE MODIFIED: ",TrailPrice);
            if(TrailingStep > 8)
            {
               ModifyStopLoss(TrailPrice);
            }
         }
         if (TrailPrice!= 0 && Bid-TrailingStep*Point > TrailPrice )
         {
            TrailPrice=Bid-TrailingStep*Point;
            Print("TRAIL PRICE MODIFIED: ",TrailPrice);
            if(TrailingStep > 8)
            {
               ModifyStopLoss(TrailPrice);
            }
         }
         if (TrailPrice != 0 && Bid <= TrailPrice )
         {
            CloseOrder(1);
         }   
      }
   }
   }
}

void CloseOrder(int ord)
{
   int res;
    for(int i=0;i<OrdersTotal();i++)
    {  
      if(OrderSelect(i, SELECT_BY_POS, MODE_TRADES))
      {
         if (OrderType()==OP_BUY && OrderMagicNumber()==MagicNumberB)
         {
            if (ord==1)
            {
            res = OrderClose(OrderTicket(),OrderLots(),Bid,Slippage,clrYellow); // close 
            TrailPrice=0;
            if(res<0)
            {
               int error=GetLastError();
               Print("Error = ",ErrorDescription(error));
            }
         }}     
         
         if (OrderType()==OP_SELL && OrderMagicNumber()==MagicNumberS )
         {
            if (ord==2) 
            {                          // MA BUY signals
               res = OrderClose(OrderTicket(),OrderLots(),Ask,Slippage,clrYellow); // close 
               TrailPrice=0;
               if(res<0)
               {
                  int error=GetLastError();
                  Print("Error = ",ErrorDescription(error));
               }
            }     
         } 
      }  
   }    
}  
 
void Scalp()
{
   double res;
   int error;
   for(int i=0;i<OrdersTotal();i++)
   {
      if(OrderSelect(i,SELECT_BY_POS,MODE_TRADES))
      {
         if(OrderSymbol()==Symbol() )
         {
            if(OrderType()==OP_BUY && OrderMagicNumber()==MagicNumberB)
            {
               if(Bid - OrderOpenPrice() >= ScalpPips*Point)
               {
                  res = OrderClose(OrderTicket(),OrderLots(),Bid,Slippage,clrYellow); // close 
                  TrailPrice=0;
                  if(res<0){
                     error=GetLastError();
                    // Print("Error = ",ErrorDescription(error));
                  }
               }
            }
            if(OrderType()==OP_SELL && OrderMagicNumber()==MagicNumberS)
            {
               if(OrderOpenPrice() - Ask >= ScalpPips*Point)
               {
                  res = OrderClose(OrderTicket(),OrderLots(),Ask,Slippage,clrYellow); // close 
                  TrailPrice=0;
                  if(res<0){
                     error=GetLastError();
                    // Print("Error = ",ErrorDescription(error));
                  }
               }
            }
         }
      }
   }
}

//+------------------------------------------------------------------+
// Order Modify function
//+------------------------------------------------------------------+
void ModifyStopLoss(double ldStop) 
{
  bool   fm;
  
  fm=OrderModify(OrderTicket(), OrderOpenPrice(), ldStop, OrderTakeProfit(), 0, clrHotPink);
}

//bool CheckSpread()
//{  
//   double spread = MarketInfo(Symbol(), MODE_SPREAD);
//   //double spreadPips = spread / Point();
//   //double spreadPips = spread / (GetPipSize() * GetPipScalar());
//   
//   //Print("spread: ", spread, " ", spreadPips);
//   
//   return (spread <= MaxSpread);
//}

//************************************************************************************************/
//*                                                                                              */
//************************************************************************************************/
void OnTick()
{
   IsNewBar = IsThisANewCandle();
   
//*************************************************************//
   if(IsNewBar && DrawdownMonitor() /*&& CheckSpread()*/)
   {
      
      SmartGrid();
      
      AutoQuitOnClose();
     
   
      if(ApplyStochasticDiNapoli)
      {
         Stochastic();
      }
      
      if(!ApplyStochasticDiNapoli && ApplyBollingerBands)
      {
         BollingerBands();
      }
      if(ApplyRandomWalk)
      {
         RandomWalk();
      }

      if(!ApplyStochasticDiNapoli && ApplyMACrossoverBol)
      {
         MACrossoverBol();
      }

      if(ApplyWillPeriod)
      {
         WillPeriod();
      }

      if(ApplyForceIndex)
      {
         ForceIndex();
      }
      
      if(ApplyRSI)
      {
         RSI();
      }
      
      if(ApplyMACD)
      {
         MACD();
      }
      
      // Trailing Stop.
      if(UseTrailing)
      {
         TrailingPositions();
      }
      if(UseTightStop)
      {
         Scalp();
      }
   }
   else{
      //Print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ IS NOT NEW BAR");
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