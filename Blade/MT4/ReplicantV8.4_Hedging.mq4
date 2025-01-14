#property copyright   "Copyright 2020, Metamorphic."
#property link        "https://jumpinalake.com"
#property version     "8.4"
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
input double      TakeProfit                 = 3;              // Take Profit (in pips)
//input double      StopLoss                   = 1000;           // Stop Loss (in pips)
//input bool        StopLossEnabled            = false;          // Enable silent Stop Loss
//input double      TrailingStopLoss           = 5;              // Trailing Stop Loss
input int         MagicNumber                = 666;            // Advisor Magic Number
input int         Slippage                   = 5;              // Slippage (in pips)
input double      MaxSpread                  = 3;
input int         MaxOpenBuy                 = 35;             // Maximum number of open buy positions
input int         MaxOpenSell                = 35;             // Maximum number of open sell positions
//input double      PipStep                    = 175;            // Pip step (random walk)
//input double      PipStepBB                  = 175;            // Pip step (bolinger)
//input string      PipStepsRW                 = "235";
//input string      PipStepsBB                 = "35";
//input double      PipStepsRW                 = 235;
input double      PipStepsBB                 = 35;
//input double      PipStepsSTO                = 1;
input double      PipScalar                  = 1.0;            // Pip scalar

//input string b = "$$$$$$$$$$$$$$ Bollinger Band Settings $$$$$$$$$$$$$$";
input int         BBPeriod                   =  35;            // Bolinger band period
input double      BBDeviation                =  2;             // Bollinger band deviation (stddev)
input int         BBShift                    =  0;             // Bollinger band shift
input int         MaxConsecutiveBuys         = 15;             // Max consecutive number of buy positions
input int         MaxConsecutiveSells        = 15;             // Max consecutive number of sell positions

//input string c = "$$$$$$$$$$$$$$ Advanced Setup $$$$$$$$$$$$$$";
input double      FirstVolume                = 0.1;            // First order volume scalar
input double      VolumeExponent             = 1.7;            // Volume exponent 
input double      volumeExponential          = 1.3;            // Volume exponential
input double      InitialVolumeScalar        = 2.0f;           // Scalar of volume for each order
input double      NumGroups                  = 1;              // Number of groups
input BL_VOLUME   VolumeFlags                = BL_BASIC_VOLUME; // Normalise volume lot sizes
input double      PowerFactor                = 1.0;            // Power factor
input double      PowerOffset                = 0.0;            // Power offset
input double      PercentIncreaseThreshold   = 100000;         // Percent increase threshold (Free margin denominator)
//input double      PercentIncreaseThreshold100 = 1000000;       // Percent increase threshold (Free margin used above $100K)
input double      RiskPercent                = 75.0;           // Risk percent (Risk managment)
//input double      RiskPercentB               = 100.0;           // Risk percent B (Risk managment)
//input int         InitialTrades              = 50;             // Initial number of trades for initial lower risk
//input double      InitialVolumeExponent      = 1.7;            // Initial volume exponent
//input double      InitialNumGroups           = 3;              // Initial number of groups for initial lower risk
//input double      DrawdownNumGroups          = 3;              // Drawdown number of groups boost
//input int         EquityInterval             = 10000;          // Equity increase threshold
//input double      EquityExponent             = 2.0;            // Equity exponent
//input double      RiskRatio                  = 1.0;            // Risk ratio
//input int         InitialNumTradesLinear     = 0;              // Initial linearity number of trades
//input double      LotIncreasePercent         = 35.0;           // Increase lot size by percentage
//input double      HedgeLotScalar             = 1.0;            // Hedge lot scalar
input double      MaxExp                     = 170.0;          // Max Exp
input double      MaxLot                     = 200;             // Max Lot

input double      maxDrawdownPercent         = 20.0;           // Maximum allowed drawdown percentage
input int         trailingStopDistance       = 50;             // Trailing stop distance
input int         breakevenDistance          = 20;             // Breakeven distance
input double      riskRewardRatio            = 2.0;            // Risk-reward ratio
input bool        ApplyAdaptiveVolumeShunt   = false;           // Apply adaptive volume shunt

//input string f = "$$$$$$$$$$$$$$ Anti Drawdown $$$$$$$$$$$$$$";
//input double      DrawdownPercent            = 25;             // Acceptable drawdown percentage %
//input double      WaitAfterDrawdownPips      = 1;              // Wait duration (in pips) after drawdown
//input bool        ApplyFastWait              = false;          // Fast recovery from drawdown.

//input int         KPeriod                    =  14; // KPeriod
//input int         Slowing                    =   3; // Slowing
//input int         DPeriod                    =   3; // DPeriod
//input ENUM_MA_METHOD MAMethod                =   2; // MAMethod
//input ENUM_STO_PRICE PriceField              =   0; // PriceField
//input int         overBought                 = 80;  // overBought
//input int         overSold                   = 20;  // overSold

//input int         FastK                      = 8;   // FastK
//input int         SlowK                      = 3;   // SlowK
//input int         SlowD                      = 3;   // SlowD

// Keltner Channel.
input int                  MA_Period         = 10;             // Keltner Channel MA Period
input ENUM_MA_METHOD       Mode_MA           = MODE_SMA;       // Keltner Channel MA Mode
input ENUM_APPLIED_PRICE   Price_Type        = PRICE_TYPICAL;  // Keltner Channel Price Type

input int         ATR_Period                 = 25;             // ATR Period       
input double      ATR_TickScalar             = 1000;           // ATR Tick Scalar

input int         ATR_Period_for_VolumeAdjust      = 14;       // ATR period for adaptive volume
input double      ATR_Multiplier_for_VolumeAdjust  = 1.5;      // ATR multiplier for adaptive volume
input double      MaxVolumeExponent                = 2.0;      // Max volume exponent for adaptive volume
input double      MinVolumeExponent                = 0.5;      // Min volume exponent for adaptive volume
input double      VolumeIncrement                  = 0.1;      // Volume step size for adaptive volume

input bool        closePriorsBeforeOrder           = false;          // Close Prior Orders Before New Order

input int         EveryMaxOrderIDCyclePeriod       = 30;             // Max order cycle peroid

input double      MaxDrawdownPercent               = 20.0; // Maximum allowed drawdown percentage
input int         DrawdownPeriod                   = 14; // Drawdown calculation period
input double      ReductionFactor                  = 0.5; // Order volume reduction factor


//input string e = "$$$$$$$$$$$$$$ Strategy $$$$$$$$$$$$$$";
input bool        ApplyBollingerBands        = false;          // Apply bollinger bands strategy
input bool        ApplyKeltnerChannel        = true;           // Apply keltner channel strategy
input bool        ApplySmartGrid             = false;          // Apply smart grid
input bool        ApplyTrailingStop          = false;          // Apply trailing stop
input bool        ApplyBreakeven             = false;          // Apply break even order mgmt
input bool        ApplyRiskRewardRatio       = false;          // Apply risk reward on tp order mgmt
input bool        Apply_ATR_LotResizing      = false;          // Apply ATR Market Trend Lot Resizing
input bool        ApplyAdaptiveVolumeExp     = false;          // Apply adaptive volume exponent
input bool        ApplyVolumeScalar          = false;          // Apply volume scalar multiplier
//input bool        ApplyDrawdownRiskManagement   = false;       // Apply drawdown risk management by extrapolation
input bool        ApplyNNetAdaption          = false;          // Apply neural net adaption
input bool        ApplyRandomWalk            = false;          // Apply random walk
//input bool        ApplyStochasticDiNapoli    = false;          // Apply Stochastic
//input bool        ApplyStochasticDiNapoli_v1 = false;          // Apply StochasticDiNapoli_v1
//input bool        ApplyStochasticDiNapoli_v2 = false;          // Apply StochasticDiNapoli_v2
input bool        ApplyHedging               = true;           // Apply hedging strategy
//input bool        AlwaysHedge                = false;          // ALWAYS Apply hedging strategy
//input bool        AlwaysProperHedge          = false;          // Apply proper hedging strategy (remember last volume)
//input bool        AlwaysProperHedge2         = false;          // Apply 2nd proper hedging strategy (remember last order volume)
//input bool        ApplyAntiDrawdown          = true;           // Apply anti-drawdown
input BL_EXP      ApplyExponential           = BL_EXPONENTIAL; // Apply exponential or linear lot sizes
input bool        AutoQuitOnCompleteClose    = false;          // Auto quit on complete close: when buys and sells are closed off in profit

string algoTitle = "Replicant";
bool continueBuyTrading = true;
bool continueSellTrading = true;
bool firstOrder = true;

// Global variables
double equityPrevious; // Previous equity
double currDrawdown; // Current drawdown
double trailingStop; // Trailing stop price
double breakevenPrice; // Breakeven price

string separator = ",";
//int countPipstepsRW;
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

double volumeExponent = 1.0;
double prevATR = 0.0;

double volumeScalar = 0.0;

double equity;
double maxEquity;
double drawdown;
double extrapolatedDrawdown;

double VolumeGlobal = 0.0;


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
   
   //parsedPipStepsRW[0] = PipStepsRW;
   parsedPipStepsBB[0] = PipStepsBB;
   //countPipstepsRW = 1;
   countPipstepsBB = 1;
   
   //ParsePipsteps(PipStepsRW, countPipstepsRW, parsedPipStepsRW);
   //ParsePipsteps(PipStepsBB, countPipstepsBB, parsedPipStepsBB);
    
   double lot_min  = MarketInfo(Symbol(),MODE_MINLOT);
   double lot_max  = MarketInfo(Symbol(),MODE_MAXLOT);
   Print("[Helpful info] Lot min: ", lot_min, " Lot max: ", lot_max);
   
   ShowOpenTrades();
   
   startingEquity = AccountEquity();
   
   equity = AccountEquity();
   maxEquity = equity;

   /*
   // Initialize weights
   for (int i = 0; i < hidden_count; i++)
   {
     for (int j = 0; j <= inputs_count; j++)
     {
       weights[i][j] = MathRand() / RAND_MAX - 0.5;
     }
   }
   */

   VolumeGlobal = FirstVolume;
   
   equityPrevious = AccountEquity();
   currDrawdown = 0;
   trailingStop = 0;
   breakevenPrice = 0;
       
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
          if(OrderMagicNumber() != MagicNumber) continue;
       
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

int GetCountWinningOpenPositions(BL_TRADE_TYPE tradeType)
{
   int count = 0;
   //for(int i=0;i<OrdersTotal();i++)
   for(int i=OrdersTotal()-1; i>=0; i--)
   {
       if(OrderSelect(i, SELECT_BY_POS, MODE_TRADES))
       {
          if(OrderMagicNumber() != MagicNumber) continue;
       
          double profit = OrderProfit() + OrderCommission() + OrderSwap();
          
          if(profit > 0)
          {
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
          if(OrderMagicNumber() != MagicNumber) continue;
       
          double profit = OrderProfit() + OrderCommission() + OrderSwap();

          if(OrderType()==OP_BUY  && OrderSymbol() == Symbol())
          {
            if(tradeType == BL_BUY)
            {
               netProfit += profit;
               count ++;
            }
          }
          if(OrderType()==OP_SELL && OrderSymbol() == Symbol())
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

double AdjustVolumeExpAdaptively(double volumeExp)
{
   if(!ApplyAdaptiveVolumeExp){
      return volumeExp;
   }
   
   if(volumeExponent == 1.0){
      volumeExponent = volumeExp;
   }
   
   int period = PERIOD_CURRENT;
   double tickValue = MarketInfo(Symbol(), MODE_TICKVALUE);
   double atr = iATR(Symbol(), period, ATR_Period_for_VolumeAdjust, 0);
   double newVolumeExponent = volumeExponent;
   
   if(prevATR > 0)
   {
      if((atr * tickValue) > prevATR * tickValue * ATR_Multiplier_for_VolumeAdjust)
      {
         newVolumeExponent -= VolumeIncrement;
      }
      else if((atr * tickValue) < (prevATR * tickValue) / ATR_Multiplier_for_VolumeAdjust)
      {
         newVolumeExponent += VolumeIncrement;
      }
   }
   
   if(newVolumeExponent > MaxVolumeExponent) newVolumeExponent = MaxVolumeExponent;
   if(newVolumeExponent < MinVolumeExponent) newVolumeExponent = MinVolumeExponent;
   
   //if(newVolumeExponent != volumeExponent)
   {
      volumeExponent = newVolumeExponent;
      // Adjust volume exponent parameter in your trading strategy here
   }
   
   prevATR = atr;

   return volumeExponent;
}
double orderID = 0.0;
double CalculateVolume(BL_TRADE_TYPE tradeType, bool hedgingPosition)
{
   orderID += 1.0f;
   
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
   //double riskB = RiskPercentB / 100.0;
   
   double  ab  = AccountBalance();
   //risk = MathSqrt(RiskMultiplier * ab)/ab; // ~const rate.
   //risk = MathSqrt(RiskMultiplier * ab * MathPow( 1 - risk, OrdersTotal() ));
   
   double numGroups;
   
   //if(numTrades < InitialTrades)
   //{
   //   numGroups = InitialNumGroups;
   //}
   //else if(isDrawdown)
   //{
   //   numGroups = DrawdownNumGroups;
   //}
   //else
   //{
      numGroups = NumGroups;
   //}
   
   //double accountMargin = AccountFreeMargin();
   double accountMargin = AccountFreeMargin();
   accountMargin = AccountEquity();
   //accountMargin = AccountBalance();
   
   double lotMultiplier = 0;
   
   if(accountMargin > PercentIncreaseThreshold)
   {
      //lotMultiplier = ((accountMargin) / PercentIncreaseThreshold100)* risk *AccountLeverage()/contract;
      //lotMultiplier = ((accountMargin) / PercentIncreaseThreshold100) * risk;
      //lotMultiplier = ((accountMargin * risk) / PercentIncreaseThreshold100);
   //}
   //else{
      //lotMultiplier = ((accountMargin) / PercentIncreaseThreshold) * risk *AccountLeverage()/contract;
      lotMultiplier = ((accountMargin) / PercentIncreaseThreshold) * risk;
      //lotMultiplier = ((accountMargin * risk) / PercentIncreaseThreshold);
   }

   //lotMultiplier = ((accountMargin) / 10000) * risk;

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
   //if(hedgingPosition)
   //{
   //   FirstOrderVolume *= HedgeLotScalar;
   //}
   
   //double ae;
   
   //if(AccountEquity() > startingEquity)
   //{  
      //double ee = (AccountEquity() * (LotIncreasePercent / 100.0));
   //   ae = (((AccountEquity()) / (EquityInterval))) * RiskRatio;
   //}
   //else 
   //{
   //    ae = 1.0;
   //}
   //if(ae <= 0)
   //{
   //   ae = 1;
   //}

   //ae = MathPow(EquityExponent, ae);
   

   
   //if(isDrawdown)
   //{
   //   ae = 1;
   //}
   
   //if(ae >= MaxExp)
   //{
   //   ae = MaxExp;
   //}

   double volume = FirstOrderVolume;
   double volumeExp = VolumeExponent;
   
   //if(numTrades < InitialTrades)
   //{
   //   volumeExp = InitialVolumeExponent;
   //}
   
   volumeExp = AdjustVolumeExpAdaptively(volumeExp);

   if(ApplyExponential == BL_LINEAR)
   {
      volume =  ((lotMultiplier + FirstOrderVolume) * MathPow(volumeExp, PowerOffset + ((PowerFactor * countPositions) / numGroups)));
   }
   else if (ApplyExponential == BL_EXPONENTIAL)
   {
      volume =  /* ae * */ ((lotMultiplier + FirstOrderVolume) * MathPow(volumeExp, PowerOffset + ((PowerFactor * (countPositions)) / numGroups)));
   }
   else if (ApplyExponential == BL_SMALL_EXPONENTIAL)
   {
      volume =  /* ae * */ FirstOrderVolume * MathPow(volumeExp, PowerOffset + ((PowerFactor * (countPositions)) / numGroups));   
      //volume =  ae * ((((FirstOrderVolume * ((1+countPositions)/ numGroups))))) ;
   }
   
   //if(numTrades <= InitialNumTradesLinear)
   //{
      // Linear
   //   volume =  ((lotMultiplier + FirstOrderVolume) * MathPow(volumeExp, PowerOffset + ((PowerFactor * countPositions) / numGroups)));
   //}
   //else
   //{
      //double ee = (volume * (LotIncreasePercent / 100.0));
      
      //if(isDrawdown)
      //{
         //volume += ee;   
      //}
   //}
   //volume =  ((lotMultiplier + FirstOrderVolume) * MathPow(VolumeExponent, PowerOffset + ((PowerFactor * countPositions) / numGroups)));
   //volume =  ae * ((lotMultiplier + FirstOrderVolume) * MathPow(VolumeExponent, PowerOffset + ((PowerFactor * countPositions) / numGroups)));
  
   if(ApplyVolumeScalar)
   {
      if(volumeScalar == 0.0)
      {
         volumeScalar = 1.0f;
      }
      if(orderID >= EveryMaxOrderIDCyclePeriod)
      {
         orderID = 1;
      }
      volume *= FirstOrderVolume * (orderID) * InitialVolumeScalar;
   }
  
  
   if(ApplyAdaptiveVolumeShunt)
   {
      double netProfit = GetAverageNetProfit(tradeType);
     
      if (netProfit < 0)
      {
        VolumeGlobal *= MathPow(volumeExponent, -volumeExponential);
      }
      else
      {
        VolumeGlobal *= MathPow(volumeExponent, volumeExponential);
      }
      volume = VolumeGlobal;
   }
  
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
   
   //if(numTrades < InitialTrades)
   //{
   //   numGroups = InitialNumGroups;
   //}
   //else if(isDrawdown)
   //{
   //   numGroups = DrawdownNumGroups;
   //}
   //else
   //{
      numGroups = NumGroups;
   //}
   
   //double accountMargin = AccountFreeMargin();
   double accountMargin = AccountFreeMargin();
   accountMargin = AccountEquity();
   //accountMargin = AccountBalance();
   
   double lotMultiplier = 0;
   
   if(accountMargin > PercentIncreaseThreshold)
   {
   //   lotMultiplier = ((accountMargin) / PercentIncreaseThreshold100) * risk;
   //}
   //else{
      lotMultiplier = ((accountMargin) / PercentIncreaseThreshold) * risk;
   }
   
   double FirstOrderVolume = GetFirstOrderVolume();
//   if(hedgingPosition)
//   {
//      FirstOrderVolume *= HedgeLotScalar;
//   }
   
   //double ae;
   
   //if(AccountEquity() > startingEquity)
   //{  
   //   //double ee = (AccountEquity() * (LotIncreasePercent / 100.0));
   //   ae = (((AccountEquity()) / (EquityInterval))) * RiskRatio;
   //}
   //else 
   //{
   //    ae = 1.0;
   //}
   //if(ae <= 0)
   //{
   //   ae = 1;
   //}

   //ae = MathPow(EquityExponent, ae);
   

   
   //if(isDrawdown)
   //{
    //  ae = 1;
   //}
   
   //if(ae >= MaxExp)
   //{
   //   ae = MaxExp;
   //}

   double volume = FirstOrderVolume;
   double volumeExp = VolumeExponent;
   
   //if(numTrades < InitialTrades)
   //{
   //   volumeExp = InitialVolumeExponent;
   //}

   volumeExp = AdjustVolumeExpAdaptively(volumeExp);   
   
   if(ApplyExponential == BL_LINEAR)
   {
      volume =  ((lotMultiplier + FirstOrderVolume) * MathPow(volumeExp, PowerOffset + ((PowerFactor * countPositions) / numGroups)));
   }
   else if (ApplyExponential == BL_EXPONENTIAL)
   {
      volume =  /* ae * */ ((lotMultiplier + FirstOrderVolume) * MathPow(volumeExp, PowerOffset + ((PowerFactor * (countPositions)) / numGroups)));
   }
   else if (ApplyExponential == BL_SMALL_EXPONENTIAL)
   {
      volume =  /* ae * */ FirstOrderVolume * MathPow(volumeExp, PowerOffset + ((PowerFactor * (countPositions)) / numGroups));   
      //volume =  ae * ((((FirstOrderVolume * ((1+countPositions)/ numGroups))))) ;
   }
   
   //if(numTrades <= InitialNumTradesLinear)
   //{
      // Linear
   //   volume =  ((lotMultiplier + FirstOrderVolume) * MathPow(volumeExp, PowerOffset + ((PowerFactor * countPositions) / numGroups)));
   //}
   //else
   //{
   //   double ee = (volume * (LotIncreasePercent / 100.0));
      
      //if(isDrawdown)
      //{
   //      volume += ee;   
      //}
   //}
   //volume =  ((lotMultiplier + FirstOrderVolume) * MathPow(VolumeExponent, PowerOffset + ((PowerFactor * countPositions) / numGroups)));
   //volume =  ae * ((lotMultiplier + FirstOrderVolume) * MathPow(VolumeExponent, PowerOffset + ((PowerFactor * countPositions) / numGroups)));
  
   if(ApplyAdaptiveVolumeShunt)
   {
      double netProfit = GetAverageNetProfit(tradeType);
     
      if (netProfit < 0)
      {
        VolumeGlobal *= MathPow(volumeExponent, -volumeExponential + ((PowerOffset + ((PowerFactor * countPositions) / numGroups))));
      }
      else
      {
        VolumeGlobal *= MathPow(volumeExponent, volumeExponential + ((PowerOffset + ((PowerFactor * countPositions) / numGroups))));
      }
      volume = VolumeGlobal;
   }
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

double ATR_LotSize(double volume)
{
   double atr = iATR(Symbol(), Period(), ATR_Period, 0);
   double atrValue = NormalizeDouble(atr, Digits);
   double tickValue = MarketInfo(Symbol(), MODE_TICKVALUE);
   double atrAdjuster = 1.0 / (atrValue * tickValue * ATR_TickScalar);
   double lotSize = MathFloor(volume * atrAdjuster);
   if(Apply_ATR_LotResizing)
   {
      return lotSize;
   }
   return volume;
}

double ExecuteMarketOrder(BL_TRADE_TYPE tradeType, double& volumeOut, double volumeIn, bool first)
{
   int ticketId = -1;
   double price = 0;
   double priceOrig = 0;
   double volume = 0;
   
   bool hedging = ApplyHedging; 
   
   //if(AlwaysHedge && first)
   //{
      hedging = true;
   //}
   //else if(AlwaysHedge && !first)
   //{
   //   hedging = true;
   //}
   //if(ApplyHedging && !AlwaysHedge && first)
   //{
   //   hedging = false;
   //}
   //if(ApplyHedging && !AlwaysHedge && !first)
   //{
   //   hedging = true;
   //}
   
   if(tradeType == BL_BUY && continueBuyTrading)
   {
      price = GetPrice(BL_BUY);
      priceOrig = price;
      
      //if(volumeIn > 0 && AlwaysProperHedge)
      //{
      //   volume = volumeIn;
      //}
      //else
      //{
      //   volume = CalculateVolume(BL_BUY, false);
      //}
      volume = CalculateVolume(BL_BUY, false);
      
      //if(previousSellVolume > 0 && AlwaysProperHedge2)
      //{
      //   volume = ExponentialVolume(BL_BUY, false, previousSellPositions);
      //   previousSellVolume = 0;
      //}
      
      previousBuyVolume = volume;
      previousBuyPositions = GetCountOpenPositions(BL_BUY);
      
      volume = ATR_LotSize(volume);
      
      ticketId = OrderSend(Symbol(), OP_BUY, volume, price, GetSlippage(), 0, 0, algoTitle, MagicNumber, 0, clrGreen);

      if(ticketId < 0)
      {
         Print("OrderSend error #", GetLastError(), " ", ErrorDescription(GetLastError()));
      }
      
      if(hedging)
      {
         price = GetPrice(BL_SELL);
         
         volume = CalculateVolume(BL_SELL, true);
         
         //if(!AlwaysProperHedge)
         //{
         //   volume = CalculateVolume(BL_SELL, true);
         //}
         
         //if(previousBuyVolume > 0 && AlwaysProperHedge2)
         //{
         //   volume = ExponentialVolume(BL_SELL, true, previousBuyPositions); 
         //}
         //else if(previousBuyVolume == 0 && AlwaysProperHedge2)
         //{
         //   volume = CalculateVolume(BL_SELL, true);
         //}
         
         volume = ATR_LotSize(volume);
         
         ticketId = OrderSend(Symbol(), OP_SELL, volume, price, GetSlippage(), 0, 0, algoTitle + " hedging", MagicNumber, 0, clrGreen);
   
         if(ticketId < 0)
         {
            Print("OrderSend error #", GetLastError(), " ", ErrorDescription(GetLastError()));
         }
      }
   }
   else if(tradeType == BL_SELL && continueSellTrading)
   {
      price = GetPrice(BL_SELL);
      priceOrig = price;
      
      //if(volumeIn > 0 && AlwaysProperHedge)
      //{
      //   volume = volumeIn;
      //}
      //else
      //{
      //   volume = CalculateVolume(BL_SELL, false);
      //}
      volume = CalculateVolume(BL_SELL, false);
      
      //if(previousBuyVolume > 0 && AlwaysProperHedge2)
      //{
      //   volume = ExponentialVolume(BL_SELL, false, previousBuyPositions);
      //   previousBuyVolume = 0;
      //}
      
      previousSellVolume = volume;
      previousSellPositions = GetCountOpenPositions(BL_SELL);
      
      volume = ATR_LotSize(volume);
      
      ticketId = OrderSend(Symbol(), OP_SELL, volume, price, GetSlippage(), 0, 0, algoTitle, MagicNumber, 0, clrRed);
      
      if(ticketId < 0)
      {
         Print("OrderSend error #", GetLastError());
      }
      
      if(hedging)
      {
         price = GetPrice(BL_BUY);
         //if(!AlwaysProperHedge)
         //{
            volume = CalculateVolume(BL_BUY, true);
         //}
         
         //if(previousSellVolume > 0 && AlwaysProperHedge2)
         //{
         //   volume = ExponentialVolume(BL_BUY, true, previousSellPositions);
         //}
         //else if(previousSellVolume == 0 && AlwaysProperHedge2)
         //{
         //   volume = CalculateVolume(BL_BUY, true);
         //}
         
         volume = ATR_LotSize(volume);
      
         ticketId = OrderSend(Symbol(), OP_BUY, volume, price, GetSlippage(), 0, 0, algoTitle + " hedging", MagicNumber, 0, clrGreen);
   
         if(ticketId < 0)
         {
            Print("OrderSend error #", GetLastError());
         }
      }
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
          if(OrderMagicNumber() != MagicNumber) continue;
       
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
          if(OrderMagicNumber() != MagicNumber) continue;
       
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
          if(OrderMagicNumber() != MagicNumber) continue;
       
          if(OrderType()==OP_BUY  && OrderSymbol() == Symbol())
          {
            if(tradeType == BL_BUY)
            {
               result = OrderClose(OrderTicket(), OrderLots(), GetClosingPrice(BL_BUY), GetSlippage(), clrYellow); // Bid
            }
          }
          if(OrderType()==OP_SELL && OrderSymbol() == Symbol())
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
   
   //double targetLoss = FirstOrderVolume * StopLoss * GetPipSize() * GetPipScalar() * 1000000;
   
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
             if(OrderMagicNumber() != MagicNumber) continue;
          
             if(OrderType()==OP_BUY  && OrderSymbol() == Symbol())
             {
               numOpenOrders ++;
               netProfit += OrderProfit() + OrderCommission() + OrderSwap();
             }
             if(OrderType()==OP_SELL && OrderSymbol() == Symbol())
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
      }
   }
   else
   {
      //for(int i=0;i<OrdersTotal();i++)
      for(int i=OrdersTotal()-1; i>=0; i--)
      {
          if(OrderSelect(i, SELECT_BY_POS, MODE_TRADES))
          {
             if(OrderMagicNumber() != MagicNumber) continue;
          
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
         averageBuyNetProfit = buyNetProfit / (double)numOpenBuy;
         
         //Print("averageBuyNetProfit: ", averageBuyNetProfit);
         
         //Print("averageBuyNetProfit: ", averageBuyNetProfit, " targetProfit: ", targetProfit);
         
         if(averageBuyNetProfit >= targetProfit)
         {
            Print("Closing BUY trades - AT PROFIT");
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
      }
   }
}
        
void CloseBuy()
{
   CloseTrades(BL_BUY);
   /*
    for (int i = OrdersTotal() - 1; i >= 0; i--)
    {
        if (OrderSelect(i, SELECT_BY_POS) && OrderType() == OP_BUY)
        {
            OrderClose(OrderTicket(), OrderLots(), Bid, Slippage, Red);
        }
    }*/
}

void CloseSell()
{
   CloseTrades(BL_SELL);
   /*
    for (int i = OrdersTotal() - 1; i >= 0; i--)
    {
        if (OrderSelect(i, SELECT_BY_POS) && OrderType() == OP_SELL)
        {
            OrderClose(OrderTicket(), OrderLots(), Ask, Slippage, Green);
        }
    }*/
}

double sigmoid(double x)
{
   return 1.0 / (1.0 + MathExp(-x));
}


input int NNET_inputs_count = 5; // Number of inputs to the neural network
input int NNET_hidden_count = 10; // Number of hidden neurons
input int NNET_outputs_count = 2; // Number of outputs (buy/sell signals)
input double NNET_learningRate = 0.1; // Learning rate
input double NNET_momentum = 0.5; // Momentum
input double NNet_TrailStopLoss = 50; // Stop loss
input double NNET_takeProfit = 100; // Take profit

// Global variables
//double weights[][inputs+1]; // Weights for each neuron
//double hiddenValues[hidden]; // Hidden neuron values
//double outputValues[outputs]; // Output values (buy/sell signals)


void NNetAdaption()
{
/*
   // Get inputs
   double inputs[] = {Close[0], Close[1], Close[2], Close[3], Close[4]};
   
   // Forward pass
   for (int i = 0; i < NNET_hidden_count; i++)
   {
      double sum = 0;
      for (int j = 0; j < NNET_inputs_count; j++)
      {
        sum += weights[i][j] * inputs[j];
      }
      sum += weights[i][NNET_inputs_count]; // Bias
      hiddenValues[i] = sigmoid(sum);
   }
   
   for (int i = 0; i < NNET_outputs_count; i++)
   {
      double sum = 0;
      for (int j = 0; j < NNET_hidden_count; j++)
      {
        sum += weights[j][i] * hiddenValues[j];
      }
      outputValues[i] = sigmoid(sum);
   }
   
   // Generate signals
   if (outputValues[0] > outputValues[1])
   {
      for(int i=OrdersTotal()-1; i>=0; i--)
      {
         // Buy signal
         if (OrderSelect(i, SELECT_BY_POS) && OrderType() == OP_BUY)
         {
            // Trailing stop
            if (Bid - OrderOpenPrice() > NNet_TrailStopLoss * Point)
            {
               if(OrderClose(OrderTicket(), OrderLots(), Bid, Slippage))
               {
               
               }
               //OrderSend(Symbol(), OP_SELL, OrderLots(), Ask, 3, 0, 0, "Sell", 0, 0, Green);
               DoSell_BB();
            }
         }
         else
         {
           //OrderSend(Symbol(), OP_BUY, 0.1, Ask, 3, 0, 0, "Buy", 0, 0, Green);
           DoBuy_BB();
         }
      }
   }
   else
   {
      for(int i=OrdersTotal()-1; i>=0; i--)
      {
         // Sell signal
         if (OrderSelect(i, SELECT_BY_POS) && OrderType() == OP_SELL)
         {
            // Trailing stop
            if (OrderOpenPrice() - Ask > NNet_TrailStopLoss * Point)
            {
               if(OrderClose(OrderTicket(), OrderLots(), Ask, Slippage))
               {
               
               }
               //OrderSend(Symbol(), OP_BUY, OrderLots(), Bid, 3, 0, 0, "Buy", 0, 0, Green);
               DoBuy_BB();
            }
         }
         else
         {
           //OrderSend(Symbol(), OP_SELL, 0.1, Bid, 3, 0, 0, "Sell", 0, 0, Red);
           DoSell_BB();
         } 
      }
   }
   */
}


//+------------------------------------------------------------------+
//| Finds the moving average of the price ranges                     |
//+------------------------------------------------------------------+
double findAvg(int period, int shift)
{
    double sum = 0;

    for (int i = shift; i < (shift + period); i++){
        sum += High[i] - Low[i];
    }
    sum = sum / period;

    return(sum);
}

int KeltnerChannel()
{
   int period = PERIOD_CURRENT;
   double Upper;
   double Middle; 
   double Lower;

   int limit;
   double avg;
   int counted_bars = IndicatorCounted();
   //if (counted_bars < 0) return -1;
   //if (Bars < MA_Period) return 0;
   //if (counted_bars > 0) counted_bars--;
   
   limit = Bars - counted_bars;
   if (limit > Bars - MA_Period) 
   {
      limit = Bars - MA_Period;
   }
   
   if(Bars > 2)
   {
      for (int i = 0; i < limit; i++)
      {
         Middle = iMA(Symbol(), period, MA_Period, 0, Mode_MA, Price_Type, i); // period
         avg = findAvg(MA_Period, i);
         Upper = Middle + avg;
         Lower = Middle - avg;
        
         int positionType = 0; // 0: no position, 1: buy, -1: sell
   
         /*
         double extrapolate0 = 0;
         double extrapolate1 = 0;
         double gradient = 0;
         if(i > 0)
         {
            extrapolate0 = Close[i];
            extrapolate1 = Close[i-1];
            gradient = (extrapolate0 - extrapolate1) / extrapolate1;
            
            if(gradient > 1.0)
            {
               CloseSell2();
            }
            else if (gradient < 0.0)
            {
               //CloseBuy2();
            }
            else if (gradient == 0)
            {
            
            }
         }
         */
   
         if(Close[i] >= Upper)
         {
            // Buy when candle closes above the upper band.
            if(closePriorsBeforeOrder)
            {            
               if (positionType != 1)
               {
                   if (positionType == -1)
                   {
                       CloseSell();
                   }
                   //OpenBuy();
                   DoBuy_BB();
                   positionType = 1;
               }
            }
            else
            {
               DoBuy_BB();
            }
         }
         if(Close[i] <= Lower)
         {
            // Sell when candle closes below the lower band.
            if(closePriorsBeforeOrder)
            {
               if (positionType != -1)
               {
                   if (positionType == 1)
                   {
                       CloseBuy();
                   }
                   //OpenSell();
                   DoSell_BB();
                   positionType = -1;
               }
            }
            else 
            {
               DoSell_BB();
            }
         }
         
      }
   }

   return 0;
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
   
   if((countConsecutiveBuys >= MaxConsecutiveBuys) && (countConsecutiveSells >= MaxConsecutiveSells))
   {
      countConsecutiveSells = 0;
      countConsecutiveBuys = 0;
   }
   
   if(Bars > 2)
   {
       //double upperBand = MODE_UPPER;
       //double lowerBand = 1;
       int positionType = 0; // 0: no position, 1: buy, -1: sell
   
       // Calculate Bollinger Bands
       if (Close[2] > BB_UPPER2)//boll.Top.Last(2))
       {
           if (Close[1] > BB_UPPER1)//boll.Top.Last(1))
           {
               countConsecutiveSells = 0;
               
               if(countConsecutiveBuys <= MaxConsecutiveBuys)
               {                  
                  if(closePriorsBeforeOrder)
                  {
                     if (positionType != 1)
                     {
                         if (positionType == -1)
                         {
                             CloseSell();
                         }
                         //OpenBuy();
                         DoBuy_BB();
                         positionType = 1;
                     }
                  }
                  else
                  {
                     //SmartGrid();
                     //CloseSell2();
                     //CloseBuy2();
                     DoBuy_BB();
                  }
                  countConsecutiveBuys++;
               }
           }
           else
           {
               countConsecutiveBuys = 0;
               
               if(countConsecutiveSells <= MaxConsecutiveSells)
               {                  
                  if(closePriorsBeforeOrder)
                  {
                     if (positionType != -1)
                     {
                         if (positionType == 1)
                         {
                             CloseBuy();
                         }
                         //OpenSell();
                         DoSell_BB();
                         positionType = -1;
                     }
                  }
                  else
                  {
                     //SmartGrid();
                     //CloseBuy2();
                     //CloseSell2();
                     DoSell_BB();
                  }
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
                  if(closePriorsBeforeOrder)
                  {
                     if (positionType != -1)
                     {
                         if (positionType == 1)
                         {
                             CloseBuy();
                         }
                         //OpenSell();
                         DoSell_BB();
                         positionType = -1;
                     }
                  }
                  else
                  {
                     //SmartGrid();
                     //CloseSell2();
                     //CloseBuy2();
                     DoSell_BB();
                  }
                  countConsecutiveSells++;
               }
           }
           else
           {
               countConsecutiveSells = 0;
               
               if(countConsecutiveBuys <= MaxConsecutiveBuys)
               {               
                  if(closePriorsBeforeOrder)
                  {
                     if (positionType != 1)
                     {
                         if (positionType == -1)
                         {
                             CloseSell();
                         }
                         //OpenBuy();
                         DoBuy_BB();
                         positionType = 1;
                     }
                  }
                  else
                  {
                     //SmartGrid();
                     //CloseBuy2();
                     //CloseSell2();
                     DoBuy_BB();
                  }
                  countConsecutiveBuys++;
               }
           }
       }
       
       /*
       if (iBands(Symbol(), period, BBPeriod, BBDeviation, BBShift, PRICE_CLOSE, upperBand, lowerBand) == 0)
       {
           double currentPrice = Ask;
   
           // Check for Buy Signal
           if (currentPrice < lowerBand)
           {
               if (positionType != 1)
               {
                   if (positionType == -1)
                   {
                       CloseSell();
                   }
                   //OpenBuy();
                   DoBuy_BB();
                   positionType = 1;
               }
           }
   
           currentPrice = Bid;
           
           // Check for Sell Signal
           if (currentPrice > upperBand)
           {
               if (positionType != -1)
               {
                   if (positionType == 1)
                   {
                       CloseBuy();
                   }
                   //OpenSell();
                   DoSell_BB();
                   positionType = -1;
               }
           }
       }
       */
       
   /*
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
       */
       
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
      DoBuy_BB();
   }
   else 
   {
      DoSell_BB();
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
          if(OrderMagicNumber() != MagicNumber) continue;
       
          if(OrderType()==OP_BUY  && OrderSymbol() == Symbol())
          {
            profit = OrderProfit() + OrderCommission() + OrderSwap();
          
            if(profit < 0)
            {
               result = OrderClose(OrderTicket(), OrderLots(), GetClosingPrice(BL_BUY), GetSlippage(), clrYellow);
            }
            
          }
          if(OrderType()==OP_SELL && OrderSymbol() == Symbol())
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
          if(OrderMagicNumber() != MagicNumber) break;
       
          if(OrderType()==OP_BUY  && OrderSymbol() == Symbol())
          {
            profit = OrderProfit() + OrderCommission() + OrderSwap();
            
            Print("@@@@@@@@@@@@@@@@@@@@@@ ORDER [BUY] index: ",i," profit: ", profit);
          }
          if(OrderType()==OP_SELL && OrderSymbol() == Symbol())
          {
            profit = OrderProfit() + OrderCommission() + OrderSwap();
            
            Print("@@@@@@@@@@@@@@@@@@@@@@ ORDER [BUY] index: ",i," profit: ", profit);
          }
       }
   }
}

bool CheckSpread()
{  
   double spread = MarketInfo(Symbol(), MODE_SPREAD);
   //double spreadPips = spread / Point();
   //double spreadPips = spread / (GetPipSize() * GetPipScalar());
   
   //Print("spread: ", spread, " ", spreadPips);
   
   return (spread <= MaxSpread);
}

void DrawdownRiskManagement()
{
   int period = PERIOD_CURRENT;

   // Update equity and max equity
   equity = AccountEquity();
   if (equity > maxEquity) maxEquity = equity;
   
   // Calculate drawdown
   drawdown = ((maxEquity - equity) / maxEquity) * 100;
   
   // Extrapolate drawdown
   //extrapolatedDrawdown = drawdown + (drawdown - iMA(drawdown, period, DrawdownPeriod, 0, MODE_SMA, PRICE_CLOSE, 0)) * 2;
   
   /*
   double x = period;
   double y = drawdown;
   double slope = 0.5;
   
   double nextDrawdown = y + slope * x;
   extrapolatedDrawdown = nextDrawdown;
   extrapolatedDrawdown = drawdown;
   
   // Check if drawdown exceeds maximum allowed
   if (extrapolatedDrawdown > MaxDrawdownPercent)
   {     
      for(int i=OrdersTotal(); i>=0; i--)
      {
         if(OrderSelect(i,SELECT_BY_POS,MODE_TRADES))
         {
            //make sure its the right currency pair
            if(OrderSymbol()==_Symbol && OrderMagicNumber() == MagicNumber)
            {
               // Reduce order volume
               double newVolume = OrderLots() * ReductionFactor;
               
               if(OrderLots() != newVolume)
               {
                  if(OrderModify(OrderTicket(), OrderOpenPrice(), OrderTakeProfit(), OrderStopLoss(), newVolume))
                  {
                  
                  }
                  else{

                  }
               }
            }
         }
      }
   }
   */
}

void DrawdownBreakEven()
{
   // Calculate drawdown
   currDrawdown = (equityPrevious - AccountEquity()) / equityPrevious * 100;
   
   // Check if drawdown exceeds maximum allowed
   if (currDrawdown > maxDrawdownPercent)
     {
      // Close all orders
      for (int i = OrdersTotal() - 1; i >= 0; i--)
        {
         if (OrderSelect(i, SELECT_BY_POS))
           {
            if (OrderType() == OP_BUY)
              OrderClose(OrderTicket(), OrderLots(), Bid, 3);
            else
              OrderClose(OrderTicket(), OrderLots(), Ask, 3);
           }
        }
     }
   
   if(ApplyTrailingStop)
   {
      // Trailing stop
      if (OrderSelect(0, SELECT_BY_POS) && OrderType() == OP_BUY)
        {
         if (Bid - OrderOpenPrice() > trailingStopDistance * Point)
           {
            trailingStop = Bid - trailingStopDistance * Point;
            OrderModify(OrderTicket(), OrderOpenPrice(), trailingStop, OrderTakeProfit(), 0);
           }
        }
      else if (OrderSelect(0, SELECT_BY_POS) && OrderType() == OP_SELL)
        {
         if (OrderOpenPrice() - Ask > trailingStopDistance * Point)
           {
            trailingStop = OrderOpenPrice() + trailingStopDistance * Point;
            OrderModify(OrderTicket(), OrderOpenPrice(), OrderTakeProfit(), trailingStop, 0);
           }
        }
   }
   
   if(ApplyBreakeven)
   {
      // Breakeven
      if (OrderSelect(0, SELECT_BY_POS) && OrderType() == OP_BUY)
        {
         if (Bid - OrderOpenPrice() > breakevenDistance * Point)
           {
            breakevenPrice = Bid - breakevenDistance * Point;
            OrderModify(OrderTicket(), OrderOpenPrice(), breakevenPrice, OrderTakeProfit(), 0);
           }
        }
      else if (OrderSelect(0, SELECT_BY_POS) && OrderType() == OP_SELL)
        {
         if (OrderOpenPrice() - Ask > breakevenDistance * Point)
           {
            breakevenPrice = OrderOpenPrice() + breakevenDistance * Point;
            OrderModify(OrderTicket(), OrderOpenPrice(), OrderTakeProfit(), breakevenPrice, 0);
           }
        }
   }
   
   if(ApplyRiskRewardRatio)
   {
      // Risk-reward ratio
      if (OrderSelect(0, SELECT_BY_POS) && OrderType() == OP_BUY)
        {
         if (Bid - OrderOpenPrice() > OrderTakeProfit() / riskRewardRatio)
           {
            OrderClose(OrderTicket(), OrderLots(), Bid, 3);
           }
        }
      else if (OrderSelect(0, SELECT_BY_POS) && OrderType() == OP_SELL)
        {
         if (OrderOpenPrice() - Ask > OrderTakeProfit() / riskRewardRatio)
           {
            OrderClose(OrderTicket(), OrderLots(), Ask, 3);
           }
        }
   }
}

//************************************************************************************************/
//*                                                                                              */
//************************************************************************************************/
void OnTick()
{
   IsNewBar = IsThisANewCandle();
   
//*************************************************************//
   if(IsNewBar /* && CheckSpread() */ /*&& DrawdownMonitor()*/ /*&& CheckSpread()*/)
   {
      
      if(ApplySmartGrid)
      {
         SmartGrid();
      }
      //TrailingStop();
      DrawdownBreakEven();
      
      AutoQuitOnClose();
     
      //if(ApplyDrawdownRiskManagement)
      //{
      //   DrawdownRiskManagement();
      //}
   
      //if(ApplyStochasticDiNapoli)
      //{
      //   Stochastic();
      //}
      if(ApplyBollingerBands)
      {
         BollingerBands();
      }
      if(ApplyKeltnerChannel)
      {
         KeltnerChannel();
      }
      if(ApplyRandomWalk)
      {
         RandomWalk();
      }
      if(ApplyNNetAdaption)
      {
         NNetAdaption();
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