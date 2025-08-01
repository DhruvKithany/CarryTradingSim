import random
import math
from datetime import datetime, timedelta

#for Animation
import time
import os
import sys


class CarryTradeBacktest:
    
    #Training Sim 2010-2014, 'building the model'
    #Testing Sim  2015-2019 (test performance)
    def __init__(self):
        self.currencies = ['USD', 'EUR', 'JPY', 'GBP', 'AUD', 'CAD', 'CHF', 'NZD', 'SEK', 'NOK']
        self.training_startYear = 2010
        self.training_endYear = 2014
        self.testing_startYear = 2015
        self.testing_endYear = 2019
        
        
        #Fake but realistic interest rate and FX return data for each currency (Not enough time to integrate a full fledged data set)
        # Simulates interest rates (A declining trend of current + randomness)
        #FX Retyrbs
    def generateMockData(self):
        """Generate realistic mock data for interest rates and FX returns"""
        # Create monthly periods (10 years * 12 months = 120 months)
        #200-01, 2010-02, etc is made here, this is for the end tables
        months = []
        for year in range(2010, 2020):
            for month in range(1, 13):
                months.append(f"{year}-{month:02d}")
        
        # Set seed for reproducibility
        #najes sure each time i run the code it is the same "random" numbers (For Testing + Presenting Purpouses)
        random.seed(42)
        
        # Base interest rates (annualized %)
        #These are typical annual interest rates for 10 major currencies, starting inital points for the data (randomly adjusted later on to simulate realistic changes)
        baseRates = {
            'USD': 2.0, 'EUR': 1.5, 'JPY': 0.1, 'GBP': 2.5, 'AUD': 4.5,
            'CAD': 2.8, 'CHF': 0.5, 'NZD': 4.0, 'SEK': 2.2, 'NOK': 3.5
        }
        
        # Generate time-varying interest rates
        interestData = {}
        fxData = {}
        
        #basic loop for each currency 
        for currency in self.currencies:
            
            #store the interest rates and forex rates (Simulated)
            interestRates = []
            fxReturns = []
            
            # Generate interest rates with trend and "noise"
            
            #base gets the abse annual interest rate for the currency from the previous dictionary 
            base = baseRates[currency]
            
            #Gaussian noise, simulate randomness (Remember, this is a simulation, we did not integrate current data so this is needed for realistic random)
            cumulativeNoise = 0
            
            for i, month in enumerate(months):
                # Add declining trend and random noise
                trend = -0.5 * (i / len(months))  # Declining over time
                
                #Add a little randomness, (Simulate economic shocks or fluctuations, just like real world has rando changes)
                cumulativeNoise += random.gauss(0, 0.1)
                
                #Current month interest rate as a combo of the randomness, the abse rate, and the predicted trend it moves, makes sure it stays above 0
                rate = max(0.0, base + trend + cumulativeNoise)
                
                #bring back to the interest rates list
                interestRates.append(rate)
                
                # Generate FX returns with carry trade effect
                
                if currency == 'USD':
                    fxReturns.append(0.0)  # USD vs USD = 0 (If its carried to itself theres no change)
                else:
                    #as long as its not it self it calculates a simulated carry trade effect, the interest rate differential between whatever chosen currency and USD
                    
                    #0.1 to scale it down a bit (like assuming 10% of the interest spread is actually priced into the FX)
                    carryEffect = (rate - interestRates[0] if i == 0 else  ##ubterest rates[0] is like i==0 since tech data not generated yet (would change if incorportated an actual data set)
                                  rate - interestData['USD'][i]) * 0.1
                    
                    
                    #Random FX movement using a gaussian distrubition (3% SD, simulate world randomness with currency changes)
                    randomReturn = random.gauss(0, 0.03)  # 3% monthly volatility
                    
                    #Combines carry affect + random currency movement into a final FX return for the cumulative month
                    fxReturns.append(carryEffect + randomReturn)
            
            
            #After looping all months, save teh full series of interest rates and FX returns for each currency
            interestData[currency] = interestRates
            fxData[currency] = fxReturns
        
        return months, interestData, fxData
    
    
    #month strings, currency data, start year and end year for filtering the period (differentiate between testing + training periods)
    def getPeriodData(self, months, data, startYear, endYear):
        """Extract data for specific period"""
        
        #grab all the months that we actually want in teh given time period, grab its data too
        periodMonths = []
        periodData = {}
        
        #add period data for each currenct
        for currency in self.currencies:
            periodData[currency] = []
        
        
        #pull out the year part of the month string itself, convery it into an integer (enumerate)
        for i, month in enumerate(months):
            year = int(month.split('-')[0])
            
            #MAKE SURE its acc in the given range
            if startYear <= year <= endYear:
                
                #add the month inot the previous list
                periodMonths.append(month)
                
                #loop again for EACH currency and add its value ot the index from the full dataset into this smaller, filtered list of period data
                for currency in self.currencies:
                    periodData[currency].append(data[currency][i])
        
        #Retyrn out the filtered data
        return periodMonths, periodData
    
    def calculateStrategyReturns(self, months, interestData, fxData, startYear, endYear, transactionCost=0.001):
        """Calculate strategy returns for given period"""
        periodMonths, interestInPeriod = self.getPeriodData(months, interestData, startYear, endYear)
        _, fxForPeriod = self.getPeriodData(months, fxData, startYear, endYear)
        
        stratReturns = []
        allPositions = []
        
        for i in range(len(periodMonths)):
            # Get current month's interest rates
            currentRates = {}
            for currency in self.currencies:
                currentRates[currency] = interestInPeriod[currency][i]
            
            # Sort currencies by interest rate (highest to lowest)
            currenciesSorted = sorted(currentRates.items(), key=lambda x: x[1], reverse=True)
            
            # Select top 3 (long) and bottom 3 (short)
            toLongCurrencies = [curr for curr, rate in currenciesSorted[:3]]
            toShortCurrencies = [curr for curr, rate in currenciesSorted[-3:]]
            
            # Create position weights
            positions = {}
            for currency in self.currencies:
                if currency in toLongCurrencies:
                    positions[currency] = 1/3  # Equal weight long
                elif currency in toShortCurrencies:
                    positions[currency] = -1/3  # Equal weight short
                else:
                    positions[currency] = 0.0
            
            allPositions.append(positions)
            
            # Calculate returns for this month (skip first month)
            # Skip first two months to allow signal delay and initial position
            if i > 0:
                # Use previous month's position with current month's returns
                prevPosition = allPositions[i-1]
                grossReturn = 0.0
                
                for currency in self.currencies:
                    grossReturn += prevPosition[currency] * fxForPeriod[currency][i]
                
                # Apply transaction costs when positions change
                totalTransactionCost = 0.0
                if i > 1:
                    for currency in self.currencies:
                        positionChange = abs(positions[currency] - allPositions[i-1][currency])
                        totalTransactionCost += positionChange * transactionCost
                else:
                    # Initial transaction cost
                    for currency in self.currencies:
                        totalTransactionCost += abs(positions[currency]) * transactionCost
                
                netReturn = grossReturn - totalTransactionCost
                stratReturns.append(netReturn)

        return periodMonths[1:], stratReturns  # Skip first month
    
    
    #calc monthly returns USING carry trade strat
    #months list, interest data is the histroical interest rates that we also added gaussian randomness, start year and end year degine the parameters
    #transaction cost is the assumed trading cost per unit of the position change (simulated to be 0.1%)
    def calculateBenchmarkReturns(self, months, fxData, startYear, endYear):
        """Calculate equal-weighted benchmark returns"""
        
        #filters interestrate and FX data to ust those in the given range of years
        #calls the preivous method to get that filtered data set
        periodMonths, fxForPeriod = self.getPeriodData(months, fxData, startYear, endYear)
        
        #store various returns
        benchReturns = []
        
        #set usd as teh base, doestn fluctuate (Reference Comparsion Point)
        notUSDCurrencies = [c for c in self.currencies if c != 'USD']
        
        #loop each month skipping the first one
        for i in range(1, len(periodMonths)):  # Skip first month
            
            #monthly return starts OFF at 0, accumulates return for each non-USD currency
            #Think of it as a sum-- a cumulative point
            monthlyReturn = 0.0
            
            #for each non USD, add return for the current month to the total then divide by the TOTAL numeber of non USD currencies for equal weighting
            #Basically averages out all of the FX returns for the given month
            for currency in notUSDCurrencies:
                monthlyReturn += fxForPeriod[currency][i] / len(notUSDCurrencies)
            benchReturns.append(monthlyReturn)
        
        #Start index 1 bc skippped first month for return calc
        return periodMonths[1:], benchReturns
    
    
    
    #Calc rates like the sharpe ratio, volatility, etc based on monthly returns
    
    #formulas from online
    #Total Return--> Overall % gain or loss from investment over the period--> (1+r1)(1+r2)...-1
        #how much money made or lost totall
    #Annualized Return--> avg return per ear, including compounding--> (1+ Mean monthly return)^12 -1
        #what the return would be smoothed out over each year
    #Volatility--> SD of the returns scaled annually, SD of monthly returns x sqrt(12)
        #how 'risky' the investment is (larger means riskier)
    #Sharpe Ratio--> Risk adjusted return metric--> (Annualized Return) / (Volatility)
        #How much return for each unit of risk (higher the better)
    #Max Drawdown--> Largest percentage drop from a peak to a trough
        #worst case drop in value anticipated/experienced in investment period
        
        
    def calculatePerformanceMetrics(self, returns):
        """Calculate performance statistics"""
        if len(returns) == 0:
            return {
                'Total Return': 0,
                'Annualized Return': 0,
                'Volatility': 0,
                'Sharpe Ratio': 0,
                'Max Drawdown': 0
            }
        
        # Calculate basic stats
        meanReturn = sum(returns) / len(returns)
        
        # Calculate standard deviation
        variance = sum((r - meanReturn) ** 2 for r in returns) / len(returns)
        StandardDeviation = math.sqrt(variance)
        
        # Calculate cumulative returns
        totReturn = 1.0
        for r in returns:
            
            #0.01 bc these are in percentages
            totReturn *= (1 + r/100)
        totReturn -= 1
        
        # Annualized metrics
        annualizedReturn = (1 + meanReturn) ** (12/ len(returns)) - 1
        volatility = StandardDeviation * math.sqrt(12)
        sharpe = annualizedReturn / volatility if volatility > 0 else 0
        
        # Calculate max drawdown
        cumulative = 1.0
        peak = 1.0
        drawdownMax = 0.0
        
        #update cumluative, update peak, calcs drawdown, tracks the MAX drawdown
        for r in returns:
            cumulative *= (1 + r)
            if cumulative > peak:
                peak = cumulative
            drawdown = (peak - cumulative) / peak
            if drawdown > drawdownMax:
                drawdownMax = drawdown
        
        #Return a DICTIONARY for each of the values
        return {
            'Total Return': totReturn,
            'Annualized Return': annualizedReturn,
            'Volatility': volatility,
            'Sharpe Ratio': sharpe,
            'Max Drawdown': -drawdownMax  # Make negative for display
        }
    
    def printTable(self, headers, rows):
        """Print a formatted table"""
        # Calculate column widths
        colWidth = [len(str(header)) for header in headers]
        for row in rows:
            for i, cell in enumerate(row):
                colWidth[i] = max(colWidth[i], len(str(cell)))
        
        # Print header
        headerLine = " | ".join(f"{str(header):<{colWidth[i]}}" for i, header in enumerate(headers))
        print(headerLine)
        print("-" * len(headerLine))
        
        # Print rows
        for row in rows:
            rowLine = " | ".join(f"{str(cell):<{colWidth[i]}}" for i, cell in enumerate(row))
            print(rowLine)
    
    def formatPercentages(self, value):
        """Format value as percentage"""
        return f"{value:.2%}"
    
    def formatNumber(self, value):
        """Format number with 3 decimal places"""
        return f"{value:.3f}"
    
    def runBacktest(self):
        """Run the complete backtest"""
        print("Generating mock data...")
        months, interestData, fxData = self.generateMockData()
        
        print("Running training period (2010-2015)...")
        train_months, training_returns = self.calculateStrategyReturns(
            months, interestData, fxData, self.training_startYear, self.training_endYear
        )
        
        print("Running testing period (2015-2020)...")
        test_months, testing_returns = self.calculateStrategyReturns(
            months, interestData, fxData, self.testing_startYear, self.testing_endYear
        )
        
        # Calculate benchmark returns
        _, training_benchmark = self.calculateBenchmarkReturns(
            months, fxData, self.training_startYear, self.training_endYear
        )
        _, testing_benchmark = self.calculateBenchmarkReturns(
            months, fxData, self.testing_startYear, self.testing_endYear
        )
        
        # Calculate performance metrics
        training_metrics = self.calculatePerformanceMetrics(training_returns)
        testing_metrics = self.calculatePerformanceMetrics(testing_returns)
        training_bench_metrics = self.calculatePerformanceMetrics(training_benchmark)
        testing_bench_metrics = self.calculatePerformanceMetrics(testing_benchmark)
        
        # Print results
        self.printResults(training_metrics, testing_metrics,
                          training_bench_metrics, testing_bench_metrics)
        
        # Display sample data
        self.print_sample_data(months, interestData, fxData)
        
        return {
            'training_returns': training_returns,
            'testing_returns': testing_returns,
            'training_benchmark': training_benchmark,
            'testing_benchmark': testing_benchmark,
            'months': months,
            'interestData': interestData,
            'fxData': fxData
        }
    
    def printResults(self, train_metrics, test_metrics, train_bench, test_bench):
        """Print formatted results"""
        print("\n" + "="*80)
        print("CARRY TRADE BACKTEST RESULTS")
        print("="*80)
        
        print("\nTRAINING PERIOD (2010-2015)")
        print("-" * 50)
        
        # Training period table
        headers = ["Metric", "Strategy", "Benchmark"]
        rows = []
        for metric in train_metrics.keys():
            if 'Return' in metric or 'Drawdown' in metric:
                strategy_val = self.formatPercentages(train_metrics[metric])
                benchmark_val = self.formatPercentages(train_bench[metric])
            else:
                strategy_val = self.formatNumber(train_metrics[metric])
                benchmark_val = self.formatNumber(train_bench[metric])
            rows.append([metric, strategy_val, benchmark_val])
        
        self.printTable(headers, rows)
        
        print("\nTESTING PERIOD (2015-2020)")
        print("-" * 50)
        
        # Testing period table
        rows = []
        for metric in test_metrics.keys():
            if 'Return' in metric or 'Drawdown' in metric:
                strategy_val = self.formatPercentages(test_metrics[metric])
                benchmark_val = self.formatPercentages(test_bench[metric])
            else:
                strategy_val = self.formatNumber(test_metrics[metric])
                benchmark_val = self.formatNumber(test_bench[metric])
            rows.append([metric, strategy_val, benchmark_val])
        
        self.printTable(headers, rows)
        
        print("\nSTRATEGY SUMMARY")
        print("-" * 50)
        print("The carry trade strategy:")
        print("• Goes LONG the 3 highest interest rate currencies")
        print("• Goes SHORT the 3 lowest interest rate currencies")
        print("• Rebalances monthly with equal weights")
        print("• Includes 0.1% transaction costs")
        
        # Performance comparison
        train_outperform = train_metrics['Sharpe Ratio'] > train_bench['Sharpe Ratio']
        test_outperform = test_metrics['Sharpe Ratio'] > test_bench['Sharpe Ratio']
        
        print(f"\nStrategy {'OUTPERFORMED' if train_outperform else 'UNDERPERFORMED'} benchmark in training")
        print(f"Strategy {'OUTPERFORMED' if test_outperform else 'UNDERPERFORMED'} benchmark in testing")
        
        if train_outperform and not test_outperform:
            #oerfitting means it matches TOO well shows no diff, failed test basicallt
            print("\n⚠️  WARNING: Potential overfitting detected")
        elif train_outperform and test_outperform:
            print("\nStrategy shows consistent outperformance")
        
        print("\n" + "="*80)
    
    def print_sample_data(self, months, interestData, fxData):
        """Print sample of the generated data"""
        print("\nSAMPLE INTEREST RATE DATA (First 5 months):")
        print("-" * 60)
        
        # Interest rate table
        headers = ["Month"] + self.currencies
        rows = []
        for i in range(5):
            row = [months[i]]
            for currency in self.currencies:
                row.append(f"{interestData[currency][i]:.2f}%")
            rows.append(row)
        
        self.printTable(headers, rows)
        
        print("\nSAMPLE FX RETURN DATA (First 5 months):")
        print("-" * 60)
        
        # FX return table
        rows = []
        for i in range(5):
            row = [months[i]]
            for currency in self.currencies:
                row.append(f"{fxData[currency][i]:.3f}")
            rows.append(row)
        
        self.printTable(headers, rows)
        
    def clearScreen(self):
        """Clear the console screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

    def createProgressBar(self, current, total, bar_length=50):
        """Create a simple progress bar"""
        progress = current / total
        block = int(round(bar_length * progress))
        bar = "█" * block + "░" * (bar_length - block)
        percentage = round(progress * 100, 1)
        return f"[{bar}] {percentage}%"

    def animateBacktestProgress(self):
        """Animate the backtest progress"""
        print("\n" + "="*80)
        print("CARRY TRADE BACKTEST - ANIMATED RESULTS")
        print("="*80)
        
        # Simulate backtest steps with animation
        steps = [
            ("Initializing currencies and parameters...", 0.5),
            ("Generating mock interest rate data...", 1.0),
            ("Calculating FX returns...", 0.8),
            ("Running training period (2010-2015)...", 1.2),
            ("Calculating strategy positions...", 0.7),
            ("Running testing period (2015-2020)...", 1.2),
            ("Computing performance metrics...", 0.6),
            ("Analyzing results...", 0.4)
        ]
        
        for i, (step, duration) in enumerate(steps):
            print(f"\n{step}")
            
            # Animate progress bar for this step
            sub_steps = 20
            for j in range(sub_steps + 1):
                progress_bar = self.createProgressBar(j, sub_steps, 30)
                print(f"\r{progress_bar}", end="", flush=True)
                time.sleep(duration / sub_steps)
            
            print(" ✓")
        
        print("\n" + "="*50)
        print("BACKTEST COMPLETE!")
        print("="*50)
        time.sleep(1)

    def animateResults(self, training_metrics, testing_metrics, training_bench, testing_bench):
        """Animate the display of results"""
        self.clearScreen()
        
        # Header animation
        title = "CARRY TRADE STRATEGY RESULTS"
        for i in range(len(title) + 1):
            print(f"\r{title[:i]}", end="", flush=True)
            time.sleep(0.05)
        
        print("\n" + "="*60)
        time.sleep(0.5)
        
        # Training Period Results
        print("\n🎯 TRAINING PERIOD (2010-2015)")
        print("-" * 40)
        time.sleep(0.3)
        
        training_data = [
            ("Total Return", "Strategy", self.formatPercentages(training_metrics['Total Return'])),
            ("", "Benchmark", self.formatPercentages(training_bench['Total Return'])),
            ("Sharpe Ratio", "Strategy", self.formatNumber(training_metrics['Sharpe Ratio'])),
            ("", "Benchmark", self.formatNumber(training_bench['Sharpe Ratio'])),
            ("Max Drawdown", "Strategy", self.formatPercentages(training_metrics['Max Drawdown'])),
            ("", "Benchmark", self.formatPercentages(training_bench['Max Drawdown']))
        ]
        
        for metric, source, value in training_data:
            if metric:
                print(f"\n{metric}:")
            color = "🟢" if "Strategy" in source else "🔵"
            print(f"  {color} {source}: {value}")
            time.sleep(0.2)
        
        time.sleep(0.5)
        
        # Testing Period Results
        print("\n🎯 TESTING PERIOD (2015-2020)")
        print("-" * 40)
        time.sleep(0.3)
        
        testing_data = [
            ("Total Return", "Strategy", self.formatPercentages(testing_metrics['Total Return'])),
            ("", "Benchmark", self.formatPercentages(testing_bench['Total Return'])),
            ("Sharpe Ratio", "Strategy", self.formatNumber(testing_metrics['Sharpe Ratio'])),
            ("", "Benchmark", self.formatNumber(testing_bench['Sharpe Ratio'])),
            ("Max Drawdown", "Strategy", self.formatPercentages(testing_metrics['Max Drawdown'])),
            ("", "Benchmark", self.formatPercentages(testing_bench['Max Drawdown']))
        ]
        
        for metric, source, value in testing_data:
            if metric:
                print(f"\n{metric}:")
            color = "🟢" if "Strategy" in source else "🔵"
            print(f"  {color} {source}: {value}")
            time.sleep(0.2)
        
        time.sleep(0.5)

    def displayCurrencyWheel(self, interest_data, month_index=60):
        """Display a rotating currency wheel showing interest rates"""
        print("\n" + "="*50)
        print("CURRENCY INTEREST RATES WHEEL")
        print("="*50)
        
        # Get interest rates for a specific month
        rates = []
        for currency in self.currencies:
            rate = interest_data[currency][month_index] if month_index < len(interest_data[currency]) else interest_data[currency][-1]
            rates.append((currency, rate))
        
        # Sort by interest rate
        rates.sort(key=lambda x: x[1], reverse=True)
        
        # Display as a rotating wheel
        for rotation in range(3):  # 3 rotations
            self.clearScreen()
            print("\n" + "="*50)
            print("CURRENCY INTEREST RATES WHEEL")
            print("="*50)
            print()
            
            for i, (currency, rate) in enumerate(rates):
                # Create rotating effect by shifting display
                display_index = (i + rotation) % len(rates)
                display_currency, display_rate = rates[display_index]
                
                # Visual representation
                bar_length = int(display_rate * 10)  # Scale for visualization
                bar = "█" * min(bar_length, 50)
                
                # Color coding
                if display_rate > 3.0:
                    emoji = "🟢"  # High rate
                elif display_rate > 1.5:
                    emoji = "🟡"  # Medium rate
                else:
                    emoji = "🔴"  # Low rate
                
                print(f"{emoji} {display_currency}: {display_rate:5.2f}% {bar}")
                time.sleep(0.1)
            
            print(f"\n{'.' * (rotation * 10)}🔄")
            time.sleep(0.5)

    def displayFinalSummary(self, train_metrics, test_metrics, train_bench, test_bench):
        """Display final animated summary"""
        self.clearScreen()
        
        print("FINAL STRATEGY ASSESSMENT")
        
        # Strategy effectiveness
        train_outperform = train_metrics['Sharpe Ratio'] > train_bench['Sharpe Ratio']
        test_outperform = test_metrics['Sharpe Ratio'] > test_bench['Sharpe Ratio']
        
        time.sleep(0.5)
        
        # Training results
        result_emoji = "✅" if train_outperform else "❌"
        print(f"\nTraining Period: {result_emoji}")
        print(f"Strategy Sharpe: {train_metrics['Sharpe Ratio']:.3f} vs Benchmark: {train_bench['Sharpe Ratio']:.3f}")
        time.sleep(0.8)
        
        # Testing results
        result_emoji = "✅" if test_outperform else "❌"
        print(f"\nTesting Period: {result_emoji}")
        print(f"Strategy Sharpe: {test_metrics['Sharpe Ratio']:.3f} vs Benchmark: {test_bench['Sharpe Ratio']:.3f}")
        time.sleep(0.8)
        
        # Overall assessment
        print(f"\n{'='*50}")
        if train_outperform and test_outperform:
            print("TRATEGY SHOWS CONSISTENT OUTPERFORMANCE")
            verdict_color = "🟢"
        elif train_outperform and not test_outperform:
            print("POTENTIAL OVERFITTING")
            verdict_color = "🟡"
        else:
            print("STRATEGY UNDERPERFORMS")
            verdict_color = "🔴"
        
        time.sleep(1)
        
        # Final animation
        for i in range(3):
            print(f"\r{verdict_color} Analysis Complete {'.' * (i + 1)}", end="", flush=True)
            time.sleep(0.5)
        
        print("\n" + "="*50)

    def runAnimatedBacktest(self):
        """Main method to run the animated backtest"""
        # Step 1: Show progress animation
        self.animateBacktestProgress()
        
        # Step 2: Generate actual data
        print("\nGenerating actual data...")
        months, interestData, fxData = self.generateMockData()
        
        # Step 3: Run calculations
        train_months, training_returns = self.calculateStrategyReturns(
            months, interestData, fxData, self.training_startYear, self.training_endYear
        )
        test_months, testing_returns = self.calculateStrategyReturns(
            months, interestData, fxData, self.testing_startYear, self.testing_endYear
        )
        
        # Calculate benchmarks
        _, training_benchmark = self.calculateBenchmarkReturns(
            months, fxData, self.training_startYear, self.training_endYear
        )
        _, testing_benchmark = self.calculateBenchmarkReturns(
            months, fxData, self.testing_startYear, self.testing_endYear
        )
        
        # Calculate metrics
        training_metrics = self.calculatePerformanceMetrics(training_returns)
        testing_metrics = self.calculatePerformanceMetrics(testing_returns)
        training_bench_metrics = self.calculatePerformanceMetrics(training_benchmark)
        testing_bench_metrics = self.calculatePerformanceMetrics(testing_benchmark)
        
        # Step 4: Animate results display
        time.sleep(1)
        self.animateResults(training_metrics, testing_metrics, 
                        training_bench_metrics, testing_bench_metrics)
        
        # Step 5: Show currency wheel
        time.sleep(2)
        self.displayCurrencyWheel(interestData)
        
        # Step 6: Final summary
        time.sleep(1)
        self.displayFinalSummary(training_metrics, testing_metrics,
                                training_bench_metrics, testing_bench_metrics)
        
        # Return results
        return {
            'training_returns': training_returns,
            'testing_returns': testing_returns,
            'training_benchmark': training_benchmark,
            'testing_benchmark': testing_benchmark,
            'months': months,
            'interestData': interestData,
            'fxData': fxData
        }

# Run the backtest

#prevents code from executing automatically in case we import it into a new file to use real data
if __name__ == "__main__":
    backtest = CarryTradeBacktest()
    
    print("Choose execution type:")
    print("1. Standard text output")
    print("2. Animated presentation")
    
    try:
        choice = input("Enter choice (1-2): ").strip()
    except:
        choice = "1"
    
    if choice == "2":
        print("\n Starting animated presentation...")
        time.sleep(1)
        results = backtest.runAnimatedBacktest()
        
        print(f"\n Animation completed")
        print(f"Analyzed {len(results['months'])} months of data")
        print(f"Training: {len(results['training_returns'])} months")
        print(f"Testing: {len(results['testing_returns'])} months")
    else:
        results = backtest.runBacktest()
        print(f"\nBacktest completed successfully!")
        print(f"Analyzed {len(results['months'])} months of data")
        print(f"Training period: {len(results['training_returns'])} months")
        print(f"Testing period: {len(results['testing_returns'])} months")
    
    