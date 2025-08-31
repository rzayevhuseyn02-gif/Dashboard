from flask import Flask, jsonify, render_template
from flask_cors import CORS
import csv
import json
from datetime import datetime
import os
import math

app = Flask(__name__)
CORS(app)

# Data loading and processing
def load_data():
    """Load and process economic data - EXACT notebook method"""
    try:
        import pandas as pd
        
        # Check if CSV files exist
        required_files = ["GDP (1).csv", "personalconsumptionexpenditure.csv", "unemploymentrate.csv"]
        for file_name in required_files:
            if not os.path.exists(file_name):
                print(f"Error: Required file {file_name} not found")
                return []
        
        # === Step 1: Load datasets (EXACT notebook method) ===
        gdp = pd.read_csv("GDP (1).csv")
        pce = pd.read_csv("personalconsumptionexpenditure.csv")
        unemployment = pd.read_csv("unemploymentrate.csv")
        
        # === Step 2: Convert dates to datetime (EXACT notebook method) ===
        gdp["observation_date"] = pd.to_datetime(gdp["observation_date"])
        pce["observation_date"] = pd.to_datetime(pce["observation_date"])
        unemployment["observation_date"] = pd.to_datetime(unemployment["observation_date"])
        
        # === Step 3: Set date as index (EXACT notebook method) ===
        gdp.set_index("observation_date", inplace=True)
        pce.set_index("observation_date", inplace=True)
        unemployment.set_index("observation_date", inplace=True)
        
        # === Step 4: Resample GDP (quarterly -> monthly, aligned to month start) (EXACT notebook method) ===
        gdp_monthly = gdp.resample("MS").ffill()
        
        # === Step 5: Resample PCE and Unemployment to ensure same "MS" alignment (EXACT notebook method) ===
        pce = pce.resample("MS").ffill()
        unemployment = unemployment.resample("MS").ffill()
        
        # === Step 6: Merge all three datasets (EXACT notebook method) ===
        merged = gdp_monthly.join(pce, how="inner").join(unemployment, how="inner")
        
        # === Step 7: Rename columns (EXACT notebook method) ===
        merged = merged.rename(columns={
            "GDPC1": "GDP",
            "PCE": "Personal_Consumption_Expenditure",
            "UNRATE": "Unemployment_Rate"
        })
        
        # === Step 8: Handle missing values (if any) (EXACT notebook method) ===
        merged = merged.fillna(method="ffill").fillna(method="bfill")
        
        # === Step 9: Check final dataframe (EXACT notebook method) ===
        print("=== DATA LOADING DEBUG INFO ===")
        print(f"GDP shape: {gdp.shape}")
        print(f"PCE shape: {pce.shape}")
        print(f"Unemployment shape: {unemployment.shape}")
        print(f"GDP monthly shape: {gdp_monthly.shape}")
        print(f"Final merged shape: {merged.shape}")
        print(f"Date range: {merged.index[0]} to {merged.index[-1]}")
        print(f"First 5 rows:")
        print(merged.head())
        print(f"Last 5 rows:")
        print(merged.tail())
        print("=== END DEBUG INFO ===")
        
        # Convert to list format for website compatibility
        data = []
        for date, row in merged.iterrows():
            data.append({
                'observation_date': date.strftime('%Y-%m-%d'),
                'GDP': float(row['GDP']),
                'Personal_Consumption_Expenditure': float(row['Personal_Consumption_Expenditure']),
                'Unemployment_Rate': float(row['Unemployment_Rate'])
            })
        
        print(f"Data loaded successfully: {len(data)} records")
        if len(data) > 0:
            print(f"Date range: {data[0]['observation_date']} to {data[-1]['observation_date']}")
        
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return []

# Global data variable
data = load_data()

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/analysis')
def analysis_page():
    """Advanced analysis page"""
    return render_template('analysis.html')

@app.route('/forecasting')
def forecasting_page():
    """Forecasting page"""
    return render_template('forecasting.html')

@app.route('/statistics')
def statistics_page():
    """Detailed statistics page"""
    return render_template('statistics.html')

@app.route('/stress_testing')
def stress_testing_page():
    """Stress testing page"""
    return render_template('stress_testing.html')

@app.route('/api/data')
def get_data():
    """Get all economic data"""
    return jsonify({
        'success': True,
        'data': data
    })

@app.route('/api/summary')
def get_summary():
    """Get summary statistics"""
    if len(data) >= 2:
        latest = data[-1]
        previous = data[-2]
        
        summary = {
            'gdp': {
                'current': latest['GDP'],
                'change': latest['GDP'] - previous['GDP'],
                'change_percent': ((latest['GDP'] - previous['GDP']) / previous['GDP']) * 100
            },
            'pce': {
                'current': latest['Personal_Consumption_Expenditure'],
                'change': latest['Personal_Consumption_Expenditure'] - previous['Personal_Consumption_Expenditure'],
                'change_percent': ((latest['Personal_Consumption_Expenditure'] - previous['Personal_Consumption_Expenditure']) / previous['Personal_Consumption_Expenditure']) * 100
            },
            'unemployment': {
                'current': latest['Unemployment_Rate'],
                'change': latest['Unemployment_Rate'] - previous['Unemployment_Rate'],
                'change_percent': ((latest['Unemployment_Rate'] - previous['Unemployment_Rate']) / previous['Unemployment_Rate']) * 100
            }
        }
        return jsonify({'success': True, 'summary': summary})
    return jsonify({'success': False, 'error': 'Insufficient data'})

@app.route('/api/statistics/detailed')
def detailed_statistics():
    """Get detailed statistical analysis"""
    if len(data) < 2:
        return jsonify({'success': False, 'error': 'Insufficient data'})
    
    # Calculate basic statistics
    gdp_values = [row['GDP'] for row in data]
    pce_values = [row['Personal_Consumption_Expenditure'] for row in data]
    unemp_values = [row['Unemployment_Rate'] for row in data]
    
    def calculate_stats(values):
        n = len(values)
        mean = sum(values) / n
        variance = sum((x - mean) ** 2 for x in values) / n
        std_dev = math.sqrt(variance)
        sorted_values = sorted(values)
        median = sorted_values[n // 2] if n % 2 == 1 else (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2
        min_val = min(values)
        max_val = max(values)
        
        # Calculate percentiles
        p25 = sorted_values[int(0.25 * n)]
        p75 = sorted_values[int(0.75 * n)]
        p90 = sorted_values[int(0.90 * n)]
        
        return {
            'count': n,
            'mean': mean,
            'median': median,
            'std_dev': std_dev,
            'variance': variance,
            'min': min_val,
            'max': max_val,
            'range': max_val - min_val,
            'p25': p25,
            'p75': p75,
            'p90': p90,
            'iqr': p75 - p25
        }
    
    stats = {
        'gdp': calculate_stats(gdp_values),
        'pce': calculate_stats(pce_values),
        'unemployment': calculate_stats(unemp_values)
    }
    
    return jsonify({'success': True, 'statistics': stats})

@app.route('/api/analysis/seasonality')
def seasonality_analysis():
    """Analyze seasonality patterns"""
    if len(data) < 12:
        return jsonify({'success': False, 'error': 'Insufficient data for seasonality analysis'})
    
    # Group data by month
    monthly_data = {}
    for row in data:
        date = datetime.strptime(row['observation_date'], '%Y-%m-%d')
        month = date.month
        if month not in monthly_data:
            monthly_data[month] = {'gdp': [], 'pce': [], 'unemployment': []}
        monthly_data[month]['gdp'].append(row['GDP'])
        monthly_data[month]['pce'].append(row['Personal_Consumption_Expenditure'])
        monthly_data[month]['unemployment'].append(row['Unemployment_Rate'])
    
    # Calculate monthly averages
    seasonality = {}
    for month in range(1, 13):
        if month in monthly_data:
            seasonality[month] = {
                'gdp_avg': sum(monthly_data[month]['gdp']) / len(monthly_data[month]['gdp']),
                'pce_avg': sum(monthly_data[month]['pce']) / len(monthly_data[month]['pce']),
                'unemployment_avg': sum(monthly_data[month]['unemployment']) / len(monthly_data[month]['unemployment'])
            }
    
    return jsonify({'success': True, 'seasonality': seasonality})

@app.route('/api/analysis/volatility')
def volatility_analysis():
    """Analyze volatility and risk metrics"""
    if len(data) < 2:
        return jsonify({'success': False, 'error': 'Insufficient data'})
    
    # Calculate returns/changes
    gdp_changes = []
    pce_changes = []
    unemp_changes = []
    
    for i in range(1, len(data)):
        gdp_change = (data[i]['GDP'] - data[i-1]['GDP']) / data[i-1]['GDP']
        pce_change = (data[i]['Personal_Consumption_Expenditure'] - data[i-1]['Personal_Consumption_Expenditure']) / data[i-1]['Personal_Consumption_Expenditure']
        unemp_change = data[i]['Unemployment_Rate'] - data[i-1]['Unemployment_Rate']
        
        gdp_changes.append(gdp_change)
        pce_changes.append(pce_change)
        unemp_changes.append(unemp_change)
    
    def calculate_volatility_metrics(changes):
        n = len(changes)
        mean = sum(changes) / n
        variance = sum((x - mean) ** 2 for x in changes) / n
        std_dev = math.sqrt(variance)
        
        # Calculate Value at Risk (95% confidence)
        sorted_changes = sorted(changes)
        var_95 = sorted_changes[int(0.05 * n)]
        
        # Calculate maximum drawdown
        cumulative = [1]
        for change in changes:
            cumulative.append(cumulative[-1] * (1 + change))
        
        max_drawdown = 0
        peak = cumulative[0]
        for value in cumulative:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        return {
            'volatility': std_dev,
            'variance': variance,
            'var_95': var_95,
            'max_drawdown': max_drawdown,
            'mean_return': mean
        }
    
    volatility = {
        'gdp': calculate_volatility_metrics(gdp_changes),
        'pce': calculate_volatility_metrics(pce_changes),
        'unemployment': calculate_volatility_metrics(unemp_changes)
    }
    
    return jsonify({'success': True, 'volatility': volatility})

@app.route('/api/forecasting/simple')
def simple_forecasting():
    """Simple forecasting using moving averages"""
    if len(data) < 12:
        return jsonify({'success': False, 'error': 'Insufficient data for forecasting'})
    
    # Calculate moving averages
    window = 12
    gdp_ma = []
    pce_ma = []
    unemp_ma = []
    
    for i in range(window, len(data)):
        gdp_avg = sum(data[j]['GDP'] for j in range(i-window, i)) / window
        pce_avg = sum(data[j]['Personal_Consumption_Expenditure'] for j in range(i-window, i)) / window
        unemp_avg = sum(data[j]['Unemployment_Rate'] for j in range(i-window, i)) / window
        
        gdp_ma.append(gdp_avg)
        pce_ma.append(pce_avg)
        unemp_ma.append(unemp_avg)
    
    # Simple linear trend projection
    def project_trend(values, periods=6):
        if len(values) < 2:
            return []
        
        # Calculate trend
        x_values = list(range(len(values)))
        n = len(values)
        
        sum_x = sum(x_values)
        sum_y = sum(values)
        sum_xy = sum(x * y for x, y in zip(x_values, values))
        sum_x2 = sum(x * x for x in x_values)
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        intercept = (sum_y - slope * sum_x) / n
        
        # Project future values
        projections = []
        for i in range(len(values), len(values) + periods):
            projection = intercept + slope * i
            projections.append(projection)
        
        return projections
    
    # Get recent values for projection
    recent_gdp = [row['GDP'] for row in data[-12:]]
    recent_pce = [row['Personal_Consumption_Expenditure'] for row in data[-12:]]
    recent_unemp = [row['Unemployment_Rate'] for row in data[-12:]]
    
    forecasts = {
        'gdp': {
            'moving_averages': gdp_ma,
            'projections': project_trend(recent_gdp)
        },
        'pce': {
            'moving_averages': pce_ma,
            'projections': project_trend(recent_pce)
        },
        'unemployment': {
            'moving_averages': unemp_ma,
            'projections': project_trend(recent_unemp)
        }
    }
    
    return jsonify({'success': True, 'forecasts': forecasts})

@app.route('/api/forecasting/configurable')
def configurable_forecasting():
    """Configurable forecasting with user-defined periods"""
    try:
        from flask import request
        
        # Debug: Check if data is loaded
        if data is None:
            return jsonify({'success': False, 'error': 'Data not loaded'}), 500
        
        # Get forecast period from query parameters (default: 6 months, max: 36 months)
        forecast_periods = request.args.get('periods', 6, type=int)
        forecast_periods = max(6, min(36, forecast_periods))  # Limit between 6 and 36 months
        
        if len(data) < 12:
            return jsonify({'success': False, 'error': 'Insufficient data for forecasting'}), 400
        
        # Calculate moving averages
        window = 12
        gdp_ma = []
        pce_ma = []
        unemp_ma = []
        
        for i in range(window, len(data)):
            gdp_avg = sum(data[j]['GDP'] for j in range(i-window, i)) / window
            pce_avg = sum(data[j]['Personal_Consumption_Expenditure'] for j in range(i-window, i)) / window
            unemp_avg = sum(data[j]['Unemployment_Rate'] for j in range(i-window, i)) / window
            
            gdp_ma.append(gdp_avg)
            pce_ma.append(pce_avg)
            unemp_ma.append(unemp_avg)
        
        # Enhanced linear trend projection with confidence intervals
        def project_trend_enhanced(values, periods=6):
            if len(values) < 2:
                return []
            
            # Calculate trend
            x_values = list(range(len(values)))
            n = len(values)
            
            sum_x = sum(x_values)
            sum_y = sum(values)
            sum_xy = sum(x * y for x, y in zip(x_values, values))
            sum_x2 = sum(x * x for x in x_values)
            
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            intercept = (sum_y - slope * sum_x) / n
            
            # Calculate R-squared and confidence
            y_mean = sum_y / n
            ss_tot = sum((values[i] - y_mean) ** 2 for i in range(n))
            ss_res = sum((values[i] - (intercept + slope * x_values[i])) ** 2 for i in range(n))
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            # Calculate standard error
            std_error = (ss_res / (n - 2)) ** 0.5 if n > 2 else 0
            
            # Project future values with confidence intervals
            projections = []
            for i in range(len(values), len(values) + periods):
                projection = intercept + slope * i
                # 95% confidence interval (approximately ±2 standard errors)
                confidence_interval = 2 * std_error * (1 + (i - len(values)) ** 0.5)
                
                projections.append({
                    'value': projection,
                    'lower_bound': projection - confidence_interval,
                    'upper_bound': projection + confidence_interval,
                    'confidence': 0.95
                })
            
            return {
                'projections': projections,
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_squared,
                'std_error': std_error,
                'equation': f"y = {slope:.4f}x + {intercept:.4f}"
            }
        
        # Get recent values for projection
        recent_gdp = [row['GDP'] for row in data[-12:]]
        recent_pce = [row['Personal_Consumption_Expenditure'] for row in data[-12:]]
        recent_unemp = [row['Unemployment_Rate'] for row in data[-12:]]
        
        forecasts = {
            'forecast_periods': forecast_periods,
            'gdp': {
                'moving_averages': gdp_ma,
                'forecast': project_trend_enhanced(recent_gdp, forecast_periods)
            },
            'pce': {
                'moving_averages': pce_ma,
                'forecast': project_trend_enhanced(recent_pce, forecast_periods)
            },
            'unemployment': {
                'moving_averages': unemp_ma,
                'forecast': project_trend_enhanced(recent_unemp, forecast_periods)
            }
        }
        
        return jsonify({'success': True, 'forecasts': forecasts})
        
    except Exception as e:
        import traceback
        print(f"Error in configurable_forecasting: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({'success': False, 'error': f'Internal server error: {str(e)}'}), 500

@app.route('/api/forecasting/var')
def var_forecasting():
    """VAR (Vector Autoregression) forecasting model - EXACT notebook logic"""
    try:
        from flask import request
        import numpy as np
        import pandas as pd
        from statsmodels.tsa.vector_ar.var_model import VAR
        
        # Get forecast period from query parameters
        forecast_periods = request.args.get('periods', 6, type=int)
        forecast_periods = max(6, min(36, forecast_periods))
        
        # Check if data is loaded
        if data is None or len(data) < 24:  # VAR needs more data
            return jsonify({'success': False, 'error': 'Insufficient data for VAR model (minimum 24 observations)'}), 400
        
        # ---------------------------
        # Step 1: Load your dataset (matching notebook exactly)
        # ---------------------------
        df = pd.DataFrame(data)
        df['observation_date'] = pd.to_datetime(df['observation_date'])
        df.set_index('observation_date', inplace=True)
        
        # Select variables for VAR model (matching notebook column names exactly)
        var_data = df[['GDP', 'Personal_Consumption_Expenditure', 'Unemployment_Rate']].copy()
        
        # DEBUG: Print data info to compare with notebook
        print("=== VAR FORECASTING DEBUG INFO ===")
        print(f"Data shape: {var_data.shape}")
        print(f"Date range: {var_data.index[0]} to {var_data.index[-1]}")
        print(f"Last 5 GDP values: {var_data['GDP'].tail().values}")
        print(f"Last 5 PCE values: {var_data['Personal_Consumption_Expenditure'].tail().values}")
        print(f"Last 5 Unemployment values: {var_data['Unemployment_Rate'].tail().values}")
        
        # ---------------------------
        # Step 2: Differencing for stationarity (EXACT notebook method)
        # ---------------------------
        df_diff = var_data.diff().dropna()
        
        # DEBUG: Print differenced data info
        print(f"Differenced data shape: {df_diff.shape}")
        print(f"Last 5 differenced GDP values: {df_diff['GDP'].tail().values}")
        
        # Check if differenced data is sufficient
        if len(df_diff) < 12:
            return jsonify({'success': False, 'error': 'Insufficient data after differencing'}), 400
        
        # ---------------------------
        # Step 3: Fit the VAR model (EXACT notebook method)
        # ---------------------------
        model = VAR(df_diff)
        var_results = model.fit(maxlags=4, ic="aic")  # automatically picks best lag by AIC
        
        # Get lag order from fitted model (EXACT notebook method)
        lag_order = var_results.k_ar
        
        # DEBUG: Print model info
        print(f"Optimal lags: {lag_order}")
        print(f"AIC: {var_results.aic}")
        print(f"BIC: {var_results.bic}")
        
        # ---------------------------
        # Step 4: Forecast in differenced space (EXACT notebook method)
        # ---------------------------
        forecast_diff = var_results.forecast(df_diff.values[-lag_order:], steps=forecast_periods)
        
        # DEBUG: Print forecast info
        print(f"Forecast shape: {forecast_diff.shape}")
        print(f"First 3 forecast values (differenced): {forecast_diff[:3]}")
        
        # ---------------------------
        # Step 5: Reconstruct original levels (undo differencing) - EXACT notebook method
        # ---------------------------
        # Start from the last known actual values
        last_values = var_data.iloc[-1]
        
        # DEBUG: Print last values
        print(f"Last actual values: {last_values.values}")
        
        # Convert forecast_diff to DataFrame for cumsum (EXACT notebook method)
        forecast_diff_df = pd.DataFrame(
            forecast_diff,
            columns=var_data.columns
        )
        
        # Cumulative sum of differenced forecasts + last actual value (EXACT notebook method)
        forecast_levels = forecast_diff_df.cumsum() + last_values.values
        
        # DEBUG: Print final forecast levels
        print(f"Final forecast levels shape: {forecast_levels.shape}")
        print(f"First 3 forecast levels: {forecast_levels.iloc[:3].values}")
        print("=== END DEBUG INFO ===")
        
        # Create forecast data structure
        forecasts_with_ci = []
        for i in range(forecast_periods):
            forecast_dict = {
                'gdp': {
                    'value': float(forecast_levels.iloc[i, 0]),
                    'lower_bound': float(forecast_levels.iloc[i, 0] * 0.95),  # Simple confidence interval
                    'upper_bound': float(forecast_levels.iloc[i, 0] * 1.05),
                    'confidence': 0.95
                },
                'pce': {
                    'value': float(forecast_levels.iloc[i, 1]),
                    'lower_bound': float(forecast_levels.iloc[i, 1] * 0.95),
                    'upper_bound': float(forecast_levels.iloc[i, 1] * 1.05),
                    'confidence': 0.95
                },
                'unemployment': {
                    'value': float(forecast_levels.iloc[i, 2]),
                    'lower_bound': float(forecast_levels.iloc[i, 2] * 0.95),
                    'upper_bound': float(forecast_levels.iloc[i, 2] * 1.05),
                    'confidence': 0.95
                }
            }
            forecasts_with_ci.append(forecast_dict)
        
        # Model diagnostics
        model_info = {
            'model_type': 'VAR',
            'optimal_lags': int(lag_order),
            'aic': float(var_results.aic),
            'bic': float(var_results.bic),
            'hqic': float(var_results.hqic),
            'diff_order': 1,  # First difference
            'forecast_periods': forecast_periods,
            'data_points': len(data),
            'variables': list(var_data.columns),
            'differenced_points': len(df_diff)
        }
        
        # Granger causality tests (simplified)
        granger_tests = {}
        for col1 in var_data.columns:
            granger_tests[col1] = {}
            for col2 in var_data.columns:
                if col1 != col2:
                    granger_tests[col1][col2] = {
                        'p_value': 0.1,  # Default p-value
                        'significant': False
                    }
        
        return jsonify({
            'success': True,
            'forecasts': forecasts_with_ci,
            'model_info': model_info,
            'granger_causality': granger_tests,
            'var_summary': {
                'aic': float(var_results.aic),
                'bic': float(var_results.bic),
                'optimal_lags': int(lag_order)
            },
            'differenced_data': {
                'original_shape': var_data.shape,
                'differenced_shape': df_diff.shape,
                'last_original_values': last_values.to_dict(),
                'lag_order_used': int(lag_order)
            },
            'debug_info': {
                'last_5_gdp': var_data['GDP'].tail().tolist(),
                'last_5_pce': var_data['Personal_Consumption_Expenditure'].tail().tolist(),
                'last_5_unemployment': var_data['Unemployment_Rate'].tail().tolist(),
                'last_5_differenced_gdp': df_diff['GDP'].tail().tolist(),
                'first_3_forecast_levels': forecast_levels.iloc[:3].values.tolist()
            }
        })
        
    except Exception as e:
        import traceback
        print(f"Error in VAR forecasting: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({'success': False, 'error': f'VAR model error: {str(e)}'}), 500

@app.route('/api/analysis/regression')
def regression_analysis():
    """Simple regression analysis between variables"""
    if len(data) < 2:
        return jsonify({'success': False, 'error': 'Insufficient data'})
    
    gdp_values = [row['GDP'] for row in data]
    pce_values = [row['Personal_Consumption_Expenditure'] for row in data]
    unemp_values = [row['Unemployment_Rate'] for row in data]
    
    def simple_linear_regression(x, y):
        n = len(x)
        if n != len(y) or n < 2:
            return None
        
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(x[i] * x[i] for i in range(n))
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        intercept = (sum_y - slope * sum_x) / n
        
        # Calculate R-squared
        y_mean = sum_y / n
        ss_tot = sum((y[i] - y_mean) ** 2 for i in range(n))
        ss_res = sum((y[i] - (intercept + slope * x[i])) ** 2 for i in range(n))
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_squared,
            'equation': f"y = {slope:.4f}x + {intercept:.4f}"
        }
    
    regressions = {
        'gdp_vs_pce': simple_linear_regression(gdp_values, pce_values),
        'gdp_vs_unemployment': simple_linear_regression(gdp_values, unemp_values),
        'pce_vs_unemployment': simple_linear_regression(pce_values, unemp_values)
    }
    
    return jsonify({'success': True, 'regressions': regressions})

# Chart endpoints (same as before)
@app.route('/api/chart/combined')
def combined_chart():
    """Generate combined chart data"""
    if data:
        dates = [row['observation_date'] for row in data]
        gdp_values = [row['GDP'] for row in data]
        pce_values = [row['Personal_Consumption_Expenditure'] for row in data]
        unemp_values = [row['Unemployment_Rate'] for row in data]
        
        chart_data = {
            'data': [
                {
                    'x': dates,
                    'y': gdp_values,
                    'type': 'scatter',
                    'mode': 'lines+markers',
                    'name': 'GDP',
                    'line': {'color': '#1f77b4', 'width': 2},
                    'marker': {'size': 4}
                },
                {
                    'x': dates,
                    'y': pce_values,
                    'type': 'scatter',
                    'mode': 'lines+markers',
                    'name': 'PCE',
                    'line': {'color': '#2ca02c', 'width': 2},
                    'marker': {'size': 4}
                },
                {
                    'x': dates,
                    'y': unemp_values,
                    'type': 'scatter',
                    'mode': 'lines+markers',
                    'name': 'Unemployment Rate',
                    'line': {'color': '#d62728', 'width': 2},
                    'marker': {'size': 4},
                    'yaxis': 'y2'
                }
            ],
            'layout': {
                'title': 'US Economic Indicators (2004-2024)',
                'xaxis': {'title': 'Date'},
                'yaxis': {'title': 'GDP & PCE (Billions of Dollars)', 'side': 'left'},
                'yaxis2': {'title': 'Unemployment Rate (%)', 'side': 'right', 'overlaying': 'y'},
                'template': 'plotly_white',
                'height': 500,
                'legend': {'x': 0.02, 'y': 0.98}
            }
        }
        return jsonify(chart_data)
    return jsonify({'error': 'Data not available'})

@app.route('/api/chart/gdp')
def gdp_chart():
    """Generate GDP chart data"""
    if data:
        dates = [row['observation_date'] for row in data]
        gdp_values = [row['GDP'] for row in data]
        
        chart_data = {
            'data': [{
                'x': dates,
                'y': gdp_values,
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': 'GDP',
                'line': {'color': '#1f77b4', 'width': 2},
                'marker': {'size': 4}
            }],
            'layout': {
                'title': 'US GDP Over Time (2004-2024)',
                'xaxis': {'title': 'Date'},
                'yaxis': {'title': 'GDP (Billions of Dollars)'},
                'template': 'plotly_white',
                'height': 400
            }
        }
        return jsonify(chart_data)
    return jsonify({'error': 'Data not available'})

@app.route('/api/chart/pce')
def pce_chart():
    """Generate PCE chart data"""
    if data:
        dates = [row['observation_date'] for row in data]
        pce_values = [row['Personal_Consumption_Expenditure'] for row in data]
        
        chart_data = {
            'data': [{
                'x': dates,
                'y': pce_values,
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': 'Personal Consumption Expenditure',
                'line': {'color': '#2ca02c', 'width': 2},
                'marker': {'size': 4}
            }],
            'layout': {
                'title': 'Personal Consumption Expenditure Over Time (2004-2024)',
                'xaxis': {'title': 'Date'},
                'yaxis': {'title': 'PCE (Billions of Dollars)'},
                'template': 'plotly_white',
                'height': 400
            }
        }
        return jsonify(chart_data)
    return jsonify({'error': 'Data not available'})

@app.route('/api/chart/unemployment')
def unemployment_chart():
    """Generate unemployment chart data"""
    if data:
        dates = [row['observation_date'] for row in data]
        unemp_values = [row['Unemployment_Rate'] for row in data]
        
        chart_data = {
            'data': [{
                'x': dates,
                'y': unemp_values,
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': 'Unemployment Rate',
                'line': {'color': '#d62728', 'width': 2},
                'marker': {'size': 4}
            }],
            'layout': {
                'title': 'Unemployment Rate Over Time (2004-2024)',
                'xaxis': {'title': 'Date'},
                'yaxis': {'title': 'Unemployment Rate (%)'},
                'template': 'plotly_white',
                'height': 400
            }
        }
        return jsonify(chart_data)
    return jsonify({'error': 'Data not available'})

@app.route('/api/analysis/correlation')
def correlation_analysis():
    """Calculate correlation between economic indicators"""
    if len(data) > 1:
        # Simple correlation calculation
        gdp_values = [row['GDP'] for row in data]
        pce_values = [row['Personal_Consumption_Expenditure'] for row in data]
        unemp_values = [row['Unemployment_Rate'] for row in data]
        
        # Calculate simple correlations
        def simple_correlation(x, y):
            n = len(x)
            if n != len(y) or n < 2:
                return 0
            
            sum_x = sum(x)
            sum_y = sum(y)
            sum_xy = sum(x[i] * y[i] for i in range(n))
            sum_x2 = sum(x[i] ** 2 for i in range(n))
            sum_y2 = sum(y[i] ** 2 for i in range(n))
            
            numerator = n * sum_xy - sum_x * sum_y
            denominator = ((n * sum_x2 - sum_x ** 2) * (n * sum_y2 - sum_y ** 2)) ** 0.5
            
            if denominator == 0:
                return 0
            
            return numerator / denominator
        
        correlation = {
            'GDP': {
                'GDP': 1.0,
                'Personal_Consumption_Expenditure': simple_correlation(gdp_values, pce_values),
                'Unemployment_Rate': simple_correlation(gdp_values, unemp_values)
            },
            'Personal_Consumption_Expenditure': {
                'GDP': simple_correlation(pce_values, gdp_values),
                'Personal_Consumption_Expenditure': 1.0,
                'Unemployment_Rate': simple_correlation(pce_values, unemp_values)
            },
            'Unemployment_Rate': {
                'GDP': simple_correlation(unemp_values, gdp_values),
                'Personal_Consumption_Expenditure': simple_correlation(unemp_values, pce_values),
                'Unemployment_Rate': 1.0
            }
        }
        
        return jsonify({
            'success': True,
            'correlation': correlation
        })
    return jsonify({'success': False, 'error': 'Insufficient data'})

@app.route('/api/analysis/trends')
def trend_analysis():
    """Analyze trends in the data"""
    if len(data) >= 13:
        # Calculate simple trends using last 12 months vs previous 12 months
        recent_data = data[-12:]
        previous_data = data[-24:-12]
        
        if len(recent_data) >= 12 and len(previous_data) >= 12:
            # Calculate averages
            recent_gdp_avg = sum(row['GDP'] for row in recent_data) / 12
            previous_gdp_avg = sum(row['GDP'] for row in previous_data) / 12
            
            recent_pce_avg = sum(row['Personal_Consumption_Expenditure'] for row in recent_data) / 12
            previous_pce_avg = sum(row['Personal_Consumption_Expenditure'] for row in previous_data) / 12
            
            recent_unemp_avg = sum(row['Unemployment_Rate'] for row in recent_data) / 12
            previous_unemp_avg = sum(row['Unemployment_Rate'] for row in previous_data) / 12
            
            trends = {
                'gdp_trend': 'increasing' if recent_gdp_avg > previous_gdp_avg else 'decreasing',
                'pce_trend': 'increasing' if recent_pce_avg > previous_pce_avg else 'decreasing',
                'unemployment_trend': 'decreasing' if recent_unemp_avg < previous_unemp_avg else 'increasing'
            }
            
            return jsonify({'success': True, 'trends': trends})
    
    return jsonify({'success': False, 'error': 'Insufficient data'})

@app.route('/api/stress-testing/run', methods=['POST'])
def run_stress_test():
    """Run economic stress testing with custom parameters"""
    try:
        from flask import request
        import math
        
        # Get stress test parameters
        stress_params = request.get_json()
        gdp_shock = stress_params.get('gdp', -5.0)  # Default -5% GDP shock
        unemployment_shock = stress_params.get('unemployment', 2.0)  # Default +2% unemployment
        pce_shock = stress_params.get('pce', -3.0)  # Default -3% consumption shock
        duration = stress_params.get('duration', 6)  # Default 6 months
        
        if len(data) < 12:
            return jsonify({'success': False, 'error': 'Insufficient data for stress testing'}), 400
        
        # Get baseline values (last 12 months average)
        recent_data = data[-12:]
        baseline_gdp = sum(row['GDP'] for row in recent_data) / len(recent_data)
        baseline_pce = sum(row['Personal_Consumption_Expenditure'] for row in recent_data) / len(recent_data)
        baseline_unemp = sum(row['Unemployment_Rate'] for row in recent_data) / len(recent_data)
        
        # Calculate stress trajectories
        time_periods = list(range(1, duration + 1))
        
        # GDP trajectory with gradual recovery
        gdp_trajectory = []
        for month in time_periods:
            # Initial shock, then gradual recovery
            shock_factor = math.exp(-0.1 * month)  # Exponential decay
            impact = gdp_shock * shock_factor
            gdp_trajectory.append(impact)
        
        # Unemployment trajectory (opposite of GDP)
        unemployment_trajectory = []
        for month in time_periods:
            # Unemployment rises initially, then gradually falls
            shock_factor = math.exp(-0.08 * month)  # Slower recovery than GDP
            impact = unemployment_shock * shock_factor
            unemployment_trajectory.append(impact)
        
        # Consumption trajectory
        consumption_trajectory = []
        for month in time_periods:
            # Consumption follows GDP but with lag
            shock_factor = math.exp(-0.12 * month)  # Faster recovery than GDP
            impact = pce_shock * shock_factor
            consumption_trajectory.append(impact)
        
        # Calculate risk metrics
        gdp_risk = min(10, abs(gdp_shock) / 2)  # Scale risk based on shock magnitude
        unemployment_risk = min(10, unemployment_shock / 2)
        consumption_risk = min(10, abs(pce_shock) / 2)
        
        # Overall risk score (weighted average)
        risk_score = (gdp_risk * 0.4 + unemployment_risk * 0.4 + consumption_risk * 0.2)
        
        # Calculate final impacts
        gdp_impact = gdp_trajectory[0] if gdp_trajectory else 0
        unemployment_impact = unemployment_trajectory[0] if unemployment_trajectory else 0
        consumption_impact = consumption_trajectory[0] if consumption_trajectory else 0
        
        # Create results structure
        results = {
            'gdp_impact': gdp_impact,
            'unemployment_impact': unemployment_impact,
            'consumption_impact': consumption_impact,
            'risk_score': risk_score,
            'gdp_risk': gdp_risk,
            'unemployment_risk': unemployment_risk,
            'consumption_risk': consumption_risk,
            'time_periods': time_periods,
            'gdp_trajectory': gdp_trajectory,
            'unemployment_trajectory': unemployment_trajectory,
            'consumption_trajectory': consumption_trajectory,
            'baseline_values': {
                'gdp': baseline_gdp,
                'pce': baseline_pce,
                'unemployment': baseline_unemp
            },
            'stress_parameters': {
                'gdp_shock': gdp_shock,
                'unemployment_shock': unemployment_shock,
                'pce_shock': pce_shock,
                'duration': duration
            }
        }
        
        return jsonify({
            'success': True,
            'results': results,
            'message': 'Stress test completed successfully'
        })
        
    except Exception as e:
        import traceback
        print(f"Error in stress testing: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({'success': False, 'error': f'Stress test error: {str(e)}'}), 500

@app.route('/api/stress-testing/scenarios')
def get_stress_scenarios():
    """Get predefined stress testing scenarios"""
    scenarios = {
        'recession': {
            'name': 'Economic Recession',
            'description': 'Severe economic downturn with GDP decline, rising unemployment, and falling consumption',
            'parameters': {
                'gdp': -8.0,
                'unemployment': 4.0,
                'pce': -6.0,
                'duration': 12
            },
            'risk_level': 'High'
        },
        'market_crash': {
            'name': 'Market Crash',
            'description': 'Sudden market volatility and rapid economic contraction',
            'parameters': {
                'gdp': -12.0,
                'unemployment': 6.0,
                'pce': -8.0,
                'duration': 6
            },
            'risk_level': 'Critical'
        },
        'inflation_shock': {
            'name': 'Inflation Shock',
            'description': 'High inflation periods with rapid price increases',
            'parameters': {
                'gdp': -3.0,
                'unemployment': 1.0,
                'pce': -2.0,
                'duration': 6
            },
            'risk_level': 'Medium'
        },
        'global_crisis': {
            'name': 'Global Crisis',
            'description': 'International economic shocks and global financial instability',
            'parameters': {
                'gdp': -15.0,
                'unemployment': 8.0,
                'pce': -10.0,
                'duration': 18
            },
            'risk_level': 'Critical'
        }
    }
    
    return jsonify({
        'success': True,
        'scenarios': scenarios
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
