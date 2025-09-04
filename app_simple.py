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

# VAR Helper Functions for Stress Testing
def fit_var_model(data_df):
    """Fit VAR model to data - same logic as forecasting"""
    try:
        import pandas as pd
        from statsmodels.tsa.vector_ar.var_model import VAR
        
        # Differencing for stationarity (same as forecasting)
        df_diff = data_df.diff().dropna()
        
        if len(df_diff) < 12:
            return None
        
        # Fit VAR model (same as forecasting)
        model = VAR(df_diff)
        var_results = model.fit(maxlags=4, ic="aic")
        
        return var_results
    except Exception as e:
        print(f"Error fitting VAR model: {e}")
        return None

def generate_var_forecast(var_model, data_df, steps=12):
    """Generate VAR forecast - same logic as forecasting"""
    try:
        import pandas as pd
        import numpy as np
        
        # Differencing for stationarity (same as forecasting)
        df_diff = data_df.diff().dropna()
        
        # Get lag order from fitted model
        lag_order = var_model.k_ar
        
        # Forecast in differenced space (same as forecasting)
        forecast_diff = var_model.forecast(df_diff.values[-lag_order:], steps=steps)
        
        # Reconstruct original levels (undo differencing) - same as forecasting
        last_values = data_df.iloc[-1]
        
        # Convert forecast_diff to DataFrame for cumsum (same as forecasting)
        forecast_diff_df = pd.DataFrame(
            forecast_diff,
            columns=data_df.columns
        )
        
        # Cumulative sum of differenced forecasts + last actual value (same as forecasting)
        forecast_levels = forecast_diff_df.cumsum() + last_values.values
        
        # Handle any NaN values that might occur
        forecast_levels = np.where(np.isnan(forecast_levels), 0, forecast_levels)
        
        # Create forecast DataFrame with proper index (same as forecasting)
        forecast_dates = pd.date_range(
            start=data_df.index[-1] + pd.DateOffset(months=1),
            periods=steps,
            freq='MS'
        )
        
        forecast_df = pd.DataFrame(
            forecast_levels,
            index=forecast_dates,
            columns=data_df.columns
        )
        
        # Ensure all values are finite numbers
        for col in forecast_df.columns:
            forecast_df[col] = forecast_df[col].replace([np.inf, -np.inf], 0)
            forecast_df[col] = forecast_df[col].fillna(0)
        
        return forecast_df
    except Exception as e:
        print(f"Error generating VAR forecast: {e}")
        return None

def apply_stress_scenarios(baseline_forecast, scenarios):
    """Apply stress scenarios to baseline forecast with FIXED percentage deviation"""
    try:
        import pandas as pd
        import numpy as np
        
        stress_results = {}
        
        for scenario_name, shocks in scenarios.items():
            stressed = baseline_forecast.copy()
            
            for var_name, shock in shocks.items():
                if var_name in stressed.columns:
                    # Get baseline values for this variable
                    baseline_values = baseline_forecast[var_name].values
                    
                    # Apply FIXED shock percentage to baseline forecast (no decay)
                    # shock is already in decimal format (e.g., -0.05 for -5%)
                    for i in range(len(stressed)):
                        # Apply shock as FIXED percentage deviation from baseline
                        # If shock is -0.05, then stressed = baseline * (1 - 0.05) = baseline * 0.95
                        # This means stressed values will be consistently 5% below baseline
                        # For unemployment, this is also a percentage change (e.g., 0.02 = 2% increase)
                        new_value = baseline_values[i] * (1 + shock)
                        
                        # Ensure the value follows the baseline pattern but with fixed deviation
                        stressed.iloc[i][var_name] = new_value
                    
                    # Debug: Print first few values to verify fixed percentage
                    print(f"=== {scenario_name} - {var_name} DEBUG ===")
                    print(f"Shock: {shock} ({shock*100:.1f}%)")
                    print(f"First 3 baseline: {baseline_values[:3]}")
                    print(f"First 3 stressed: {stressed[var_name].iloc[:3].values}")
                    if len(baseline_values) > 0:
                        first_deviation = ((stressed[var_name].iloc[0] - baseline_values[0]) / baseline_values[0]) * 100
                        print(f"First period deviation: {first_deviation:.2f}%")
                        if len(baseline_values) > 1:
                            last_deviation = ((stressed[var_name].iloc[-1] - baseline_values[-1]) / baseline_values[-1]) * 100
                            print(f"Last period deviation: {last_deviation:.2f}%")
                    print("=== END DEBUG ===")
            
            # Final cleanup: replace any remaining NaN or inf values
            for col in stressed.columns:
                stressed[col] = stressed[col].replace([np.inf, -np.inf], 0)
                stressed[col] = stressed[col].fillna(0)
            
            stress_results[scenario_name] = stressed
        
        return stress_results
    except Exception as e:
        print(f"Error applying stress scenarios: {e}")
        return None

def calculate_var_risk_metrics(stressed_forecast, baseline_forecast):
    """Calculate risk metrics for VAR stress testing"""
    try:
        risk_metrics = {}
        
        for column in stressed_forecast.columns:
            baseline_values = baseline_forecast[column].values
            stressed_values = stressed_forecast[column].values
            
            # Calculate maximum deviation
            max_deviation = max(abs(stressed_values - baseline_values))
            
            # Calculate average deviation
            avg_deviation = sum(abs(stressed_values - baseline_values)) / len(stressed_values)
            
            # Calculate risk score (0-10)
            risk_score = min(10, max_deviation / baseline_values[0] * 100)
            
            risk_metrics[column] = {
                'max_deviation': float(max_deviation),
                'avg_deviation': float(avg_deviation),
                'risk_score': float(risk_score)
            }
        
        return risk_metrics
    except Exception as e:
        print(f"Error calculating risk metrics: {e}")
        return None

def calculate_stress_scenario_warnings(historical_data, baseline_forecast, stress_params):
    """Calculate warning signals for stress scenarios"""
    try:
        import pandas as pd
        import numpy as np
        from sklearn.preprocessing import StandardScaler
        
        # Convert to DataFrame if it's a list
        if isinstance(historical_data, list):
            df = pd.DataFrame(historical_data)
        else:
            df = historical_data.copy()
        
        # Ensure we have the right columns
        required_columns = ['GDP', 'Personal_Consumption_Expenditure', 'Unemployment_Rate']
        for col in required_columns:
            if col not in df.columns:
                return {'warnings': {}, 'warning_dates': {}}
        
        # Convert to numeric and handle any non-numeric values
        for col in required_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove any rows with NaN values
        df_clean = df.dropna(subset=required_columns)
        
        if len(df_clean) < 24:
            return {'warnings': {}, 'warning_dates': {}}
        
        # Calculate indicators from levels
        def calculate_indicators(levels_df):
            indicators = pd.DataFrame(index=levels_df.index)
            indicators['GDP_growth'] = levels_df['GDP'].pct_change()
            indicators['PCE_growth'] = levels_df['Personal_Consumption_Expenditure'].pct_change()
            indicators['Unemployment_change'] = levels_df['Unemployment_Rate'].diff()
            return indicators.dropna()
        
        # Calculate indicators for historical data
        hist_indicators = calculate_indicators(df_clean)
        
        # Fit StandardScaler on historical data
        scaler = StandardScaler()
        z_hist = scaler.fit_transform(hist_indicators[['GDP_growth', 'PCE_growth', 'Unemployment_change']])
        z_hist_df = pd.DataFrame(z_hist, index=hist_indicators.index, 
                                columns=['GDP_growth', 'PCE_growth', 'Unemployment_change'])
        
        # Calculate thresholds in z-space
        K = 1.0
        thr_std = {}
        for col in ['GDP_growth', 'PCE_growth', 'Unemployment_change']:
            thr_std[col] = np.mean(z_hist_df[col]) + K * np.std(z_hist_df[col])
        
        # Create stress scenario data
        if baseline_forecast is not None:
            # Apply stress shocks to baseline forecast
            stress_forecast = baseline_forecast.copy()
            stress_forecast['GDP'] = stress_forecast['GDP'] * (1 + stress_params.get('gdp_shock', 0))
            stress_forecast['Personal_Consumption_Expenditure'] = stress_forecast['Personal_Consumption_Expenditure'] * (1 + stress_params.get('pce_shock', 0))
            # For unemployment, apply as percentage change (e.g., 0.02 = 2% increase)
            stress_forecast['Unemployment_Rate'] = stress_forecast['Unemployment_Rate'] * (1 + stress_params.get('unemployment_shock', 0))
            
            # Calculate indicators for stress scenario
            stress_indicators = calculate_indicators(stress_forecast)
            
            # Apply same scaler and thresholds to stress data
            z_stress = scaler.transform(stress_indicators[['GDP_growth', 'PCE_growth', 'Unemployment_change']])
            z_stress_df = pd.DataFrame(z_stress, index=stress_indicators.index, 
                                     columns=['GDP_growth', 'PCE_growth', 'Unemployment_change'])
            
            # Calculate warning signals for stress scenario
            gdp_warning = (z_stress_df['GDP_growth'] < -thr_std['GDP_growth'])
            pce_warning = (z_stress_df['PCE_growth'] < -thr_std['PCE_growth'])
            unemp_warning = (z_stress_df['Unemployment_change'] > thr_std['Unemployment_change'])
            
            # Count warnings
            warnings = {
                'gdp': int(gdp_warning.sum()),
                'pce': int(pce_warning.sum()),
                'unemployment': int(unemp_warning.sum())
            }
            
            # Get warning dates
            warning_dates = {}
            for indicator, warning_series in [('GDP_growth', gdp_warning), 
                                            ('PCE_growth', pce_warning), 
                                            ('Unemployment_change', unemp_warning)]:
                warning_indices = warning_series[warning_series].index
                warning_dates[indicator] = [idx.strftime('%Y-%m-%d') for idx in warning_indices]
            
            return {'warnings': warnings, 'warning_dates': warning_dates}
        else:
            return {'warnings': {}, 'warning_dates': {}}
            
    except Exception as e:
        print(f"Error calculating stress scenario warnings: {e}")
        return {'warnings': {}, 'warning_dates': {}}

def calculate_ews_risk_score(historical_data, baseline_forecast, stress_scenario):
    """Calculate EWS Risk Score using proper z-score methodology with StandardScaler"""
    try:
        import pandas as pd
        import numpy as np
        from sklearn.preprocessing import StandardScaler
        
        # Debug: Check data structure
        print(f"=== EWS DEBUG (Z-Score Method) ===")
        print(f"Historical data type: {type(historical_data)}")
        print(f"Historical data length: {len(historical_data) if hasattr(historical_data, '__len__') else 'N/A'}")
        
        # Convert to DataFrame if it's a list
        if isinstance(historical_data, list):
            df = pd.DataFrame(historical_data)
        else:
            df = historical_data.copy()
        
        # Ensure we have the right columns
        required_columns = ['GDP', 'Personal_Consumption_Expenditure', 'Unemployment_Rate']
        for col in required_columns:
            if col not in df.columns:
                print(f"Missing column: {col}")
                return {
                    'error': f'Missing required column: {col}',
                    'risk_score': 0,
                    'risk_level': 'Unknown'
                }
        
        # Convert to numeric and handle any non-numeric values
        for col in required_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove any rows with NaN values
        df_clean = df.dropna(subset=required_columns)
        
        if len(df_clean) < 24:  # Need at least 24 observations for robust statistics
            print(f"Insufficient clean data: {len(df_clean)} rows (minimum 24 required)")
            return {
                'error': 'Insufficient clean data for EWS calculation (minimum 24 observations required)',
                'risk_score': 0,
                'risk_level': 'Unknown'
            }
        
        print(f"Clean data shape: {df_clean.shape}")
        print(f"GDP range: {df_clean['GDP'].min()} to {df_clean['GDP'].max()}")
        print(f"PCE range: {df_clean['Personal_Consumption_Expenditure'].min()} to {df_clean['Personal_Consumption_Expenditure'].max()}")
        print(f"Unemployment range: {df_clean['Unemployment_Rate'].min()} to {df_clean['Unemployment_Rate'].max()}")
        
        # 1. Calculate indicators from levels (proper methodology)
        def calculate_indicators(levels_df):
            """Calculate GDP_growth, PCE_growth, Unemployment_change from levels"""
            indicators = pd.DataFrame(index=levels_df.index)
            indicators['GDP_growth'] = levels_df['GDP'].pct_change()
            indicators['PCE_growth'] = levels_df['Personal_Consumption_Expenditure'].pct_change()
            indicators['Unemployment_change'] = levels_df['Unemployment_Rate'].diff()
            return indicators.dropna()
        
        # Calculate indicators for historical data
        hist_indicators = calculate_indicators(df_clean)
        print(f"Historical indicators shape: {hist_indicators.shape}")
        print(f"GDP growth sample: {hist_indicators['GDP_growth'].head().tolist()}")
        print(f"PCE growth sample: {hist_indicators['PCE_growth'].head().tolist()}")
        print(f"Unemployment change sample: {hist_indicators['Unemployment_change'].head().tolist()}")
        
        # 2. Fit StandardScaler ONLY on historical data (avoid look-ahead bias)
        scaler = StandardScaler()
        z_hist = scaler.fit_transform(hist_indicators[['GDP_growth', 'PCE_growth', 'Unemployment_change']])
        z_hist_df = pd.DataFrame(z_hist, index=hist_indicators.index, 
                                columns=['GDP_growth', 'PCE_growth', 'Unemployment_change'])
        
        print(f"Z-scores mean: {z_hist_df.mean().tolist()}")
        print(f"Z-scores std: {z_hist_df.std().tolist()}")
        
        # 3. Calculate thresholds in z-space (K = 1.5 for enhanced sensitivity as requested)
        K = 1.5  # Changed from 1.0 to 1.5 as per user request
        thr_std = {}
        for col in ['GDP_growth', 'PCE_growth', 'Unemployment_change']:
            thr_std[col] = np.mean(z_hist_df[col]) + K * np.std(z_hist_df[col])
        
        print(f"Z-space thresholds (K={K}): {thr_std}")
        
        # 4. Convert to raw units for interpretation
        mu = scaler.mean_
        sigma = scaler.scale_
        thr_raw = {}
        for i, col in enumerate(['GDP_growth', 'PCE_growth', 'Unemployment_change']):
            if col in ['GDP_growth', 'PCE_growth']:
                # Lower tail risk (negative growth)
                thr_raw[f"{col}_neg"] = mu[i] - thr_std[col] * sigma[i]
            else:
                # Upper tail risk (positive unemployment change)
                thr_raw[f"{col}_pos"] = mu[i] + thr_std[col] * sigma[i]
        
        print(f"Raw thresholds: {thr_raw}")
        
        # 5. Calculate warning signals for historical data
        def compute_signals(indicators_df, scaler, thr_std):
            """Compute warning signals for given indicators"""
            # Transform to z-space
            z_scores = scaler.transform(indicators_df[['GDP_growth', 'PCE_growth', 'Unemployment_change']])
            z_df = pd.DataFrame(z_scores, index=indicators_df.index, 
                              columns=['GDP_growth', 'PCE_growth', 'Unemployment_change'])
            
            # Warning signals (proper z-score methodology)
            gdp_warning = (z_df['GDP_growth'] < -thr_std['GDP_growth'])
            pce_warning = (z_df['PCE_growth'] < -thr_std['PCE_growth'])
            unemp_warning = (z_df['Unemployment_change'] > thr_std['Unemployment_change'])
            
            # EWS score (0-3)
            ews_score = gdp_warning.astype(int) + pce_warning.astype(int) + unemp_warning.astype(int)
            
            return {
                'gdp_warning': gdp_warning,
                'pce_warning': pce_warning,
                'unemp_warning': unemp_warning,
                'ews_score': ews_score,
                'z_scores': z_df
            }
        
        # Calculate signals for historical data
        hist_signals = compute_signals(hist_indicators, scaler, thr_std)
        
        # 6. Calculate overall risk metrics
        total_warnings = (hist_signals['gdp_warning'].sum() + 
                         hist_signals['pce_warning'].sum() + 
                         hist_signals['unemp_warning'].sum())
        
        # 7. Get warning dates for each indicator
        warning_dates = {}
        
        # Create a mapping from hist_indicators index to actual dates
        # hist_indicators starts from the second row of data (index 1)
        date_mapping = {}
        for i in range(len(hist_indicators)):
            if i + 1 < len(data):
                # Use the actual date from the data list
                date_mapping[i] = data[i + 1]['observation_date']
        
        print(f"DEBUG: Created date mapping for {len(date_mapping)} indices")
        print(f"DEBUG: Sample mapping: {list(date_mapping.items())[:5]}")
        
        for indicator, warning_series in [('GDP_growth', hist_signals['gdp_warning']), 
                                        ('PCE_growth', hist_signals['pce_warning']), 
                                        ('Unemployment_change', hist_signals['unemp_warning'])]:
            # Get warning dates safely - use date mapping
            warning_indices = warning_series[warning_series].index
            warning_dates_list = []
            
            print(f"DEBUG: {indicator} warning indices: {warning_indices}")
            
            for idx in warning_indices:
                try:
                    # Get the actual date from the date mapping
                    if hasattr(idx, 'strftime'):
                        # If idx is already a datetime
                        warning_dates_list.append(idx.strftime('%Y-%m-%d'))
                    else:
                        # If idx is a position number, get the corresponding date from mapping
                        if isinstance(idx, (int, np.integer)) and idx in date_mapping:
                            warning_dates_list.append(date_mapping[idx])
                        else:
                            warning_dates_list.append(str(idx))
                except Exception as e:
                    print(f"Warning: Could not convert index {idx} to date: {e}")
                    warning_dates_list.append(str(idx))
            
            warning_dates[indicator] = warning_dates_list
        
        print(f"Warning dates:")
        for indicator, dates in warning_dates.items():
            print(f"  {indicator}: {dates}")
        
        # Risk score (0-10 scale)
        risk_score = min(10, total_warnings / len(hist_indicators) * 100)
        
        print(f"Total warnings: {total_warnings}")
        print(f"Risk score: {risk_score:.2f}")
        
        # 8. Calculate forecasting-based warnings (NEW FEATURE)
        forecast_warnings = {}
        forecast_warning_dates = {}
        
        if baseline_forecast is not None:
            print("=== FORECASTING EWS ANALYSIS ===")
            
            # Calculate indicators for forecast data
            forecast_indicators = calculate_indicators(baseline_forecast)
            print(f"Forecast indicators shape: {forecast_indicators.shape}")
            
            # Apply same scaler and thresholds to forecast data
            forecast_signals = compute_signals(forecast_indicators, scaler, thr_std)
            
            # Count forecast warnings
            forecast_warnings = {
                'gdp': int(forecast_signals['gdp_warning'].sum()),
                'pce': int(forecast_signals['pce_warning'].sum()),
                'unemployment': int(forecast_signals['unemp_warning'].sum())
            }
            
            # Get forecast warning dates
            forecast_date_mapping = {}
            for i in range(len(forecast_indicators)):
                if i < len(baseline_forecast):
                    forecast_date_mapping[i] = baseline_forecast.index[i].strftime('%Y-%m-%d')
            
            for indicator, warning_series in [('GDP_growth', forecast_signals['gdp_warning']), 
                                            ('PCE_growth', forecast_signals['pce_warning']), 
                                            ('Unemployment_change', forecast_signals['unemp_warning'])]:
                warning_indices = warning_series[warning_series].index
                forecast_warning_dates_list = []
                
                for idx in warning_indices:
                    if isinstance(idx, (int, np.integer)) and idx in forecast_date_mapping:
                        forecast_warning_dates_list.append(forecast_date_mapping[idx])
                    else:
                        forecast_warning_dates_list.append(str(idx))
                
                forecast_warning_dates[indicator] = forecast_warning_dates_list
            
            print(f"Forecast warnings: {forecast_warnings}")
            print(f"Forecast warning dates: {forecast_warning_dates}")
            print("=== END FORECASTING EWS ===")
        
        # 9. Risk level determination
        if risk_score < 3:
            risk_level = "Low"
        elif risk_score < 6:
            risk_level = "Medium"
        elif risk_score < 8:
            risk_level = "High"
        else:
            risk_level = "Critical"
        
        print(f"Risk level: {risk_level}")
        print("=== END EWS DEBUG ===")
        
        return {
            'risk_score': round(risk_score, 2),
            'risk_level': risk_level,
            'warnings': {
                'gdp': int(hist_signals['gdp_warning'].sum()),
                'pce': int(hist_signals['pce_warning'].sum()),
                'unemployment': int(hist_signals['unemp_warning'].sum())
            },
            'warning_dates': warning_dates,
            'forecast_warnings': forecast_warnings,
            'forecast_warning_dates': forecast_warning_dates,
            'thresholds': {
                'gdp_growth_neg': round(thr_raw['GDP_growth_neg'], 4),
                'pce_growth_neg': round(thr_raw['PCE_growth_neg'], 4),
                'unemployment_change_pos': round(thr_raw['Unemployment_change_pos'], 4)
            },
            'z_thresholds': {
                'gdp_growth': round(thr_std['GDP_growth'], 4),
                'pce_growth': round(thr_std['PCE_growth'], 4),
                'unemployment_change': round(thr_std['Unemployment_change'], 4)
            },
            'k_value': K,
            'methodology': 'StandardScaler + Z-score + 1.5×Std Dev + Forecasting EWS',
            'scaler_params': {
                'mean': mu.tolist(),
                'scale': sigma.tolist()
            }
        }
        
    except Exception as e:
        import traceback
        print(f"EWS Error: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return {
            'error': str(e),
            'risk_score': 0,
            'risk_level': 'Unknown'
        }

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

@app.route('/api/forecasting/arima')
def arima_forecasting():
    """ARIMA forecasting model - EXACT notebook logic"""
    try:
        from flask import request
        import numpy as np
        import pandas as pd
        from statsmodels.tsa.arima.model import ARIMA
        
        # Get forecast period from query parameters
        forecast_periods = request.args.get('periods', 6, type=int)
        forecast_periods = max(6, min(36, forecast_periods))
        
        # Check if data is loaded
        if data is None or len(data) < 24:  # ARIMA needs more data
            return jsonify({'success': False, 'error': 'Insufficient data for ARIMA model (minimum 24 observations)'}), 400
        
        # ---------------------------
        # Step 1: Load your dataset (matching notebook exactly)
        # ---------------------------
        df = pd.DataFrame(data)
        df['observation_date'] = pd.to_datetime(df['observation_date'])
        df.set_index('observation_date', inplace=True)
        
        # Select variables for ARIMA model (matching notebook column names exactly)
        arima_data = df[['GDP', 'Personal_Consumption_Expenditure', 'Unemployment_Rate']].copy()
        
        # DEBUG: Print data info to compare with notebook
        print("=== ARIMA FORECASTING DEBUG INFO ===")
        print(f"Data shape: {arima_data.shape}")
        print(f"Date range: {arima_data.index[0]} to {arima_data.index[-1]}")
        print(f"Last 5 GDP values: {arima_data['GDP'].tail().values}")
        print(f"Last 5 PCE values: {arima_data['Personal_Consumption_Expenditure'].tail().values}")
        print(f"Last 5 Unemployment values: {arima_data['Unemployment_Rate'].tail().values}")
        
        # ---------------------------
        # Step 2: Detect frequency automatically from df index (EXACT notebook method)
        # ---------------------------
        freq = pd.infer_freq(arima_data.index) or "M"   # fallback to monthly if not detected
        
        # ---------------------------
        # Step 3: Forecast index (future periods) (EXACT notebook method)
        # ---------------------------
        forecast_index = pd.date_range(start=arima_data.index[-1] + pd.tseries.frequencies.to_offset(freq),
                                       periods=forecast_periods, freq=freq)
        
        # ---------------------------
        # Step 4: Store forecasts (EXACT notebook method)
        # ---------------------------
        arima_forecasts = pd.DataFrame(index=forecast_index, columns=arima_data.columns)
        
        # ---------------------------
        # Step 5: Fit ARIMA and forecast each series (EXACT notebook method)
        # ---------------------------
        model_info = {}
        forecasts_with_ci = []
        
        for col in arima_data.columns:
            try:
                # Fit ARIMA model with order (1,1,1) - you can tune (p,d,q)
                model = ARIMA(arima_data[col], order=(1,1,1))
                model_fit = model.fit()
                
                # Get forecast with confidence intervals
                forecast_result = model_fit.get_forecast(steps=forecast_periods)
                forecast_values = forecast_result.predicted_mean
                confidence_intervals = forecast_result.conf_int()
                
                # Store forecasts
                arima_forecasts[col] = forecast_values.values
                
                # Store model info
                model_info[col] = {
                    'aic': float(model_fit.aic),
                    'bic': float(model_fit.bic),
                    'order': (1, 1, 1),
                    'params': model_fit.params.to_dict()
                }
                
                # Create forecast data structure for this variable
                for i in range(forecast_periods):
                    if i >= len(forecasts_with_ci):
                        forecasts_with_ci.append({})
                    
                    # Map column names to the expected format
                    if col == 'GDP':
                        key = 'gdp'
                    elif col == 'Personal_Consumption_Expenditure':
                        key = 'pce'
                    elif col == 'Unemployment_Rate':
                        key = 'unemployment'
                    else:
                        key = col.lower().replace('_', '')
                    
                    forecasts_with_ci[i][key] = {
                        'value': float(forecast_values.iloc[i]),
                        'lower_bound': float(confidence_intervals.iloc[i, 0]),
                        'upper_bound': float(confidence_intervals.iloc[i, 1])
                    }
                
                print(f"ARIMA model for {col}: AIC={model_fit.aic:.2f}, BIC={model_fit.bic:.2f}")
                
            except Exception as e:
                print(f"Error fitting ARIMA for {col}: {str(e)}")
                # Use simple trend projection as fallback
                recent_values = arima_data[col].tail(12).values
                trend = np.mean(np.diff(recent_values))
                last_value = recent_values[-1]
                
                for i in range(forecast_periods):
                    if i >= len(forecasts_with_ci):
                        forecasts_with_ci.append({})
                    
                    # Map column names to the expected format
                    if col == 'GDP':
                        key = 'gdp'
                    elif col == 'Personal_Consumption_Expenditure':
                        key = 'pce'
                    elif col == 'Unemployment_Rate':
                        key = 'unemployment'
                    else:
                        key = col.lower().replace('_', '')
                    
                    forecast_value = last_value + trend * (i + 1)
                    forecasts_with_ci[i][key] = {
                        'value': float(forecast_value),
                        'lower_bound': float(forecast_value * 0.95),
                        'upper_bound': float(forecast_value * 1.05)
                    }
        
        # DEBUG: Print final forecast info
        print(f"Final ARIMA forecasts shape: {arima_forecasts.shape}")
        print(f"First 3 forecast values: {arima_forecasts.iloc[:3].values}")
        print("=== END ARIMA DEBUG INFO ===")
        
        return jsonify({
            'success': True,
            'forecasts': forecasts_with_ci,
            'model_info': model_info,
            'arima_summary': {
                'frequency_detected': freq,
                'forecast_periods': forecast_periods,
                'models_fitted': len(model_info)
            },
            'debug_info': {
                'last_5_gdp': arima_data['GDP'].tail().tolist(),
                'last_5_pce': arima_data['Personal_Consumption_Expenditure'].tail().tolist(),
                'last_5_unemployment': arima_data['Unemployment_Rate'].tail().tolist(),
                'first_3_forecast_values': arima_forecasts.iloc[:3].values.tolist()
            }
        })
        
    except Exception as e:
        import traceback
        print(f"Error in ARIMA forecasting: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({'success': False, 'error': f'ARIMA model error: {str(e)}'}), 500

@app.route('/api/stress-testing/arima-chart-data', methods=['POST'])
def arima_stress_testing_chart_data():
    """Generate chart data for ARIMA-based stress testing (specifically for unemployment)"""
    try:
        from flask import request
        import numpy as np
        import pandas as pd
        from statsmodels.tsa.arima.model import ARIMA
        
        # Get stress parameters
        params = request.get_json()
        gdp_shock = params.get('gdp', -0.05)
        unemployment_shock = params.get('unemployment', 0.02)
        pce_shock = params.get('pce', -0.03)
        duration = params.get('duration', 12)
        
        # Check if data is loaded
        if data is None or len(data) < 24:
            return jsonify({'success': False, 'error': 'Insufficient data for ARIMA model'}), 400
        
        # Load and prepare data
        df = pd.DataFrame(data)
        df['observation_date'] = pd.to_datetime(df['observation_date'])
        df.set_index('observation_date', inplace=True)
        
        # Get historical data (last 24 months for context)
        historical_data = df.tail(24)
        
        # Generate ARIMA forecast for unemployment (since it's more accurate)
        unemployment_series = df['Unemployment_Rate']
        
        # Fit ARIMA model for unemployment
        model = ARIMA(unemployment_series, order=(1,1,1))
        model_fit = model.fit()
        
        # Generate baseline forecast
        forecast_result = model_fit.get_forecast(steps=duration)
        baseline_forecast = forecast_result.predicted_mean
        
        # Generate stress scenario forecast
        # Apply unemployment shock to the baseline forecast
        stressed_forecast = baseline_forecast * (1 + unemployment_shock)
        
        # Create date ranges
        last_date = df.index[-1]
        freq = pd.infer_freq(df.index) or "M"
        
        # Historical dates
        historical_dates = historical_data.index.strftime('%Y-%m-%d').tolist()
        
        # Future dates for forecasts
        future_dates = pd.date_range(start=last_date + pd.tseries.frequencies.to_offset(freq),
                                   periods=duration, freq=freq).strftime('%Y-%m-%d').tolist()
        
        # Prepare chart data
        chart_data = {
            'historical': {
                'dates': historical_dates,
                'unemployment': historical_data['Unemployment_Rate'].tolist(),
                'gdp': historical_data['GDP'].tolist(),
                'consumption': historical_data['Personal_Consumption_Expenditure'].tolist()
            },
            'baseline': {
                'dates': future_dates,
                'unemployment': baseline_forecast.tolist(),
                'gdp': [],  # Will be filled with VAR or simple projection
                'consumption': []  # Will be filled with VAR or simple projection
            },
            'stressed': {
                'dates': future_dates,
                'unemployment': stressed_forecast.tolist(),
                'gdp': [],  # Will be filled with VAR or simple projection
                'consumption': []  # Will be filled with VAR or simple projection
            },
            'stress_parameters': {
                'gdp_shock': gdp_shock,
                'unemployment_shock': unemployment_shock,
                'pce_shock': pce_shock,
                'duration': duration
            },
            'model_info': {
                'unemployment_arima': {
                    'aic': float(model_fit.aic),
                    'bic': float(model_fit.bic),
                    'order': (1, 1, 1)
                }
            }
        }
        
        # For GDP and consumption, use VAR model (since ARIMA is specifically for unemployment)
        try:
            from statsmodels.tsa.vector_ar.var_model import VAR
            
            # Prepare VAR data
            var_data = df[['GDP', 'Personal_Consumption_Expenditure', 'Unemployment_Rate']].copy()
            df_diff = var_data.diff().dropna()
            
            if len(df_diff) >= 12:
                # Fit VAR model
                model = VAR(df_diff)
                var_results = model.fit(maxlags=4, ic="aic")
                lag_order = var_results.k_ar
                
                # Generate VAR forecast
                forecast_diff = var_results.forecast(df_diff.values[-lag_order:], steps=duration)
                
                # Reconstruct levels
                last_values = var_data.iloc[-1]
                forecast_diff_df = pd.DataFrame(forecast_diff, columns=var_data.columns)
                forecast_levels = forecast_diff_df.cumsum() + last_values.values
                
                # Extract GDP and PCE forecasts
                gdp_forecast = forecast_levels['GDP'].values
                consumption_forecast = forecast_levels['Personal_Consumption_Expenditure'].values
                
                # Apply stress shocks
                stressed_gdp = gdp_forecast * (1 + gdp_shock)
                stressed_consumption = consumption_forecast * (1 + pce_shock)
                
                chart_data['baseline']['gdp'] = gdp_forecast.tolist()
                chart_data['baseline']['consumption'] = consumption_forecast.tolist()
                chart_data['stressed']['gdp'] = stressed_gdp.tolist()
                chart_data['stressed']['consumption'] = stressed_consumption.tolist()
                
                # Add VAR model info
                chart_data['model_info']['var_model'] = {
                    'aic': float(var_results.aic),
                    'bic': float(var_results.bic),
                    'optimal_lags': int(lag_order)
                }
                
            else:
                # Fallback to simple trend if VAR fails
                last_gdp = df['GDP'].iloc[-1]
                last_consumption = df['Personal_Consumption_Expenditure'].iloc[-1]
                gdp_trend = df['GDP'].diff().mean()
                consumption_trend = df['Personal_Consumption_Expenditure'].diff().mean()
                
                for i in range(duration):
                    baseline_gdp = last_gdp + gdp_trend * (i + 1)
                    baseline_consumption = last_consumption + consumption_trend * (i + 1)
                    stressed_gdp = baseline_gdp * (1 + gdp_shock)
                    stressed_consumption = baseline_consumption * (1 + pce_shock)
                    
                    chart_data['baseline']['gdp'].append(float(baseline_gdp))
                    chart_data['baseline']['consumption'].append(float(baseline_consumption))
                    chart_data['stressed']['gdp'].append(float(stressed_gdp))
                    chart_data['stressed']['consumption'].append(float(stressed_consumption))
                    
        except Exception as e:
            print(f"VAR model failed, using simple trend: {str(e)}")
            # Fallback to simple trend
            last_gdp = df['GDP'].iloc[-1]
            last_consumption = df['Personal_Consumption_Expenditure'].iloc[-1]
            gdp_trend = df['GDP'].diff().mean()
            consumption_trend = df['Personal_Consumption_Expenditure'].diff().mean()
            
            for i in range(duration):
                baseline_gdp = last_gdp + gdp_trend * (i + 1)
                baseline_consumption = last_consumption + consumption_trend * (i + 1)
                stressed_gdp = baseline_gdp * (1 + gdp_shock)
                stressed_consumption = baseline_consumption * (1 + pce_shock)
                
                chart_data['baseline']['gdp'].append(float(baseline_gdp))
                chart_data['baseline']['consumption'].append(float(baseline_consumption))
                chart_data['stressed']['gdp'].append(float(stressed_gdp))
                chart_data['stressed']['consumption'].append(float(stressed_consumption))
        
        return jsonify({
            'success': True,
            'chart_data': chart_data,
            'message': 'Hybrid model stress testing: ARIMA for unemployment, VAR for GDP & PCE'
        })
        
    except Exception as e:
        import traceback
        print(f"Error in ARIMA stress testing chart data: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({'success': False, 'error': f'ARIMA stress testing error: {str(e)}'}), 500

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
        duration = stress_params.get('duration', 6)  # Default 6 months (same as forecasting)
        
        # Validate duration (max 36 months)
        duration = max(3, min(36, duration))
        
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
        
        # Calculate EWS Risk Score with VAR forecasting
        # Generate VAR forecast for EWS analysis
        try:
            import pandas as pd
            from statsmodels.tsa.vector_ar.var_model import VAR
            
            # Convert data to DataFrame for VAR
            df = pd.DataFrame(data)
            df['observation_date'] = pd.to_datetime(df['observation_date'])
            df.set_index('observation_date', inplace=True)
            
            # Select variables for VAR model
            var_data = df[['GDP', 'Personal_Consumption_Expenditure', 'Unemployment_Rate']].copy()
            
            # Fit VAR model
            var_model = fit_var_model(var_data)
            if var_model is not None:
                # Generate baseline forecast
                baseline_forecast = generate_var_forecast(var_model, var_data, steps=12)
                ews_risk_data = calculate_ews_risk_score(data, baseline_forecast, stress_params)
                
                # Calculate stress scenario warning signals
                stress_ews_data = calculate_stress_scenario_warnings(data, baseline_forecast, stress_params)
                ews_risk_data['stress_warnings'] = stress_ews_data.get('warnings', {})
                ews_risk_data['stress_warning_dates'] = stress_ews_data.get('warning_dates', {})
            else:
                ews_risk_data = calculate_ews_risk_score(data, None, stress_params)
                ews_risk_data['stress_warnings'] = {}
                ews_risk_data['stress_warning_dates'] = {}
        except Exception as e:
            print(f"Error in VAR forecasting for EWS: {e}")
            ews_risk_data = calculate_ews_risk_score(data, None, stress_params)
            ews_risk_data['stress_warnings'] = {}
            ews_risk_data['stress_warning_dates'] = {}
        
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
            'ews_risk_score': ews_risk_data.get('risk_score', 0),
            'ews_risk_level': ews_risk_data.get('risk_level', 'Unknown'),
            'ews_warnings': ews_risk_data.get('warnings', {}),
            'ews_forecast_warnings': ews_risk_data.get('forecast_warnings', {}),
            'ews_warning_dates': ews_risk_data.get('warning_dates', {}),
            'ews_forecast_warning_dates': ews_risk_data.get('forecast_warning_dates', {}),
            'ews_stress_warnings': ews_risk_data.get('stress_warnings', {}),
            'ews_stress_warning_dates': ews_risk_data.get('stress_warning_dates', {}),
            'ews_thresholds': ews_risk_data.get('thresholds', {}),
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
                'gdp': -0.08,  # -8% as decimal
                'unemployment': 0.04,  # +4% as decimal
                'pce': -0.06,  # -6% as decimal
                'duration': 18
            },
            'risk_level': 'High',
            'trend_consistency': 'Maintains baseline economic trends while applying recession shocks'
        },
        'market_crash': {
            'name': 'Market Crash',
            'description': 'Sudden market volatility and rapid economic contraction',
            'parameters': {
                'gdp': -0.12,  # -12% as decimal
                'unemployment': 0.06,  # +6% as decimal
                'pce': -0.08,  # -8% as decimal
                'duration': 24
            },
            'risk_level': 'Critical',
            'trend_consistency': 'Applies severe shocks while preserving baseline economic trajectory patterns'
        },
        'inflation_shock': {
            'name': 'Inflation Shock',
            'description': 'High inflation periods with rapid price increases',
            'parameters': {
                'gdp': -0.03,  # -3% as decimal
                'unemployment': 0.01,  # +1% as decimal
                'pce': -0.02,  # -2% as decimal
                'duration': 6
            },
            'risk_level': 'Medium',
            'trend_consistency': 'Applies inflation shocks while maintaining baseline economic growth patterns'
        },
        'global_crisis': {
            'name': 'Global Crisis',
            'description': 'International economic shocks and global financial instability',
            'parameters': {
                'gdp': -0.15,  # -15% as decimal
                'unemployment': 0.08,  # +8% as decimal
                'pce': -0.10,  # -10% as decimal
                'duration': 36
            },
            'risk_level': 'Critical',
            'trend_consistency': 'Applies severe global crisis shocks while preserving long-term economic trajectory patterns'
        }
    }
    
    # Calculate EWS for stressed scenarios (after results is created)
    stress_ews_data = {}
    for scenario_name, scenario_data in results['scenarios'].items():
        try:
            # Convert stressed data to the same format as historical data
            stressed_data = []
            for i, row in enumerate(scenario_data):
                stressed_data.append({
                    'observation_date': data[i]['observation_date'],
                    'GDP': row['GDP'],
                    'Personal_Consumption_Expenditure': row['Personal_Consumption_Expenditure'],
                    'Unemployment_Rate': row['Unemployment_Rate']
                })
            
            # Calculate EWS for this stressed scenario
            stress_ews = calculate_ews_risk_score(stressed_data, None, stress_params)
            stress_ews_data[scenario_name] = stress_ews
            
        except Exception as e:
            print(f"Error calculating EWS for scenario {scenario_name}: {e}")
            stress_ews_data[scenario_name] = {
                'risk_score': 0,
                'risk_level': 'Unknown',
                'warnings': {'gdp': 0, 'pce': 0, 'unemployment': 0},
                'warning_dates': {'GDP_growth': [], 'PCE_growth': [], 'Unemployment_change': []},
                'thresholds': {}
            }
    
    # Add stress EWS data to results
    results['stress_ews_data'] = stress_ews_data
    
    return jsonify({
        'success': True,
        'scenarios': scenarios
    })

@app.route('/api/stress-testing/chart-data', methods=['POST'])
def get_stress_testing_chart_data():
    """Get comprehensive chart data for stress testing with historical + VAR + stress scenarios"""
    try:
        from flask import request
        import numpy as np
        import pandas as pd
        
        # Get stress test parameters
        stress_params = request.get_json()
        gdp_shock = stress_params.get('gdp', -0.05)
        unemployment_shock = stress_params.get('unemployment', 0.02)
        pce_shock = stress_params.get('pce', -0.03)
        duration = stress_params.get('duration', 6)  # Same default as forecasting
        
        # Validate duration (max 36 months for VAR model)
        duration = max(3, min(36, duration))
        
        if len(data) < 24:
            return jsonify({'success': False, 'error': 'Insufficient data for VAR stress testing (minimum 24 observations required)'}), 400
        
        if duration > 36:
            return jsonify({'success': False, 'error': 'Duration cannot exceed 36 months for VAR model stability and accuracy. Longer forecasts may become unreliable due to model drift and uncertainty accumulation.'}), 400
        
        # Convert data to DataFrame (same as forecasting)
        df = pd.DataFrame(data)
        df['observation_date'] = pd.to_datetime(df['observation_date'])
        df.set_index('observation_date', inplace=True)
        
        # Select variables for VAR model (same as forecasting)
        var_data = df[['GDP', 'Personal_Consumption_Expenditure', 'Unemployment_Rate']].copy()
        
        # Fit VAR model (same logic as forecasting)
        var_model = fit_var_model(var_data)
        if var_model is None:
            return jsonify({'success': False, 'error': 'Failed to fit VAR model'}), 400
        
        # Generate baseline forecast (same as forecasting)
        baseline_forecast = generate_var_forecast(var_model, var_data, steps=duration)
        if baseline_forecast is None:
            return jsonify({'success': False, 'error': 'Failed to generate baseline forecast'}), 400
        
        # Apply stress scenarios to baseline forecast
        scenarios = {
            'custom': {
                'GDP': gdp_shock,
                'Unemployment_Rate': unemployment_shock,  # This will be applied as percentage change
                'Personal_Consumption_Expenditure': pce_shock
            }
        }
        
        stress_results = apply_stress_scenarios(baseline_forecast, scenarios)
        if stress_results is None:
            return jsonify({'success': False, 'error': 'Failed to apply stress scenarios'}), 400
        
        custom_stressed = stress_results['custom']
        
        # Helper function to safely convert values to JSON-serializable format
        def safe_convert_to_list(series):
            """Convert pandas series to list, handling NaN and inf values"""
            import numpy as np
            values = []
            for val in series:
                if np.isnan(val) or np.isinf(val):
                    values.append(0.0)
                else:
                    values.append(float(val))
            return values
        
        # Prepare chart data (same format as forecasting)
        chart_data = {
            'historical': {
                'dates': df.index[-24:].strftime('%Y-%m-%d').tolist(),
                'gdp': safe_convert_to_list(df['GDP'].iloc[-24:]),
                'unemployment': safe_convert_to_list(df['Unemployment_Rate'].iloc[-24:]),
                'consumption': safe_convert_to_list(df['Personal_Consumption_Expenditure'].iloc[-24:])
            },
            'baseline': {
                'dates': baseline_forecast.index.strftime('%Y-%m-%d').tolist(),
                'gdp': safe_convert_to_list(baseline_forecast['GDP']),
                'unemployment': safe_convert_to_list(baseline_forecast['Unemployment_Rate']),
                'consumption': safe_convert_to_list(baseline_forecast['Personal_Consumption_Expenditure'])
            },
            'stressed': {
                'dates': custom_stressed.index.strftime('%Y-%m-%d').tolist(),
                'gdp': safe_convert_to_list(custom_stressed['GDP']),
                'unemployment': safe_convert_to_list(custom_stressed['Unemployment_Rate']),
                'consumption': safe_convert_to_list(custom_stressed['Personal_Consumption_Expenditure'])
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
            'chart_data': chart_data
        })
        
    except Exception as e:
        print(f"Error in comprehensive chart data: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/stress-testing/ews-system', methods=['POST'])
def ews_system_analysis():
    """EWS (Early Warning System) analysis based on notebook methodology"""
    try:
        from flask import request
        import pandas as pd
        import numpy as np
        from sklearn.preprocessing import StandardScaler
        
        # Get stress test parameters
        stress_params = request.get_json()
        gdp_shock = stress_params.get('gdp', -0.05)
        unemployment_shock = stress_params.get('unemployment', 0.02)
        pce_shock = stress_params.get('pce', -0.03)
        duration = stress_params.get('duration', 6)
        
        # Validate duration
        duration = max(3, min(36, duration))
        
        if len(data) < 24:
            return jsonify({'success': False, 'error': 'Insufficient data for EWS analysis (minimum 24 observations required)'}), 400
        
        # Convert data to DataFrame
        df = pd.DataFrame(data)
        df['observation_date'] = pd.to_datetime(df['observation_date'])
        df.set_index('observation_date', inplace=True)
        
        # Select variables for analysis
        COLS = ['GDP', 'Personal_Consumption_Expenditure', 'Unemployment_Rate']
        var_data = df[COLS].copy()
        
        # EWS Parameters
        K = 0.4  # threshold coefficient (mean ± K*std)
        
        # Define stress scenarios (from notebook)
        scenarios = {
            "Mild Shock": {"GDP": -0.02, "Personal_Consumption_Expenditure": -0.01, "Unemployment_Rate": 0.01},
            "Moderate Shock": {"GDP": -0.05, "Personal_Consumption_Expenditure": -0.03, "Unemployment_Rate": 0.03},
            "Severe Shock": {"GDP": -0.1, "Personal_Consumption_Expenditure": -0.07, "Unemployment_Rate": 0.07},
            "Custom Shock": {"GDP": gdp_shock, "Personal_Consumption_Expenditure": pce_shock, "Unemployment_Rate": unemployment_shock}
        }
        
        # Generate VAR forecast
        var_model = fit_var_model(var_data)
        if var_model is None:
            return jsonify({'success': False, 'error': 'Failed to fit VAR model'}), 400
        
        baseline_forecast = generate_var_forecast(var_model, var_data, steps=duration)
        if baseline_forecast is None:
            return jsonify({'success': False, 'error': 'Failed to generate baseline forecast'}), 400
        
        # Create stress paths
        apply_from = 0  # Apply shocks to entire forecast horizon
        stress_paths = {}
        for scen, shocks in scenarios.items():
            x = baseline_forecast.copy()
            if apply_from > 0:
                mask = np.arange(len(x)) >= apply_from
            else:
                mask = slice(None)
            for col, shock in shocks.items():
                if col in x.columns:
                    x.loc[x.index[mask], col] = x.loc[x.index[mask], col] * (1 + shock)
            stress_paths[scen] = x
        
        # Generate indicators from levels (from notebook)
        def indicators_from_levels(df_levels):
            out = pd.DataFrame(index=df_levels.index)
            out["GDP_growth"] = df_levels["GDP"].pct_change()
            out["PCE_growth"] = df_levels["Personal_Consumption_Expenditure"].pct_change()
            out["Unemployment_change"] = df_levels["Unemployment_Rate"].diff()
            return out.dropna()
        
        # Historical indicators (thresholds learned from this)
        ind_hist = indicators_from_levels(var_data)
        
        # Learn thresholds from historical data (from notebook)
        scaler = StandardScaler()
        cols_ind = ["GDP_growth", "PCE_growth", "Unemployment_change"]
        Xs_hist = scaler.fit_transform(ind_hist[cols_ind].values)
        s_hist = pd.DataFrame(Xs_hist, index=ind_hist.index, columns=cols_ind)
        
        thr_std = s_hist.mean() + K * s_hist.std()      # threshold in z-space
        mu = pd.Series(scaler.mean_, index=cols_ind)    # raw space mean
        sigma = pd.Series(scaler.scale_, index=cols_ind) # raw space std
        thr_raw_neg = mu - thr_std * sigma              # risk in growth = low tail
        thr_raw_pos = mu + thr_std * sigma              # risk in unemployment change = high tail
        
        # Compute signals function (from notebook)
        def compute_signals(full_levels):
            ind_all = indicators_from_levels(full_levels)
            Xs = scaler.transform(ind_all[cols_ind].values)
            s = pd.DataFrame(Xs, index=ind_all.index, columns=cols_ind)
            s["GDP_warning"] = (s["GDP_growth"] < -thr_std["GDP_growth"]).astype(int)
            s["PCE_warning"] = (s["PCE_growth"] < -thr_std["PCE_growth"]).astype(int)
            s["Unemp_warning"] = (s["Unemployment_change"] > thr_std["Unemployment_change"]).astype(int)
            s["EWS_score"] = s[["GDP_warning", "PCE_warning", "Unemp_warning"]].sum(axis=1)
            return s, ind_all
        
        # Analyze all scenarios
        results = {}
        
        # Baseline
        levels_baseline = pd.concat([var_data, baseline_forecast], axis=0)
        sig_base, ind_base = compute_signals(levels_baseline)
        results["Baseline"] = {"signals": sig_base, "ind": ind_base}
        
        # Stress scenarios
        for name, path in stress_paths.items():
            levels_path = pd.concat([var_data, path], axis=0)
            sig_scen, ind_scen = compute_signals(levels_path)
            results[name] = {"signals": sig_scen, "ind": ind_scen}
        
        # Prepare response data
        hist_end = var_data.index[-1]
        
        # Extract forecast period data for each scenario
        response_data = {}
        for name, pack in results.items():
            signals = pack["signals"]
            indicators = pack["ind"]
            
            # Get full data (historical + forecast) for EWS score chart
            forecast_signals = signals  # Use full signals data
            forecast_indicators = indicators  # Use full indicators data
            
            # Calculate summary statistics (forecast period only)
            forecast_period_signals = signals.loc[signals.index > hist_end]
            total_warnings = forecast_period_signals["EWS_score"].sum()
            gdp_warnings = forecast_period_signals["GDP_warning"].sum()
            pce_warnings = forecast_period_signals["PCE_warning"].sum()
            unemp_warnings = forecast_period_signals["Unemp_warning"].sum()
            
            # Get warning dates (forecast period only for stress test relevance)
            warning_dates = {
                "GDP_growth": forecast_period_signals[forecast_period_signals["GDP_warning"] == 1].index.strftime('%Y-%m-%d').tolist(),
                "PCE_growth": forecast_period_signals[forecast_period_signals["PCE_warning"] == 1].index.strftime('%Y-%m-%d').tolist(),
                "Unemployment_change": forecast_period_signals[forecast_period_signals["Unemp_warning"] == 1].index.strftime('%Y-%m-%d').tolist()
            }
            
            # Get EWS scores over time
            ews_scores = forecast_signals["EWS_score"].tolist()
            dates = forecast_signals.index.strftime('%Y-%m-%d').tolist()
            
            response_data[name] = {
                "total_warnings": int(total_warnings),
                "gdp_warnings": int(gdp_warnings),
                "pce_warnings": int(pce_warnings),
                "unemployment_warnings": int(unemp_warnings),
                "warning_dates": warning_dates,
                "ews_scores": ews_scores,
                "dates": dates,
                "max_ews_score": int(forecast_signals["EWS_score"].max()),
                "avg_ews_score": float(forecast_signals["EWS_score"].mean())
            }
        
        # Historical analysis
        hist_signals = results["Baseline"]["signals"].loc[results["Baseline"]["signals"].index <= hist_end]
        historical_analysis = {
            "total_warnings": int(hist_signals["EWS_score"].sum()),
            "gdp_warnings": int(hist_signals["GDP_warning"].sum()),
            "pce_warnings": int(hist_signals["PCE_warning"].sum()),
            "unemployment_warnings": int(hist_signals["Unemp_warning"].sum()),
            "warning_dates": {
                "GDP_growth": hist_signals[hist_signals["GDP_warning"] == 1].index.strftime('%Y-%m-%d').tolist(),
                "PCE_growth": hist_signals[hist_signals["PCE_warning"] == 1].index.strftime('%Y-%m-%d').tolist(),
                "Unemployment_change": hist_signals[hist_signals["Unemp_warning"] == 1].index.strftime('%Y-%m-%d').tolist()
            }
        }
        
        return jsonify({
            'success': True,
            'methodology': 'EWS (Early Warning System) - Notebook Implementation',
            'parameters': {
                'k_value': K,
                'duration': duration,
                'scenarios': list(scenarios.keys())
            },
            'thresholds': {
                'z_space': {k: round(v, 4) for k, v in thr_std.items()},
                'raw_units_negative': {k: round(v, 4) for k, v in thr_raw_neg.items()},
                'raw_units_positive': {k: round(v, 4) for k, v in thr_raw_pos.items()}
            },
            'scaler_params': {
                'mean': mu.tolist(),
                'scale': sigma.tolist()
            },
            'historical_analysis': historical_analysis,
            'scenario_analysis': response_data,
            'forecast_start_date': hist_end.strftime('%Y-%m-%d')
        })
        
    except Exception as e:
        import traceback
        print(f"Error in EWS system analysis: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({'success': False, 'error': f'EWS analysis error: {str(e)}'}), 500

@app.route('/api/stress-testing/macro-indicators', methods=['POST'])
def macro_indicators_stress_testing():
    """Enhanced stress testing with macro indicators analysis as requested"""
    try:
        from flask import request
        import pandas as pd
        import numpy as np
        from sklearn.preprocessing import StandardScaler
        
        # Get stress test parameters
        stress_params = request.get_json()
        gdp_shock = stress_params.get('gdp', -0.05)
        unemployment_shock = stress_params.get('unemployment', 0.02)
        pce_shock = stress_params.get('pce', -0.03)
        duration = stress_params.get('duration', 6)
        
        # Validate duration
        duration = max(3, min(36, duration))
        
        if len(data) < 24:
            return jsonify({'success': False, 'error': 'Insufficient data for macro indicators analysis (minimum 24 observations required)'}), 400
        
        # Convert data to DataFrame
        df = pd.DataFrame(data)
        df['observation_date'] = pd.to_datetime(df['observation_date'])
        df.set_index('observation_date', inplace=True)
        
        # Select variables for analysis
        var_data = df[['GDP', 'Personal_Consumption_Expenditure', 'Unemployment_Rate']].copy()
        
        # Step 1: Prepare the data - Calculate changes from raw levels
        def calculate_indicators(levels_df):
            """Calculate percentage growth for GDP/consumption and differences for unemployment"""
            indicators = pd.DataFrame(index=levels_df.index)
            # Percentage growth for GDP and consumption
            indicators['GDP_growth'] = levels_df['GDP'].pct_change()
            indicators['PCE_growth'] = levels_df['Personal_Consumption_Expenditure'].pct_change()
            # Difference for unemployment rate
            indicators['Unemployment_change'] = levels_df['Unemployment_Rate'].diff()
            

            
            return indicators.dropna()
        
        # Calculate indicators for historical data
        hist_indicators = calculate_indicators(var_data)
        
        # Step 2: Standardize the indicators (mean=0, std=1)
        scaler = StandardScaler()
        z_hist = scaler.fit_transform(hist_indicators[['GDP_growth', 'PCE_growth', 'Unemployment_change']])
        z_hist_df = pd.DataFrame(z_hist, index=hist_indicators.index, 
                                columns=['GDP_growth', 'PCE_growth', 'Unemployment_change'])
        
        # Step 3: Set thresholds as mean + 1.5 * standard deviation
        K = 1.5  # As requested by user
        thresholds = {}
        for col in ['GDP_growth', 'PCE_growth', 'Unemployment_change']:
            # In z-space, mean=0 and std=1, so threshold = 0 + K * 1 = K
            thresholds[col] = K
        
        # Step 4: Generate warning signals for historical data
        def compute_warning_signals(indicators_df, scaler, thresholds):
            """Generate warning signals: 1 for GDP/consumption below threshold, 1 for unemployment above threshold"""
            # Transform to z-space
            z_scores = scaler.transform(indicators_df[['GDP_growth', 'PCE_growth', 'Unemployment_change']])
            z_df = pd.DataFrame(z_scores, index=indicators_df.index, 
                              columns=['GDP_growth', 'PCE_growth', 'Unemployment_change'])
            
            # Warning signals as requested
            # For GDP and PCE: warning when growth is below the negative threshold (severe negative growth)
            # For Unemployment: warning when change is above the positive threshold (large increase)
            gdp_warning = (z_df['GDP_growth'] < -thresholds['GDP_growth']).astype(int)
            pce_warning = (z_df['PCE_growth'] < -thresholds['PCE_growth']).astype(int)
            unemp_warning = (z_df['Unemployment_change'] > thresholds['Unemployment_change']).astype(int)
            

            
            return {
                'gdp_warning': gdp_warning,
                'pce_warning': pce_warning,
                'unemp_warning': unemp_warning,
                'z_scores': z_df
            }
        
        # Calculate historical warning signals
        hist_signals = compute_warning_signals(hist_indicators, scaler, thresholds)
        
        # Step 5: Apply to different scenarios
        
        # 5a. Historical data analysis
        historical_analysis = {
            'total_warnings': int(hist_signals['gdp_warning'].sum() + hist_signals['pce_warning'].sum() + hist_signals['unemp_warning'].sum()),
            'gdp_warnings': int(hist_signals['gdp_warning'].sum()),
            'pce_warnings': int(hist_signals['pce_warning'].sum()),
            'unemployment_warnings': int(hist_signals['unemp_warning'].sum()),
            'warning_dates': {
                'GDP_growth': [idx.strftime('%Y-%m-%d') for idx in hist_indicators.index[hist_signals['gdp_warning'] == 1]],
                'PCE_growth': [idx.strftime('%Y-%m-%d') for idx in hist_indicators.index[hist_signals['pce_warning'] == 1]],
                'Unemployment_change': [idx.strftime('%Y-%m-%d') for idx in hist_indicators.index[hist_signals['unemp_warning'] == 1]]
            }
        }
        
        # 5b. Forecasted values analysis (using VAR model)
        forecast_analysis = {}
        baseline_forecast = None  # Initialize for use in stress analysis
        
        if len(var_data) >= 24:
            try:
                # Fit VAR model
                var_model = fit_var_model(var_data)
                if var_model is not None:
                    # Generate baseline forecast
                    baseline_forecast = generate_var_forecast(var_model, var_data, steps=duration)
                    if baseline_forecast is not None:
                        # Calculate indicators for forecast data
                        forecast_indicators = calculate_indicators(baseline_forecast)
                        
                        if len(forecast_indicators) > 0:
                            # Apply same scaler and thresholds to forecast data
                            forecast_signals = compute_warning_signals(forecast_indicators, scaler, thresholds)
                            
                            forecast_analysis = {
                                'total_warnings': int(forecast_signals['gdp_warning'].sum() + forecast_signals['pce_warning'].sum() + forecast_signals['unemp_warning'].sum()),
                                'gdp_warnings': int(forecast_signals['gdp_warning'].sum()),
                                'pce_warnings': int(forecast_signals['pce_warning'].sum()),
                                'unemployment_warnings': int(forecast_signals['unemp_warning'].sum()),
                                'warning_dates': {
                                    'GDP_growth': [idx.strftime('%Y-%m-%d') for idx in forecast_indicators.index[forecast_signals['gdp_warning'] == 1]],
                                    'PCE_growth': [idx.strftime('%Y-%m-%d') for idx in forecast_indicators.index[forecast_signals['pce_warning'] == 1]],
                                    'Unemployment_change': [idx.strftime('%Y-%m-%d') for idx in forecast_indicators.index[forecast_signals['unemp_warning'] == 1]]
                                }
                            }
                        else:
                            forecast_analysis = {'error': 'No forecast indicators calculated'}
                    else:
                        forecast_analysis = {'error': 'Failed to generate baseline forecast'}
                else:
                    forecast_analysis = {'error': 'Failed to fit VAR model'}
            except Exception as e:
                import traceback
                print(f"Error in forecast analysis: {e}")
                print(f"Traceback: {traceback.format_exc()}")
                forecast_analysis = {'error': str(e)}
        
        # 5c. Stress-test scenarios analysis
        stress_analysis = {}
        if baseline_forecast is not None:
            try:
                # Apply stress shocks to baseline forecast
                stress_forecast = baseline_forecast.copy()
                
                # Apply cumulative shocks to create more dramatic effects
                for i in range(len(stress_forecast)):
                    # Apply cumulative shocks (each period gets worse)
                    cumulative_gdp_shock = 1 + gdp_shock * (i + 1) / len(stress_forecast)
                    cumulative_pce_shock = 1 + pce_shock * (i + 1) / len(stress_forecast)
                    cumulative_unemp_shock = 1 + unemployment_shock * (i + 1) / len(stress_forecast)
                    
                    stress_forecast.iloc[i, stress_forecast.columns.get_loc('GDP')] *= cumulative_gdp_shock
                    stress_forecast.iloc[i, stress_forecast.columns.get_loc('Personal_Consumption_Expenditure')] *= cumulative_pce_shock
                    stress_forecast.iloc[i, stress_forecast.columns.get_loc('Unemployment_Rate')] *= cumulative_unemp_shock
                
                # Calculate indicators for stress scenario
                stress_indicators = calculate_indicators(stress_forecast)
                
                if len(stress_indicators) > 0:
                    # Apply same scaler and thresholds to stress data
                    stress_signals = compute_warning_signals(stress_indicators, scaler, thresholds)
                    
                    stress_analysis = {
                        'total_warnings': int(stress_signals['gdp_warning'].sum() + stress_signals['pce_warning'].sum() + stress_signals['unemp_warning'].sum()),
                        'gdp_warnings': int(stress_signals['gdp_warning'].sum()),
                        'pce_warnings': int(stress_signals['pce_warning'].sum()),
                        'unemployment_warnings': int(stress_signals['unemp_warning'].sum()),
                        'warning_dates': {
                            'GDP_growth': [idx.strftime('%Y-%m-%d') for idx in stress_indicators.index[stress_signals['gdp_warning'] == 1]],
                            'PCE_growth': [idx.strftime('%Y-%m-%d') for idx in stress_indicators.index[stress_signals['pce_warning'] == 1]],
                            'Unemployment_change': [idx.strftime('%Y-%m-%d') for idx in stress_indicators.index[stress_signals['unemp_warning'] == 1]]
                        }
                    }
                else:
                    stress_analysis = {'error': 'No stress indicators calculated'}
            except Exception as e:
                import traceback
                print(f"Error in stress analysis: {e}")
                print(f"Traceback: {traceback.format_exc()}")
                stress_analysis = {'error': str(e)}
        else:
            stress_analysis = {'error': 'No baseline forecast available for stress analysis'}
        
        # Convert thresholds to raw units for interpretation
        mu = scaler.mean_
        sigma = scaler.scale_
        raw_thresholds = {}
        for i, col in enumerate(['GDP_growth', 'PCE_growth', 'Unemployment_change']):
            if col in ['GDP_growth', 'PCE_growth']:
                # Lower tail risk (negative growth)
                raw_thresholds[f"{col}_neg"] = mu[i] - thresholds[col] * sigma[i]
            else:
                # Upper tail risk (positive unemployment change)
                raw_thresholds[f"{col}_pos"] = mu[i] + thresholds[col] * sigma[i]
        
        # Calculate overall risk metrics
        total_historical_warnings = historical_analysis['total_warnings']
        risk_score = min(10, total_historical_warnings / len(hist_indicators) * 100)
        
        if risk_score < 3:
            risk_level = "Low"
        elif risk_score < 6:
            risk_level = "Medium"
        elif risk_score < 8:
            risk_level = "High"
        else:
            risk_level = "Critical"
        
        return jsonify({
            'success': True,
            'methodology': 'Macro Indicators Stress Testing with StandardScaler + 1.5×Std Dev Thresholds',
            'parameters': {
                'gdp_shock': gdp_shock,
                'unemployment_shock': unemployment_shock,
                'pce_shock': pce_shock,
                'duration': duration,
                'k_value': K
            },
            'thresholds': {
                'z_space': {k: round(v, 4) for k, v in thresholds.items()},
                'raw_units': {k: round(v, 4) for k, v in raw_thresholds.items()}
            },
            'scaler_params': {
                'mean': mu.tolist(),
                'scale': sigma.tolist()
            },
            'historical_analysis': historical_analysis,
            'forecast_analysis': forecast_analysis,
            'stress_analysis': stress_analysis,
            'overall_risk': {
                'risk_score': round(risk_score, 2),
                'risk_level': risk_level,
                'total_observations': len(hist_indicators)
            }
        })
        
    except Exception as e:
        import traceback
        print(f"Error in macro indicators stress testing: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({'success': False, 'error': f'Macro indicators analysis error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
