from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.utils
import json
from datetime import datetime
import os
import math

app = Flask(__name__)
CORS(app)

# Data loading and processing
def load_data():
    """Load and process economic data"""
    try:
        # Load CSV files
        gdp = pd.read_csv("GDP (1).csv")
        pce = pd.read_csv("personalconsumptionexpenditure.csv")
        unemployment = pd.read_csv("unemploymentrate.csv")
        
        # Convert dates
        for df in [gdp, pce, unemployment]:
            df["observation_date"] = pd.to_datetime(df["observation_date"])
            df.set_index("observation_date", inplace=True)
        
        # Resample GDP to monthly and merge data
        gdp_monthly = gdp.resample("MS").ffill()
        pce_monthly = pce.resample("MS").ffill()
        
        merged = gdp_monthly.join(pce_monthly, how="inner").join(unemployment, how="inner")
        merged = merged.rename(columns={
            "GDPC1": "GDP",
            "PCE": "Personal_Consumption_Expenditure",
            "UNRATE": "Unemployment_Rate"
        })
        
        return merged
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# Global data variable
data = load_data()

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/api/data')
def get_data():
    """Get all economic data"""
    if data is not None:
        return jsonify({
            'success': True,
            'data': data.reset_index().to_dict('records')
        })
    return jsonify({'success': False, 'error': 'Data not available'})

@app.route('/api/summary')
def get_summary():
    """Get summary statistics"""
    if data is not None:
        summary = {
            'gdp': {
                'current': float(data['GDP'].iloc[-1]),
                'change': float(data['GDP'].iloc[-1] - data['GDP'].iloc[-2]),
                'change_percent': float(((data['GDP'].iloc[-1] - data['GDP'].iloc[-2]) / data['GDP'].iloc[-2]) * 100)
            },
            'pce': {
                'current': float(data['Personal_Consumption_Expenditure'].iloc[-1]),
                'change': float(data['Personal_Consumption_Expenditure'].iloc[-1] - data['Personal_Consumption_Expenditure'].iloc[-2]),
                'change_percent': float(((data['Personal_Consumption_Expenditure'].iloc[-1] - data['Personal_Consumption_Expenditure'].iloc[-2]) / data['Personal_Consumption_Expenditure'].iloc[-2]) * 100)
            },
            'unemployment': {
                'current': float(data['Unemployment_Rate'].iloc[-1]),
                'change': float(data['Unemployment_Rate'].iloc[-1] - data['Unemployment_Rate'].iloc[-2]),
                'change_percent': float(((data['Unemployment_Rate'].iloc[-1] - data['Unemployment_Rate'].iloc[-2]) / data['Unemployment_Rate'].iloc[-2]) * 100)
            }
        }
        return jsonify({'success': True, 'summary': summary})
    return jsonify({'success': False, 'error': 'Data not available'})

@app.route('/api/chart/gdp')
def gdp_chart():
    """Generate GDP chart data"""
    if data is not None:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['GDP'],
            mode='lines+markers',
            name='GDP',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=4)
        ))
        fig.update_layout(
            title='US GDP Over Time (2004-2024)',
            xaxis_title='Date',
            yaxis_title='GDP (Billions of Dollars)',
            template='plotly_white',
            height=400
        )
        return jsonify(json.loads(fig.to_json()))
    return jsonify({'error': 'Data not available'})

@app.route('/api/chart/pce')
def pce_chart():
    """Generate PCE chart data"""
    if data is not None:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Personal_Consumption_Expenditure'],
            mode='lines+markers',
            name='Personal Consumption Expenditure',
            line=dict(color='#2ca02c', width=2),
            marker=dict(size=4)
        ))
        fig.update_layout(
            title='Personal Consumption Expenditure Over Time (2004-2024)',
            xaxis_title='Date',
            yaxis_title='PCE (Billions of Dollars)',
            template='plotly_white',
            height=400
        )
        return jsonify(json.loads(fig.to_json()))
    return jsonify({'error': 'Data not available'})

@app.route('/api/chart/unemployment')
def unemployment_chart():
    """Generate unemployment chart data"""
    if data is not None:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Unemployment_Rate'],
            mode='lines+markers',
            name='Unemployment Rate',
            line=dict(color='#d62728', width=2),
            marker=dict(size=4)
        ))
        fig.update_layout(
            title='Unemployment Rate Over Time (2004-2024)',
            xaxis_title='Date',
            yaxis_title='Unemployment Rate (%)',
            template='plotly_white',
            height=400
        )
        return jsonify(json.loads(fig.to_json()))
    return jsonify({'error': 'Data not available'})

@app.route('/api/chart/combined')
def combined_chart():
    """Generate combined chart with all metrics"""
    if data is not None:
        fig = go.Figure()
        
        # GDP
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['GDP'],
            mode='lines',
            name='GDP',
            line=dict(color='#1f77b4', width=2),
            yaxis='y'
        ))
        
        # PCE
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Personal_Consumption_Expenditure'],
            mode='lines',
            name='PCE',
            line=dict(color='#2ca02c', width=2),
            yaxis='y'
        ))
        
        # Unemployment (secondary y-axis)
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Unemployment_Rate'],
            mode='lines',
            name='Unemployment Rate',
            line=dict(color='#d62728', width=2),
            yaxis='y2'
        ))
        
        fig.update_layout(
            title='US Economic Indicators (2004-2024)',
            xaxis_title='Date',
            yaxis=dict(title='GDP & PCE (Billions of Dollars)', side='left'),
            yaxis2=dict(title='Unemployment Rate (%)', side='right', overlaying='y'),
            template='plotly_white',
            height=500,
            legend=dict(x=0.02, y=0.98)
        )
        return jsonify(json.loads(fig.to_json()))
    return jsonify({'error': 'Data not available'})

@app.route('/api/analysis/correlation')
def correlation_analysis():
    """Calculate correlation between economic indicators"""
    if data is not None:
        corr_matrix = data.corr()
        return jsonify({
            'success': True,
            'correlation': corr_matrix.to_dict()
        })
    return jsonify({'success': False, 'error': 'Data not available'})

@app.route('/api/analysis/trends')
def trend_analysis():
    """Analyze trends in the data"""
    if data is not None:
        # Calculate 12-month moving averages
        gdp_ma = data['GDP'].rolling(window=12).mean()
        pce_ma = data['Personal_Consumption_Expenditure'].rolling(window=12).mean()
        unemp_ma = data['Unemployment_Rate'].rolling(window=12).mean()
        
        trends = {
            'gdp_trend': 'increasing' if gdp_ma.iloc[-1] > gdp_ma.iloc[-13] else 'decreasing',
            'pce_trend': 'increasing' if pce_ma.iloc[-1] > pce_ma.iloc[-13] else 'decreasing',
            'unemployment_trend': 'decreasing' if unemp_ma.iloc[-1] < unemp_ma.iloc[-13] else 'increasing'
        }
        
        return jsonify({'success': True, 'trends': trends})
    return jsonify({'success': False, 'error': 'Data not available'})

@app.route('/statistics')
def statistics_page():
    """Detailed statistics page"""
    return render_template('statistics.html')

@app.route('/api/statistics/detailed')
def detailed_statistics():
    """Get detailed statistical analysis using the notebook logic"""
    if data is not None and len(data) > 0:
        try:
            # Get the data values
            gdp_values = data['GDP'].dropna().values
            pce_values = data['Personal_Consumption_Expenditure'].dropna().values
            unemp_values = data['Unemployment_Rate'].dropna().values
            
            def calculate_stats(values):
                """Calculate comprehensive statistics for a dataset"""
                n = len(values)
                if n == 0:
                    return None
                    
                mean = np.mean(values)
                median = np.median(values)
                std_dev = np.std(values, ddof=1)  # Sample standard deviation
                variance = np.var(values, ddof=1)  # Sample variance
                min_val = np.min(values)
                max_val = np.max(values)
                
                # Calculate percentiles
                p25 = np.percentile(values, 25)
                p75 = np.percentile(values, 75)
                p90 = np.percentile(values, 90)
                
                return {
                    'count': int(n),
                    'mean': float(mean),
                    'median': float(median),
                    'std_dev': float(std_dev),
                    'variance': float(variance),
                    'min': float(min_val),
                    'max': float(max_val),
                    'range': float(max_val - min_val),
                    'p25': float(p25),
                    'p75': float(p75),
                    'p90': float(p90),
                    'iqr': float(p75 - p25)
                }
            
            # Calculate statistics for each metric
            gdp_stats = calculate_stats(gdp_values)
            pce_stats = calculate_stats(pce_values)
            unemp_stats = calculate_stats(unemp_values)
            
            if gdp_stats and pce_stats and unemp_stats:
                stats = {
                    'gdp': gdp_stats,
                    'pce': pce_stats,
                    'unemployment': unemp_stats
                }
                return jsonify({'success': True, 'statistics': stats})
            else:
                return jsonify({'success': False, 'error': 'Unable to calculate statistics'})
                
        except Exception as e:
            print(f"Error calculating statistics: {e}")
            return jsonify({'success': False, 'error': f'Error calculating statistics: {str(e)}'})
    
    return jsonify({'success': False, 'error': 'Data not available'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
