// Economic Overview JavaScript functionality

class EconomicOverview {
    constructor() {
        this.data = null;
        this.charts = {};
        this.init();
    }

    async init() {
        this.showLoading();
        this.setCurrentDate();
        await this.loadData();
        await this.loadSummary();
        await this.loadCharts();
        await this.loadAnalysis();
        await this.loadTable();
        this.hideLoading();
        this.setupEventListeners();
    }

    showLoading() {
        document.getElementById('loading-spinner').style.display = 'flex';
    }

    hideLoading() {
        document.getElementById('loading-spinner').style.display = 'none';
    }

    setCurrentDate() {
        const now = new Date();
        const options = { 
            year: 'numeric', 
            month: 'long', 
            day: 'numeric',
            weekday: 'long'
        };
        
        // For the old single date display (if it exists)
        const currentDateElement = document.getElementById('current-date');
        if (currentDateElement) {
            currentDateElement.textContent = now.toLocaleDateString('en-US', options);
        }
        
        // For the new split date display
        const currentDateTop = document.getElementById('current-date-top');
        const currentDateBottom = document.getElementById('current-date-bottom');
        
        if (currentDateTop && currentDateBottom) {
            const weekday = now.toLocaleDateString('en-US', { weekday: 'long' });
            const month = now.toLocaleDateString('en-US', { month: 'long' });
            const day = now.getDate();
            const year = now.getFullYear();
            
            currentDateTop.textContent = `${weekday}, ${month}`;
            currentDateBottom.textContent = `${day}, ${year}`;
        }
    }

    async loadData() {
        try {
            const response = await fetch('/api/data');
            const result = await response.json();
            if (result.success) {
                this.data = result.data;
            } else {
                throw new Error(result.error);
            }
        } catch (error) {
            console.error('Error loading data:', error);
            this.showError('Failed to load data');
        }
    }

    async loadSummary() {
        try {
            const response = await fetch('/api/summary');
            const result = await response.json();
            if (result.success) {
                this.updateSummaryCards(result.summary);
            }
        } catch (error) {
            console.error('Error loading summary:', error);
        }
    }

    updateSummaryCards(summary) {
        // GDP Card
        document.getElementById('gdp-value').textContent = this.formatCurrency(summary.gdp.current);
        const gdpChange = summary.gdp.change_percent;
        const gdpChangeElement = document.getElementById('gdp-change');
        gdpChangeElement.textContent = `${gdpChange >= 0 ? '+' : ''}${gdpChange.toFixed(2)}% from previous period`;
        gdpChangeElement.className = `text-muted ${gdpChange >= 0 ? 'change-positive' : 'change-negative'}`;

        // PCE Card
        document.getElementById('pce-value').textContent = this.formatCurrency(summary.pce.current);
        const pceChange = summary.pce.change_percent;
        const pceChangeElement = document.getElementById('pce-change');
        pceChangeElement.textContent = `${pceChange >= 0 ? '+' : ''}${pceChange.toFixed(2)}% from previous period`;
        pceChangeElement.className = `text-muted ${pceChange >= 0 ? 'change-positive' : 'change-negative'}`;

        // Unemployment Card
        document.getElementById('unemployment-value').textContent = `${summary.unemployment.current.toFixed(1)}%`;
        const unempChange = summary.unemployment.change_percent;
        const unempChangeElement = document.getElementById('unemployment-change');
        unempChangeElement.textContent = `${unempChange >= 0 ? '+' : ''}${unempChange.toFixed(2)}% from previous period`;
        unempChangeElement.className = `text-muted ${unempChange >= 0 ? 'change-negative' : 'change-positive'}`;
    }

    async loadCharts() {
        await Promise.all([
            this.loadCombinedChart(),
            this.loadGDPChart(),
            this.loadPCEChart(),
            this.loadUnemploymentChart()
        ]);
    }

    async loadCombinedChart() {
        try {
            const response = await fetch('/api/chart/combined');
            const chartData = await response.json();
            Plotly.newPlot('combined-chart', chartData.data, chartData.layout, {
                responsive: true,
                displayModeBar: false
            });
        } catch (error) {
            console.error('Error loading combined chart:', error);
        }
    }

    async loadGDPChart() {
        try {
            const response = await fetch('/api/chart/gdp');
            const chartData = await response.json();
            Plotly.newPlot('gdp-chart', chartData.data, chartData.layout, {
                responsive: true,
                displayModeBar: false
            });
        } catch (error) {
            console.error('Error loading GDP chart:', error);
        }
    }

    async loadPCEChart() {
        try {
            const response = await fetch('/api/chart/pce');
            const chartData = await response.json();
            Plotly.newPlot('pce-chart', chartData.data, chartData.layout, {
                responsive: true,
                displayModeBar: false
            });
        } catch (error) {
            console.error('Error loading PCE chart:', error);
        }
    }

    async loadUnemploymentChart() {
        try {
            const response = await fetch('/api/chart/unemployment');
            const chartData = await response.json();
            Plotly.newPlot('unemployment-chart', chartData.data, chartData.layout, {
                responsive: true,
                displayModeBar: false
            });
        } catch (error) {
            console.error('Error loading unemployment chart:', error);
        }
    }

    async loadAnalysis() {
        await Promise.all([
            this.loadCorrelationAnalysis(),
            this.loadTrendAnalysis()
        ]);
    }

    async loadCorrelationAnalysis() {
        try {
            const response = await fetch('/api/analysis/correlation');
            const result = await response.json();
            if (result.success) {
                this.displayCorrelationMatrix(result.correlation);
            }
        } catch (error) {
            console.error('Error loading correlation analysis:', error);
        }
    }

    displayCorrelationMatrix(correlation) {
        const container = document.getElementById('correlation-matrix');
        const metrics = ['GDP', 'Personal_Consumption_Expenditure', 'Unemployment_Rate'];
        const labels = ['GDP', 'PCE', 'Unemployment Rate'];
        
        let html = '<div class="row">';
        
        for (let i = 0; i < metrics.length; i++) {
            for (let j = i + 1; j < metrics.length; j++) {
                const value = correlation[metrics[i]][metrics[j]];
                const strength = this.getCorrelationStrength(value);
                
                html += `
                    <div class="col-12 mb-2">
                        <div class="correlation-item">
                            <strong>${labels[i]} ↔ ${labels[j]}</strong>
                            <span class="correlation-value ms-2">${value.toFixed(3)}</span>
                            <span class="trend-indicator ${strength.class} ms-2">${strength.label}</span>
                        </div>
                    </div>
                `;
            }
        }
        
        html += '</div>';
        container.innerHTML = html;
    }

    getCorrelationStrength(value) {
        const absValue = Math.abs(value);
        if (absValue >= 0.7) {
            return { label: 'Strong', class: 'trend-up' };
        } else if (absValue >= 0.3) {
            return { label: 'Moderate', class: 'trend-stable' };
        } else {
            return { label: 'Weak', class: 'trend-down' };
        }
    }

    async loadTrendAnalysis() {
        try {
            const response = await fetch('/api/analysis/trends');
            const result = await response.json();
            if (result.success) {
                this.displayTrendAnalysis(result.trends);
            }
        } catch (error) {
            console.error('Error loading trend analysis:', error);
        }
    }

    displayTrendAnalysis(trends) {
        const container = document.getElementById('trend-analysis');
        
        const html = `
            <div class="row">
                <div class="col-12 mb-3">
                    <div class="d-flex justify-content-between align-items-center">
                        <span><strong>GDP Trend:</strong></span>
                        <span class="trend-indicator ${trends.gdp_trend === 'increasing' ? 'trend-up' : 'trend-down'}">
                            ${trends.gdp_trend === 'increasing' ? '↗ Increasing' : '↘ Decreasing'}
                        </span>
                    </div>
                </div>
                <div class="col-12 mb-3">
                    <div class="d-flex justify-content-between align-items-center">
                        <span><strong>PCE Trend:</strong></span>
                        <span class="trend-indicator ${trends.pce_trend === 'increasing' ? 'trend-up' : 'trend-down'}">
                            ${trends.pce_trend === 'increasing' ? '↗ Increasing' : '↘ Decreasing'}
                        </span>
                    </div>
                </div>
                <div class="col-12 mb-3">
                    <div class="d-flex justify-content-between align-items-center">
                        <span><strong>Unemployment Trend:</strong></span>
                        <span class="trend-indicator ${trends.unemployment_trend === 'decreasing' ? 'trend-up' : 'trend-down'}">
                            ${trends.unemployment_trend === 'decreasing' ? '↘ Decreasing' : '↗ Increasing'}
                        </span>
                    </div>
                </div>
            </div>
        `;
        
        container.innerHTML = html;
    }

    async loadTable() {
        if (!this.data) return;
        
        const tableBody = document.getElementById('table-body');
        const recentData = this.data.slice(-20).reverse(); // Show last 20 entries
        
        let html = '';
        recentData.forEach(row => {
            const date = new Date(row.observation_date).toLocaleDateString('en-US', {
                year: 'numeric',
                month: 'short',
                day: 'numeric'
            });
            
            html += `
                <tr class="fade-in">
                    <td>${date}</td>
                    <td>${this.formatCurrency(row.GDP)}</td>
                    <td>${this.formatCurrency(row.Personal_Consumption_Expenditure)}</td>
                    <td>${row.Unemployment_Rate.toFixed(1)}%</td>
                </tr>
            `;
        });
        
        tableBody.innerHTML = html;
    }

    formatCurrency(value) {
        return new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: 'USD',
            minimumFractionDigits: 0,
            maximumFractionDigits: 0
        }).format(value);
    }

    showError(message) {
        // Create error notification
        const errorDiv = document.createElement('div');
        errorDiv.className = 'alert alert-danger alert-dismissible fade show position-fixed';
        errorDiv.style.cssText = 'top: 20px; right: 20px; z-index: 10000;';
        errorDiv.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        document.body.appendChild(errorDiv);
        
        // Auto remove after 5 seconds
        setTimeout(() => {
            if (errorDiv.parentNode) {
                errorDiv.parentNode.removeChild(errorDiv);
            }
        }, 5000);
    }

    setupEventListeners() {
        // Add refresh functionality
        const refreshButton = document.createElement('button');
        refreshButton.className = 'btn btn-custom position-fixed';
        refreshButton.style.cssText = 'bottom: 20px; right: 20px; z-index: 1000;';
        refreshButton.innerHTML = '<i class="fas fa-sync-alt"></i>';
        refreshButton.onclick = () => this.refreshData();
        document.body.appendChild(refreshButton);

        // Add smooth scrolling for better UX
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function (e) {
                e.preventDefault();
                const target = document.querySelector(this.getAttribute('href'));
                if (target) {
                    target.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            });
        });
    }

    async refreshData() {
        this.showLoading();
        await this.loadData();
        await this.loadSummary();
        await this.loadCharts();
        await this.loadAnalysis();
        await this.loadTable();
        this.hideLoading();
        
        // Show success message
        const successDiv = document.createElement('div');
        successDiv.className = 'alert alert-success alert-dismissible fade show position-fixed';
        successDiv.style.cssText = 'top: 20px; right: 20px; z-index: 10000;';
        successDiv.innerHTML = `
            Data refreshed successfully!
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        document.body.appendChild(successDiv);
        
        setTimeout(() => {
            if (successDiv.parentNode) {
                successDiv.parentNode.removeChild(successDiv);
            }
        }, 3000);
    }
}

// Initialize economic overview when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new EconomicOverview();
});

// Add window resize handler for responsive charts
window.addEventListener('resize', () => {
    // Trigger chart resize
    const chartIds = ['combined-chart', 'gdp-chart', 'pce-chart', 'unemployment-chart'];
    chartIds.forEach(id => {
        const element = document.getElementById(id);
        if (element && element.data) {
            Plotly.relayout(id, {
                width: element.offsetWidth,
                height: element.offsetHeight
            });
        }
    });
});
