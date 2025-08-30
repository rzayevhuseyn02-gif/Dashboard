# US Economic Dashboard

Bu dashboard ABŞ iqtisadi göstəricilərini (GDP, PCE, İşsizlik dərəcəsi) analiz edən və vizuallaşdıran web tətbiqidir.

## 📊 Xüsusiyyətlər

- **4 Əsas Səhifə:**
  - 🏠 **Dashboard** - Əsas görünüş və ümumi məlumatlar
  - 📈 **Analysis** - Təkmilləşdirilmiş analizlər (Seasonality, Volatility, Regression)
  - 🔮 **Forecasting** - Proqnozlaşdırma və trend analizi
  - 📊 **Statistics** - Detallı statistika və paylanma analizi

- **Analiz Növləri:**
  - Korrelyasiya analizi
  - Trend analizi
  - Seasonality (dövrilik) analizi
  - Volatility (dəyişkənlik) analizi
  - Regression analizi
  - Proqnozlaşdırma (6 aylıq)
  - Detallı statistika

- **Vizualizasiya:**
  - İnteraktiv qrafiklər (Plotly.js)
  - Box plots və histogramlar
  - Moving averages
  - Trend proyeksiyaları

## 🚀 Windows-da Quraşdırma və İstifadə

### 1. Python Quraşdırın
1. [Python.org](https://www.python.org/downloads/) saytından Python 3.8+ yükləyin
2. Quraşdırma zamanı "Add Python to PATH" seçimini işarələyin
3. Quraşdırmanı tamamlayın

### 2. Layihəni Yükləyin
1. Layihə fayllarını kompüterinizə köçürün
2. Command Prompt (cmd) açın
3. Layihə qovluğuna keçin:
```cmd
cd "C:\path\to\Dashboard Ph"
```

### 3. Virtual Environment Yaradın
```cmd
python -m venv venv
venv\Scripts\activate
```

### 4. Lazımi Paketləri Quraşdırın
```cmd
pip install -r requirements.txt
```

### 5. Tətbiqi Başladın
```cmd
python app_simple.py
```

### 6. Brauzerda Açın
Brauzerinizdə bu ünvana keçin: `http://localhost:5001`

## 📁 Fayl Strukturu

```
Dashboard Ph/
├── app_simple.py              # Əsas Flask tətbiqi
├── requirements.txt           # Python paketləri
├── README.md                 # Bu fayl
├── GDP (1).csv              # GDP məlumatları
├── personalconsumptionexpenditure.csv  # PCE məlumatları
├── unemploymentrate.csv      # İşsizlik məlumatları
├── templates/               # HTML şablonları
│   ├── index.html          # Əsas səhifə
│   ├── analysis.html       # Analiz səhifəsi
│   ├── forecasting.html    # Proqnoz səhifəsi
│   └── statistics.html     # Statistika səhifəsi
└── static/                 # CSS və JS faylları
    ├── css/
    │   └── style.css
    └── js/
        └── dashboard.js
```

## 🌐 API Endpoint-lər

### Məlumat Endpoint-ləri
- `GET /api/data` - Bütün iqtisadi məlumatlar
- `GET /api/summary` - Ümumi statistika
- `GET /api/statistics/detailed` - Detallı statistika

### Qrafik Endpoint-ləri
- `GET /api/chart/combined` - Kombinə edilmiş qrafik
- `GET /api/chart/gdp` - GDP qrafiki
- `GET /api/chart/pce` - PCE qrafiki
- `GET /api/chart/unemployment` - İşsizlik qrafiki

### Analiz Endpoint-ləri
- `GET /api/analysis/correlation` - Korrelyasiya analizi
- `GET /api/analysis/trends` - Trend analizi
- `GET /api/analysis/seasonality` - Seasonality analizi
- `GET /api/analysis/volatility` - Volatility analizi
- `GET /api/analysis/regression` - Regression analizi
- `GET /api/forecasting/simple` - Proqnozlaşdırma

## 🎯 Səhifələr

### 🏠 Dashboard (`/`)
- Əsas iqtisadi göstəricilər
- Kombinə edilmiş qrafiklər
- Korrelyasiya və trend analizi
- Export funksiyası

### 📈 Analysis (`/analysis`)
- **Seasonality Analysis** - Aylıq dövrilik analizi
- **Volatility Analysis** - Dəyişkənlik və risk ölçmələri
- **Regression Analysis** - Dəyişənlər arası əlaqə
- **Risk Metrics** - VaR, Max Drawdown, Sharpe Ratio

### 🔮 Forecasting (`/forecasting`)
- **GDP Forecasting** - 6 aylıq proqnoz
- **PCE Forecasting** - İstehlak xərcləri proqnozu
- **Unemployment Forecasting** - İşsizlik dərəcəsi proqnozu
- **Moving Averages** - Hərəkətli ortalama
- **Trend Projections** - Trend proyeksiyaları

### 📊 Statistics (`/statistics`)
- **Detailed Statistics** - Orta, median, standart kənarlaşma
- **Distribution Charts** - Paylanma qrafikləri
- **Box Plots** - Qutu diaqramları və outlier-lər
- **Percentile Analysis** - Percentil analizi

## 🛠️ Texnologiyalar

- **Backend:** Flask (Python)
- **Frontend:** HTML5, CSS3, JavaScript (ES6+)
- **UI Framework:** Bootstrap 5
- **Charts:** Plotly.js
- **Icons:** Font Awesome
- **Data Processing:** Python (csv module)

## 📋 Tələblər

- Python 3.8+
- Flask 3.1.2+
- Plotly 6.3.0+
- Modern web brauzer (Chrome, Firefox, Safari, Edge)

## 🔧 Xəta Həlli

### Port 5000/5001 istifadədədir
```cmd
# Əvvəlcə mövcud prosesi dayandırın
taskkill /f /im python.exe
# Sonra yenidən başladın
python app_simple.py
```

### Python tapılmır
- Python-un PATH-ə əlavə edildiyini yoxlayın
- Python quraşdırmanı yenidən edin

### Paket quraşdırma xətası
```cmd
pip install --upgrade pip
pip install -r requirements.txt
```

## 📞 Dəstək

Əgər problem yaşayırsınızsa:
1. README faylını diqqətlə oxuyun
2. Python və paketlərin düzgün quraşdırıldığını yoxlayın
3. Command Prompt-da xəta mesajlarını oxuyun

## 📄 Lisenziya

Bu layihə təhsil məqsədləri üçün yaradılmışdır.

---

**Qeyd:** Bu dashboard ABŞ Federal Reserve Economic Data (FRED) məlumatlarından istifadə edir və 2004-2024 dövrünü əhatə edir.
