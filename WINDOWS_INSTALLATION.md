# 🚀 Windows-da Dashboard Quraşdırma Təlimatları

## 📋 Lazım Olan Fayllar

Dostunuzdan bu faylları almalısınız:
- `Dashboard Ph` qovluğu (bütün fayllarla birlikdə)
- `requirements.txt` faylı
- `README.md` faylı

## ⚡ Sürətli Quraşdırma (5 dəqiqə)

### 1️⃣ Python Quraşdırın
1. [Python.org](https://www.python.org/downloads/) saytına keçin
2. "Download Python" düyməsinə basın
3. Quraşdırma zamanı **"Add Python to PATH"** seçimini işarələyin ✅
4. "Install Now" düyməsinə basın

### 2️⃣ Layihəni Açın
1. `Dashboard Ph` qovluğunu kompüterinizə köçürün
2. Qovluğa daxil olun
3. Adres sətrində `cmd` yazın və Enter basın

### 3️⃣ Virtual Environment Yaradın
Command Prompt-da bu əmrləri yazın:
```cmd
python -m venv venv
venv\Scripts\activate
```

### 4️⃣ Paketləri Quraşdırın
```cmd
pip install -r requirements.txt
```

### 5️⃣ Tətbiqi Başladın
```cmd
python app_simple.py
```

### 6️⃣ Brauzerda Açın
Brauzerinizdə bu ünvana keçin: `http://localhost:5001`
duzgun kod
## 🎉 Hazır! 

İndi dashboard işləyir və bu səhifələrə daxil ola bilərsiniz:
- **Əsas səhifə**: `http://localhost:5001`
- **Analiz**: `http://localhost:5001/analysis`
- **Proqnoz**: `http://localhost:5001/forecasting`
- **Statistika**: `http://localhost:5001/statistics`

## 🔧 Problem Həlli

### ❌ "Python is not recognized"
- Python quraşdırmanı yenidən edin
- "Add Python to PATH" seçimini işarələyin

### ❌ "Port already in use"
```cmd
taskkill /f /im python.exe
python app_simple.py
```

### ❌ "pip not found"
```cmd
python -m pip install -r requirements.txt
```

## 📞 Kömək Lazımdırsa

1. Command Prompt-da xəta mesajını kopyalayın
2. Dostunuza göndərin
3. Və ya README.md faylını oxuyun

## 🎯 Dashboard Xüsusiyyətləri

- 📊 **4 səhifə**: Dashboard, Analiz, Proqnoz, Statistika
- 📈 **İnteraktiv qrafiklər**: Hover edin, zoom edin
- 📥 **Export**: Məlumatları CSV faylı kimi yükləyin
- 📱 **Responsive**: Telefon və tabletdə də işləyir
- 🎨 **Modern dizayn**: Gözəl və professional görünüş

---

**Qeyd**: Bu dashboard ABŞ iqtisadi məlumatlarını (2004-2024) analiz edir.
