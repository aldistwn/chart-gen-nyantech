# 🎮 Gaming Performance Chart Generator

<div align="center">

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-v1.28+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

*Transform your gaming logs into professional performance charts with advanced smoothing filters*

[Demo](https://your-app-url.streamlit.app) • [Report Bug](https://github.com/yourusername/gaming-chart-generator/issues) • [Request Feature](https://github.com/yourusername/gaming-chart-generator/issues)

</div>

---

## 📖 Table of Contents

- [Features](#-features)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Usage](#-usage)
- [Data Format](#-supported-data-format)
- [Savitzky-Golay Filter](#-savitzky-golay-filter)
- [Screenshots](#-screenshots)
- [Contributing](#-contributing)
- [License](#-license)

## ✨ Features

### 📊 **Professional Chart Generation**
- High-resolution PNG export (1920x1080)
- Transparent background for overlays
- Customizable colors and styling
- Gaming-optimized visual design

### 🔧 **Advanced Smoothing**
- **Savitzky-Golay filters** with separate controls for FPS and CPU
- Independent toggle switches for each data type
- Real-time parameter adjustment
- Noise reduction while preserving curve shape

### 📈 **Performance Analysis**
- Automatic performance grading system
- Detailed statistics (min/max/average)
- Frame drop detection
- 60+ FPS time calculation

### 🎨 **Customization Options**
- Custom game titles and settings
- Color picker for FPS and CPU lines
- Display options (original vs smoothed data)
- Professional 3-line chart titles

### 📱 **Smart CSV Detection**
- Auto-detect various CSV formats (`,`, `;`, `\t`, `|`)
- Intelligent column recognition
- Support for different naming conventions
- Error handling and user guidance

## 🚀 Quick Start

### Online Demo
Try the app online: **[Gaming Chart Generator](https://your-app-url.streamlit.app)**

### Local Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/gaming-chart-generator.git
   cd gaming-chart-generator
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser**
   ```
   http://localhost:8501
   ```

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Dependencies
```bash
streamlit>=1.28.0
pandas>=2.0.0
matplotlib>=3.7.0
scipy>=1.11.0
numpy>=1.24.0
Pillow>=10.0.0
```

### Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv gaming-chart-env

# Activate (Windows)
gaming-chart-env\Scripts\activate

# Activate (macOS/Linux)
source gaming-chart-env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 📋 Usage

### 1. **Upload CSV File**
- Drag and drop your gaming log CSV file
- Supported formats: CSV with comma, semicolon, tab, or pipe separators

### 2. **Configure Chart**
- **Game Title**: Enter your game name
- **Graphics Settings**: Describe your game settings
- **Performance Mode**: Add performance mode info
- **Colors**: Customize FPS and CPU line colors

### 3. **Adjust Smoothing**
- **FPS Smoothing**: Toggle and configure FPS filter parameters
- **CPU Smoothing**: Toggle and configure CPU filter parameters
- **Window Size**: Control smoothing intensity (5-51)
- **Polynomial Order**: Adjust curve fitting (1-5)

### 4. **Export Results**
- Download high-resolution PNG charts
- Transparent background for video overlays
- Automatic filename with filter status

## 📊 Supported Data Format

Your CSV file should contain columns with:

### Required Columns
| Column Type | Detection Pattern | Example Names |
|-------------|-------------------|---------------|
| **FPS Data** | Contains 'fps' (case-insensitive) | `FPS`, `fps`, `Fps`, `Frame_FPS` |
| **CPU Usage** | Contains 'cpu' AND '%' | `CPU(%)`, `cpu%`, `CPU_Usage(%)` |

### Example CSV Structure
```csv
FPS,CPU(%),JANK,BigJANK,GPU(%)
60,45.2,0,0,78.5
58,48.1,1,0,82.1
62,42.8,0,0,75.3
59,51.2,0,1,79.8
61,44.9,0,0,77.2
```

### Sample Data
Download sample CSV files from the [`assets/`](assets/) folder to test the application.

## 🔧 Savitzky-Golay Filter

The application uses **Savitzky-Golay filters** for advanced data smoothing:

### What is Savitzky-Golay?
- **Noise reduction** without destroying curve features
- **Edge preservation** maintains peaks and valleys
- **Polynomial fitting** for smooth curves
- **Widely used** in scientific data analysis

### Filter Parameters

#### Window Size (5-51, odd numbers only)
- **Small (5-15)**: Minimal smoothing, preserves detail
- **Medium (17-31)**: Balanced smoothing (recommended)
- **Large (33-51)**: Heavy smoothing, removes most noise

#### Polynomial Order (1-5)
- **Order 1**: Linear fitting
- **Order 2-3**: Quadratic/cubic fitting (recommended)
- **Order 4-5**: Higher-order curves

### Independent Controls
- **🎯 FPS Smoothing**: Separate toggle and parameters for FPS data
- **🖥️ CPU Smoothing**: Separate toggle and parameters for CPU data
- **Mix and Match**: Use different settings or enable only one filter

### Recommended Settings
| Data Type | Window Size | Poly Order | Use Case |
|-----------|-------------|------------|----------|
| **Clean Data** | 11-21 | 2-3 | Light noise reduction |
| **Noisy Data** | 31-41 | 2 | Heavy noise reduction |
| **Preserve Details** | 7-15 | 3-4 | Maintain curve features |
| **Smooth Trends** | 21-31 | 2 | General gaming data |

## 📸 Screenshots

### Main Interface
![Main Interface](assets/ui_screenshot.png)

### Chart Examples
![Sample Chart](assets/demo_chart.png)

### Filter Controls
![Filter Controls](assets/filter_controls.png)

> **Note**: Add actual screenshots to the `assets/` folder

## 🎯 Use Cases

### Content Creators
- Create professional charts for YouTube videos
- Overlay performance data on gameplay footage
- Compare different game settings

### Gamers & Enthusiasts
- Monitor system performance
- Optimize game settings
- Track performance improvements

### Developers & Testers
- Analyze game performance
- Quality assurance testing
- Performance regression testing

## 🤝 Contributing

We welcome contributions! Here's how you can help:

### 🐛 Reporting Bugs
1. Check existing [issues](https://github.com/yourusername/gaming-chart-generator/issues)
2. Create a new issue with:
   - Clear description
   - Steps to reproduce
   - Expected vs actual behavior
   - Sample data (if applicable)

### 💡 Suggesting Features
1. Open an issue with the "enhancement" label
2. Describe your feature idea
3. Explain the use case

### 🔧 Contributing Code
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes
4. Add tests if applicable
5. Commit: `git commit -m 'Add amazing feature'`
6. Push: `git push origin feature/amazing-feature`
7. Open a Pull Request

### 📝 Development Setup
```bash
# Clone your fork
git clone https://github.com/yourusername/gaming-chart-generator.git

# Install development dependencies
pip install -r requirements.txt

# Run tests (if available)
python -m pytest tests/

# Run the app
streamlit run app.py
```

## 📋 Roadmap

- [ ] **v1.1**: GPU usage support
- [ ] **v1.2**: Multiple games comparison
- [ ] **v1.3**: Real-time monitoring
- [ ] **v1.4**: API for automation
- [ ] **v1.5**: Cloud storage integration

## 🔄 Changelog

### v1.0.0 (Current)
- ✅ Initial release
- ✅ Savitzky-Golay smoothing
- ✅ Independent FPS/CPU filters
- ✅ Professional chart generation
- ✅ High-resolution PNG export

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **[Streamlit](https://streamlit.io/)** - Web app framework
- **[SciPy](https://scipy.org/)** - Savitzky-Golay implementation
- **[Matplotlib](https://matplotlib.org/)** - Chart generation
- **[Pandas](https://pandas.pydata.org/)** - Data manipulation

## 📞 Support

- 📧 **Email**: your-email@example.com
- 💬 **Issues**: [GitHub Issues](https://github.com/yourusername/gaming-chart-generator/issues)
- 📖 **Docs**: [Wiki](https://github.com/yourusername/gaming-chart-generator/wiki)

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/gaming-chart-generator&type=Date)](https://star-history.com/#yourusername/gaming-chart-generator&Date)

---

<div align="center">

**Made with ❤️ for the gaming community**

[⬆ Back to Top](#-gaming-performance-chart-generator)

</div>
