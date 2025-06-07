# 🎮 Gaming Performance Chart Generator

Personal use only, made with AI
<div align="center">

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-v1.28+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

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

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **[Streamlit](https://streamlit.io/)** - Web app framework
- **[SciPy](https://scipy.org/)** - Savitzky-Golay implementation
- **[Matplotlib](https://matplotlib.org/)** - Chart generation
- **[Pandas](https://pandas.pydata.org/)** - Data manipulation

[⬆ Back to Top](#-gaming-performance-chart-generator)

</div>
