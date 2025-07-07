import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io
from datetime import datetime
from scipy.signal import savgol_filter
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="🎮 Gaming Performance Analyzer",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

class GamingPerformanceAnalyzer:
    def __init__(self):
        self.original_data = None
        self.processed_data = None
        self.removed_indices = []
        self.processing_stats = {}
    
    @st.cache_data
    def load_csv_data(_self, file_data, filename):
        """Load and validate CSV data with intelligent parsing"""
        try:
            # Try multiple encoding and delimiter combinations
            delimiters = [',', ';', '\t', '|']
            encodings = ['utf-8', 'latin-1', 'cp1252']
            
            for encoding in encodings:
                for delimiter in delimiters:
                    try:
                        df = pd.read_csv(io.StringIO(file_data.decode(encoding)), delimiter=delimiter)
                        if len(df.columns) > 1 and len(df) > 0:
                            _self.original_data = df
                            success = _self._validate_and_process_columns(delimiter, encoding)
                            if success:
                                return True
                    except Exception as e:
                        continue
            
            st.error("❌ Could not parse CSV file. Please check the format.")
            return False
            
        except Exception as e:
            st.error(f"❌ Error loading CSV: {str(e)}")
            return False
    
    def _validate_and_process_columns(self, delimiter, encoding):
        """Validate required columns and process data"""
        columns = list(self.original_data.columns)
        
        # Find FPS column (flexible matching)
        fps_col = None
        for col in columns:
            if 'fps' in col.lower() or col.strip().upper() == 'FPS':
                fps_col = col
                break
        
        # Find CPU column (flexible matching) 
        cpu_col = None
        for col in columns:
            if 'cpu' in col.lower() and '%' in col:
                cpu_col = col
                break
        
        if not fps_col:
            st.error("❌ FPS column not found. Looking for columns containing 'FPS' or 'fps'")
            st.info(f"Available columns: {', '.join(columns)}")
            return False
            
        if not cpu_col:
            st.error("❌ CPU(%) column not found. Looking for columns containing 'cpu' and '%'")
            st.info(f"Available columns: {', '.join(columns)}")
            return False
        
        # Convert to numeric and clean data
        fps_data = pd.to_numeric(self.original_data[fps_col], errors='coerce')
        cpu_data = pd.to_numeric(self.original_data[cpu_col], errors='coerce')
        
        # Remove invalid data
        valid_mask = ~(fps_data.isna() | cpu_data.isna() | (fps_data <= 0) | (cpu_data < 0))
        valid_data = self.original_data[valid_mask].copy()
        
        # Standardize column names
        valid_data['FPS'] = fps_data[valid_mask]
        valid_data['CPU'] = cpu_data[valid_mask]
        valid_data['TimeMinutes'] = np.arange(len(valid_data)) / 60
        
        self.original_data = valid_data
        
        # Validation summary
        removed_rows = len(self.original_data) - len(valid_data)
        if removed_rows > 0:
            st.warning(f"⚠️ Removed {removed_rows} invalid rows")
        
        st.success(f"✅ Data loaded successfully using delimiter '{delimiter}' and encoding '{encoding}'")
        
        # Data summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Total Records", f"{len(valid_data):,}")
        with col2:
            st.metric("🎯 FPS Range", f"{fps_data.min():.0f} - {fps_data.max():.0f}")
        with col3:
            st.metric("🖥️ CPU Range", f"{cpu_data.min():.0f}% - {cpu_data.max():.0f}%")
        
        return True
    
    def remove_outliers(self, method='percentile', threshold=1):
        """Remove FPS outliers using various methods"""
        if self.original_data is None:
            return False
        
        fps_data = self.original_data['FPS']
        
        if method == 'percentile':
            # Remove bottom percentile
            percentile_threshold = np.percentile(fps_data, threshold)
            keep_mask = fps_data >= percentile_threshold
            
        elif method == 'iqr':
            # Interquartile range method
            Q1 = fps_data.quantile(0.25)
            Q3 = fps_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            keep_mask = fps_data >= lower_bound
            
        elif method == 'zscore':
            # Z-score method
            z_scores = np.abs((fps_data - fps_data.mean()) / fps_data.std())
            keep_mask = z_scores <= threshold
            
        # Apply mask
        self.processed_data = self.original_data[keep_mask].copy()
        self.removed_indices = self.original_data[~keep_mask].index.tolist()
        
        # Update time column
        self.processed_data['TimeMinutes'] = np.arange(len(self.processed_data)) / 60
        
        removed_count = len(self.removed_indices)
        removal_pct = (removed_count / len(self.original_data)) * 100
        
        st.info(f"🚫 Removed {removed_count} outliers ({removal_pct:.1f}%)")
        
        return True
    
    def apply_smoothing(self, fps_smooth=False, cpu_smooth=False, fps_window=7, cpu_window=7):
        """Apply Savitzky-Golay smoothing - when disabled, use pure original data"""
        # Safety check - ensure we have data to work with
        if self.original_data is None:
            st.error("❌ No original data available for smoothing")
            return False
            
        # Initialize processed_data if it doesn't exist
        if self.processed_data is None:
            self.processed_data = self.original_data.copy()
        
        data_length = len(self.processed_data)
        
        # FPS Smoothing
        if fps_smooth and data_length >= 5:
            window = min(max(fps_window, 5), data_length)
            if window % 2 == 0:
                window -= 1
            
            try:
                smoothed_fps = savgol_filter(
                    self.processed_data['FPS'], 
                    window_length=window, 
                    polyorder=min(2, window-1)
                )
                # Prevent overshoot
                original_min, original_max = self.processed_data['FPS'].min(), self.processed_data['FPS'].max()
                self.processed_data['FPS_Smooth'] = np.clip(smoothed_fps, original_min, original_max)
                st.success(f"✅ FPS smoothed (window: {window})")
            except Exception as e:
                self.processed_data['FPS_Smooth'] = self.processed_data['FPS'].copy()
                st.warning(f"⚠️ FPS smoothing failed: {str(e)}, using original data")
        else:
            # When smoothing is OFF, use pure original data (no processing)
            self.processed_data['FPS_Smooth'] = self.processed_data['FPS'].copy()
            if not fps_smooth:
                st.info("📊 FPS smoothing disabled - using raw CSV data")
        
        # CPU Smoothing
        if cpu_smooth and data_length >= 5:
            window = min(max(cpu_window, 5), data_length)
            if window % 2 == 0:
                window -= 1
            
            try:
                smoothed_cpu = savgol_filter(
                    self.processed_data['CPU'], 
                    window_length=window, 
                    polyorder=min(2, window-1)
                )
                self.processed_data['CPU_Smooth'] = np.clip(smoothed_cpu, 0, 100)
                st.success(f"✅ CPU smoothed (window: {window})")
            except Exception as e:
                self.processed_data['CPU_Smooth'] = self.processed_data['CPU'].copy()
                st.warning(f"⚠️ CPU smoothing failed: {str(e)}, using original data")
        else:
            # When smoothing is OFF, use pure original data (no processing)
            self.processed_data['CPU_Smooth'] = self.processed_data['CPU'].copy()
            if not cpu_smooth:
                st.info("📊 CPU smoothing disabled - using raw CSV data")
        
        return True
    
    def create_performance_chart(self, config):
        """Create optimized performance chart"""
        try:
            # Determine data source
            data = self.processed_data if self.processed_data is not None else self.original_data
            
            # Create figure with dark theme
            plt.style.use('dark_background')
            fig, ax1 = plt.subplots(figsize=(16, 9))
            fig.patch.set_facecolor('#0E1117')
            
            # Configure primary axis (FPS)
            ax1.set_xlabel('Time (minutes)', fontsize=12, color='white', fontweight='bold')
            ax1.set_ylabel('FPS', fontsize=12, color=config['fps_color'], fontweight='bold')
            ax1.tick_params(axis='both', colors='white', labelsize=10)
            
            # Configure secondary axis (CPU)
            ax2 = ax1.twinx()
            ax2.set_ylabel('CPU Usage (%)', fontsize=12, color=config['cpu_color'], fontweight='bold')
            ax2.tick_params(axis='y', colors='white', labelsize=10)
            ax2.set_ylim(0, 100)
            
            # Plot data
            time_data = data['TimeMinutes']
            
            if not config['hide_fps']:
                fps_data = data['FPS_Smooth'] if 'FPS_Smooth' in data else data['FPS']
                line1 = ax1.plot(time_data, fps_data, 
                               color=config['fps_color'], linewidth=2.5, 
                               label='FPS', alpha=0.9, zorder=3)
            
            if not config['hide_cpu']:
                cpu_data = data['CPU_Smooth'] if 'CPU_Smooth' in data else data['CPU']
                line2 = ax2.plot(time_data, cpu_data, 
                               color=config['cpu_color'], linewidth=2.0, 
                               label='CPU Usage', alpha=0.7, zorder=2)
            
            # Set FPS axis limits with padding
            if not config['hide_fps']:
                fps_max = fps_data.max() * 1.1
                ax1.set_ylim(0, fps_max)
            
            # Chart styling
            ax1.grid(True, alpha=0.3, linestyle='--', color='gray')
            ax1.set_facecolor('#0E1117')
            ax2.set_facecolor('#0E1117')
            
            # Title
            title_lines = [config['game_title']]
            if config['game_settings']:
                title_lines.append(config['game_settings'])
            if config['game_mode']:
                title_lines.append(config['game_mode'])
            
            plt.suptitle('\n'.join(title_lines), 
                        fontsize=20, fontweight='bold', 
                        color='white', y=0.95)
            
            # Legend
            if config['smartphone_name']:
                legend_elements = [plt.Line2D([0], [0], color='none', label=config['smartphone_name'])]
                
                if not config['hide_fps']:
                    legend_elements.append(plt.Line2D([0], [0], color=config['fps_color'], 
                                                    linewidth=2.5, label='FPS'))
                if not config['hide_cpu']:
                    legend_elements.append(plt.Line2D([0], [0], color=config['cpu_color'], 
                                                    linewidth=2, label='CPU Usage'))
                
                legend = ax1.legend(handles=legend_elements, loc='upper right', 
                                  framealpha=0.9, fancybox=True)
                legend.get_frame().set_facecolor('#262730')
                for text in legend.get_texts():
                    text.set_color('white')
                    if text.get_text() == config['smartphone_name']:
                        text.set_fontweight('bold')
            
            # Remove spines
            for spine in ax1.spines.values():
                spine.set_visible(False)
            for spine in ax2.spines.values():
                spine.set_visible(False)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            st.error(f"❌ Chart generation failed: {str(e)}")
            return None
    
    def get_performance_stats(self):
        """Calculate comprehensive performance statistics"""
        if self.original_data is None:
            return {}
        
        # Use processed data if available
        data = self.processed_data if self.processed_data is not None else self.original_data
        fps_data = data['FPS_Smooth'] if 'FPS_Smooth' in data else data['FPS']
        cpu_data = data['CPU_Smooth'] if 'CPU_Smooth' in data else data['CPU']
        
        # Performance calculations
        avg_fps = fps_data.mean()
        min_fps = fps_data.min()
        max_fps = fps_data.max()
        fps_std = fps_data.std()
        
        # Performance grading
        if avg_fps >= 90:
            grade = "🏆 Excellent"
            grade_color = "green"
        elif avg_fps >= 60:
            grade = "✅ Good"
            grade_color = "blue"
        elif avg_fps >= 30:
            grade = "⚠️ Playable"
            grade_color = "orange"
        else:
            grade = "❌ Poor"
            grade_color = "red"
        
        # Frame time analysis
        fps_60_plus = (fps_data >= 60).sum() / len(fps_data) * 100
        fps_30_minus = (fps_data < 30).sum()
        
        # CPU statistics
        avg_cpu = cpu_data.mean()
        max_cpu = cpu_data.max()
        
        return {
            'grade': grade,
            'grade_color': grade_color,
            'duration': len(data) / 60,
            'total_frames': len(data),
            'avg_fps': avg_fps,
            'min_fps': min_fps,
            'max_fps': max_fps,
            'fps_std': fps_std,
            'fps_60_plus': fps_60_plus,
            'frame_drops': fps_30_minus,
            'avg_cpu': avg_cpu,
            'max_cpu': max_cpu,
            'removed_frames': len(self.removed_indices)
        }
    
    def export_processed_data(self, game_title):
        """Export processed data to CSV"""
        if self.processed_data is None:
            return None, None
        
        # Prepare export data
        export_data = pd.DataFrame({
            'Time_Minutes': self.processed_data['TimeMinutes'].round(3),
            'FPS': (self.processed_data['FPS_Smooth'] if 'FPS_Smooth' in self.processed_data 
                   else self.processed_data['FPS']).round(1),
            'CPU_Percent': (self.processed_data['CPU_Smooth'] if 'CPU_Smooth' in self.processed_data 
                           else self.processed_data['CPU']).round(1)
        })
        
        # Convert to CSV
        csv_buffer = io.StringIO()
        export_data.to_csv(csv_buffer, index=False)
        csv_content = csv_buffer.getvalue()
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{game_title.replace(' ', '_')}_processed_{timestamp}.csv"
        
        return csv_content, filename

def main():
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #262730;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎮 Gaming Performance Analyzer</h1>
        <p>Professional gaming chart generator with advanced analytics</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize analyzer
    analyzer = GamingPerformanceAnalyzer()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("🎮 Game Configuration")
        
        game_title = st.text_input("🎯 Game Title", value="Mobile Legends: Bang Bang")
        game_settings = st.text_input("⚙️ Graphics Settings", value="Ultra - 120 FPS")
        game_mode = st.text_input("🚀 Performance Mode", value="Game Boost Mode")
        smartphone_name = st.text_input("📱 Device Model", value="iPhone 15 Pro Max")
        
        st.divider()
        
        st.header("🎨 Visual Settings")
        col1, col2 = st.columns(2)
        with col1:
            fps_color = st.color_picker("FPS Color", "#00D4FF")
        with col2:
            cpu_color = st.color_picker("CPU Color", "#FF6B35")
        
        hide_fps = st.checkbox("Hide FPS Line", value=False)
        hide_cpu = st.checkbox("Hide CPU Line", value=False)
        
        st.divider()
        
        st.header("🔧 Data Processing")
        
        # Outlier removal
        enable_outlier_removal = st.toggle("🚫 Remove Outliers", value=False)
        if enable_outlier_removal:
            outlier_method = st.selectbox("Method", ['percentile', 'iqr', 'zscore'])
            if outlier_method == 'percentile':
                outlier_threshold = st.slider("Bottom Percentile", 0.1, 5.0, 1.0, 0.1)
            elif outlier_method == 'zscore':
                outlier_threshold = st.slider("Z-Score Threshold", 1.0, 4.0, 2.0, 0.1)
            else:
                outlier_threshold = 1.5
        
        # Smoothing
        col1, col2 = st.columns(2)
        with col1:
            fps_smooth = st.toggle("🎯 FPS Smoothing", value=False)
            if fps_smooth:
                fps_window = st.slider("FPS Window", 3, 51, 7, step=2, 
                                     help="Larger window = smoother line")
        
        with col2:
            cpu_smooth = st.toggle("🖥️ CPU Smoothing", value=False)
            if cpu_smooth:
                cpu_window = st.slider("CPU Window", 3, 51, 7, step=2,
                                     help="Larger window = smoother line")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # File upload
        st.subheader("📁 Upload Performance Data")
        uploaded_file = st.file_uploader(
            "Upload your gaming log CSV file", 
            type=['csv'],
            help="CSV should contain FPS and CPU usage columns"
        )
        
        if uploaded_file is not None:
            # Load data
            file_data = uploaded_file.read()
            
            with st.spinner('📊 Loading and validating data...'):
                if analyzer.load_csv_data(file_data, uploaded_file.name):
                    
                    # Process data
                    if enable_outlier_removal:
                        with st.spinner('🚫 Removing outliers...'):
                            analyzer.remove_outliers(outlier_method, outlier_threshold)
                    
                    # Apply smoothing (only if data is loaded)
                    if analyzer.original_data is not None:
                        fps_window = fps_window if fps_smooth else 7
                        cpu_window = cpu_window if cpu_smooth else 7
                        
                        # Show processing status
                        if fps_smooth or cpu_smooth:
                            with st.spinner('🔧 Applying smoothing filters...'):
                                analyzer.apply_smoothing(fps_smooth, cpu_smooth, fps_window, cpu_window)
                        else:
                            with st.spinner('📊 Using raw CSV data...'):
                                analyzer.apply_smoothing(False, False, fps_window, cpu_window)
                    else:
                        st.error("❌ No data loaded. Please upload a valid CSV file first.")
                    
                    # Create chart
                    st.subheader("📊 Performance Chart")
                    
                    chart_config = {
                        'game_title': game_title,
                        'game_settings': game_settings,
                        'game_mode': game_mode,
                        'smartphone_name': smartphone_name,
                        'fps_color': fps_color,
                        'cpu_color': cpu_color,
                        'hide_fps': hide_fps,
                        'hide_cpu': hide_cpu
                    }
                    
                    with st.spinner('🎨 Generating chart...'):
                        chart_fig = analyzer.create_performance_chart(chart_config)
                        
                        if chart_fig:
                            st.pyplot(chart_fig, use_container_width=True)
                        else:
                            st.error("Failed to generate chart")
    
    with col2:
        # Statistics panel
        if uploaded_file is not None and analyzer.original_data is not None:
            st.subheader("📈 Performance Stats")
            
            stats = analyzer.get_performance_stats()
            
            # Performance grade
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem; background: linear-gradient(45deg, #667eea, #764ba2); 
                        border-radius: 10px; margin-bottom: 1rem;">
                <h3 style="color: white; margin: 0;">{stats['grade']}</h3>
                <p style="color: white; margin: 0;">Overall Performance</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Key metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("📊 Avg FPS", f"{stats['avg_fps']:.1f}")
                st.metric("⏱️ Duration", f"{stats['duration']:.1f} min")
                st.metric("🎯 Min FPS", f"{stats['min_fps']:.1f}")
            
            with col2:
                st.metric("🚀 Max FPS", f"{stats['max_fps']:.1f}")
                st.metric("🖥️ Avg CPU", f"{stats['avg_cpu']:.1f}%")
                st.metric("🔥 Max CPU", f"{stats['max_cpu']:.1f}%")
            
            # Advanced metrics
            st.metric("🎮 60+ FPS Time", f"{stats['fps_60_plus']:.1f}%")
            st.metric("⚠️ Frame Drops", f"{stats['frame_drops']}")
            
            if stats['removed_frames'] > 0:
                st.metric("🚫 Removed Frames", f"{stats['removed_frames']}")
            
            # Export section
            st.subheader("💾 Export Options")
            
            # Chart export
            if 'chart_fig' in locals() and chart_fig:
                img_buffer = io.BytesIO()
                chart_fig.savefig(img_buffer, format='png', dpi=300, 
                                bbox_inches='tight', facecolor='#0E1117')
                img_buffer.seek(0)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                png_filename = f"{game_title.replace(' ', '_')}_chart_{timestamp}.png"
                
                st.download_button(
                    label="📸 Download Chart",
                    data=img_buffer.getvalue(),
                    file_name=png_filename,
                    mime="image/png",
                    use_container_width=True
                )
            
            # Data export
            csv_content, csv_filename = analyzer.export_processed_data(game_title)
            if csv_content:
                st.download_button(
                    label="📄 Download Data",
                    data=csv_content,
                    file_name=csv_filename,
                    mime="text/csv",
                    use_container_width=True
                )
        else:
            st.info("📤 Upload CSV file to see performance statistics")
    
    # Help section
    with st.expander("📋 CSV Format Requirements"):
        st.markdown("""
        **Required Columns:**
        - **FPS**: Frame rate data (accepts variations like 'fps', 'Fps')
        - **CPU(%)**: CPU usage percentage (must contain 'cpu' and '%')
        
        **Example CSV:**
        ```
        FPS,CPU(%),JANK
        60,45.2,0
        58,48.1,1
        62,42.8,0
        ```
        
        **Supported Formats:**
        - Comma-separated (,)
        - Semicolon-separated (;)
        - Tab-separated
        - UTF-8, Latin-1, CP1252 encoding
        """)
    
    with st.expander("🔧 Processing Options"):
        st.markdown("""
        **Outlier Removal Methods:**
        - **Percentile**: Remove bottom X% of FPS values
        - **IQR**: Remove values below Q1 - 1.5*IQR
        - **Z-Score**: Remove values beyond Z standard deviations
        
        **Smoothing Filter:**
        - **Savitzky-Golay**: Preserves peaks while reducing noise
        - **Window Size**: 3-51 points (larger = smoother, smaller = more detail)
        - **When disabled**: Shows pure raw CSV data without any processing
        - **Applied separately** to FPS and CPU data
        """)

if __name__ == "__main__":
    main()
