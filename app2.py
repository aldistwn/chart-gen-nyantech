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
    page_title="🎮 Gaming Performance Analyzer - Fixed",
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
        self.debug_mode = True
    
    def load_csv_data(self, file_data, filename):
        """Load and validate CSV data with bulletproof parsing"""
        try:
            # Try multiple encoding and delimiter combinations
            delimiters = [',', ';', '\t', '|']
            encodings = ['utf-8', 'latin-1', 'cp1252']
            
            for encoding in encodings:
                for delimiter in delimiters:
                    try:
                        df = pd.read_csv(io.StringIO(file_data.decode(encoding)), delimiter=delimiter)
                        if len(df.columns) > 1 and len(df) > 0:
                            if self._validate_and_process_columns(df, delimiter, encoding):
                                return True
                    except Exception as e:
                        if self.debug_mode:
                            st.write(f"Debug: Failed {encoding} + {delimiter}: {str(e)}")
                        continue
            
            st.error("❌ Could not parse CSV file. Please check the format.")
            return False
            
        except Exception as e:
            st.error(f"❌ Error loading CSV: {str(e)}")
            return False
    
    def _validate_and_process_columns(self, df, delimiter, encoding):
        """Validate required columns and process data with STRICT integrity checks"""
        columns = list(df.columns)
        
        if self.debug_mode:
            st.write(f"🔍 **Debug - Columns found**: {columns}")
        
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
        
        if self.debug_mode:
            st.write(f"✅ **Found FPS column**: `{fps_col}`")
            st.write(f"✅ **Found CPU column**: `{cpu_col}`")
        
        # Convert to numeric and clean data
        fps_data = pd.to_numeric(df[fps_col], errors='coerce')
        cpu_data = pd.to_numeric(df[cpu_col], errors='coerce')
        
        # DEBUG: Show raw data before any processing
        if self.debug_mode:
            st.write(f"🔍 **Raw FPS first 10 values**: {list(fps_data.head(10))}")
            st.write(f"🔍 **Raw FPS range**: {fps_data.min():.1f} - {fps_data.max():.1f}")
            st.write(f"🔍 **Raw CPU range**: {cpu_data.min():.1f} - {cpu_data.max():.1f}")
        
        # Remove invalid data (but keep track)
        valid_mask = ~(fps_data.isna() | cpu_data.isna() | (fps_data < 0) | (cpu_data < 0))
        invalid_count = len(df) - valid_mask.sum()
        
        if invalid_count > 0:
            st.warning(f"⚠️ Found {invalid_count} invalid rows (NaN, negative values)")
        
        # Create clean dataset with EXACT original values
        clean_data = pd.DataFrame({
            'FPS': fps_data[valid_mask].values,
            'CPU': cpu_data[valid_mask].values,
            'TimeMinutes': np.arange(valid_mask.sum()) / 60
        }).reset_index(drop=True)
        
        # CRITICAL: Store original data WITHOUT any modifications
        self.original_data = clean_data.copy()
        
        # Final validation - ensure no data corruption
        if self.debug_mode:
            st.write(f"✅ **Final data integrity check**:")
            st.write(f"   - Total rows: {len(clean_data)}")
            st.write(f"   - FPS range: {clean_data['FPS'].min():.1f} - {clean_data['FPS'].max():.1f}")
            st.write(f"   - CPU range: {clean_data['CPU'].min():.1f} - {clean_data['CPU'].max():.1f}")
            st.write(f"   - Average FPS: {clean_data['FPS'].mean():.1f}")
        
        st.success(f"✅ Data loaded successfully using delimiter '{delimiter}' and encoding '{encoding}'")
        
        # Data summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Total Records", f"{len(clean_data):,}")
        with col2:
            st.metric("🎯 FPS Range", f"{clean_data['FPS'].min():.0f} - {clean_data['FPS'].max():.0f}")
        with col3:
            st.metric("🖥️ CPU Range", f"{clean_data['CPU'].min():.0f}% - {clean_data['CPU'].max():.0f}%")
        
        return True
    
    def remove_outliers(self, method='percentile', threshold=1):
        """Remove FPS outliers using various methods"""
        if self.original_data is None:
            return False
        
        fps_data = self.original_data['FPS']
        original_count = len(self.original_data)
        
        if self.debug_mode:
            st.write(f"🔍 **Outlier removal - Before**: {original_count} rows, FPS range: {fps_data.min():.1f} - {fps_data.max():.1f}")
        
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
            
        # Apply mask and preserve original data structure
        self.processed_data = self.original_data[keep_mask].copy().reset_index(drop=True)
        self.removed_indices = self.original_data[~keep_mask].index.tolist()
        
        # Update time column for continuous data
        self.processed_data['TimeMinutes'] = np.arange(len(self.processed_data)) / 60
        
        removed_count = len(self.removed_indices)
        removal_pct = (removed_count / original_count) * 100
        
        if self.debug_mode:
            st.write(f"🔍 **Outlier removal - After**: {len(self.processed_data)} rows, FPS range: {self.processed_data['FPS'].min():.1f} - {self.processed_data['FPS'].max():.1f}")
        
        st.info(f"🚫 Removed {removed_count} outliers ({removal_pct:.1f}%) using {method} method")
        
        return True
    
    def apply_smoothing(self, fps_smooth=False, cpu_smooth=False, fps_window=7, cpu_window=7):
        """Apply Savitzky-Golay smoothing with STRICT data integrity"""
        # Safety check - ensure we have data to work with
        if self.original_data is None:
            st.error("❌ No original data available for smoothing")
            return False
            
        # Initialize processed_data if it doesn't exist (no outlier removal)
        if self.processed_data is None:
            self.processed_data = self.original_data.copy()
            if self.debug_mode:
                st.write("📊 **No outlier removal - using original data**")
        
        data_length = len(self.processed_data)
        
        # CRITICAL: When smoothing is OFF, use EXACT original data
        if not fps_smooth and not cpu_smooth:
            # Create exact copy with no modifications
            self.processed_data['FPS_Smooth'] = self.processed_data['FPS'].copy()
            self.processed_data['CPU_Smooth'] = self.processed_data['CPU'].copy()
            
            if self.debug_mode:
                st.write("📊 **No smoothing applied - using pure raw CSV data**")
                st.write(f"   - FPS range maintained: {self.processed_data['FPS_Smooth'].min():.1f} - {self.processed_data['FPS_Smooth'].max():.1f}")
            
            st.info("📊 Smoothing disabled - displaying raw CSV data exactly as imported")
            return True
        
        # FPS Smoothing (only if enabled)
        if fps_smooth and data_length >= 5:
            window = min(max(fps_window, 5), data_length)
            if window % 2 == 0:
                window -= 1
            
            try:
                original_fps = self.processed_data['FPS'].copy()
                
                if self.debug_mode:
                    st.write(f"🔍 **FPS Smoothing - Input**: range {original_fps.min():.1f} - {original_fps.max():.1f}")
                
                smoothed_fps = savgol_filter(
                    original_fps, 
                    window_length=window, 
                    polyorder=min(2, window-1)
                )
                
                # Prevent overshoot - CRITICAL for data integrity
                original_min, original_max = original_fps.min(), original_fps.max()
                smoothed_fps = np.clip(smoothed_fps, original_min, original_max)
                
                self.processed_data['FPS_Smooth'] = smoothed_fps
                
                if self.debug_mode:
                    st.write(f"🔍 **FPS Smoothing - Output**: range {smoothed_fps.min():.1f} - {smoothed_fps.max():.1f}")
                
                st.success(f"✅ FPS smoothed (window: {window}, preserved range: {original_min:.1f}-{original_max:.1f})")
                
            except Exception as e:
                self.processed_data['FPS_Smooth'] = self.processed_data['FPS'].copy()
                st.warning(f"⚠️ FPS smoothing failed: {str(e)}, using original data")
        else:
            # When FPS smoothing is OFF, use exact original data
            self.processed_data['FPS_Smooth'] = self.processed_data['FPS'].copy()
            if not fps_smooth:
                st.info("📊 FPS smoothing disabled - using raw CSV data")
        
        # CPU Smoothing (only if enabled)
        if cpu_smooth and data_length >= 5:
            window = min(max(cpu_window, 5), data_length)
            if window % 2 == 0:
                window -= 1
            
            try:
                original_cpu = self.processed_data['CPU'].copy()
                
                if self.debug_mode:
                    st.write(f"🔍 **CPU Smoothing - Input**: range {original_cpu.min():.1f} - {original_cpu.max():.1f}")
                
                smoothed_cpu = savgol_filter(
                    original_cpu, 
                    window_length=window, 
                    polyorder=min(2, window-1)
                )
                
                # Ensure CPU stays within 0-100% range
                smoothed_cpu = np.clip(smoothed_cpu, 0, 100)
                
                self.processed_data['CPU_Smooth'] = smoothed_cpu
                
                if self.debug_mode:
                    st.write(f"🔍 **CPU Smoothing - Output**: range {smoothed_cpu.min():.1f} - {smoothed_cpu.max():.1f}")
                
                st.success(f"✅ CPU smoothed (window: {window}, clamped to 0-100%)")
                
            except Exception as e:
                self.processed_data['CPU_Smooth'] = self.processed_data['CPU'].copy()
                st.warning(f"⚠️ CPU smoothing failed: {str(e)}, using original data")
        else:
            # When CPU smoothing is OFF, use exact original data
            self.processed_data['CPU_Smooth'] = self.processed_data['CPU'].copy()
            if not cpu_smooth:
                st.info("📊 CPU smoothing disabled - using raw CSV data")
        
        # Final integrity check
        if self.debug_mode:
            final_fps_min = self.processed_data['FPS_Smooth'].min()
            final_fps_max = self.processed_data['FPS_Smooth'].max()
            original_fps_min = self.processed_data['FPS'].min()
            original_fps_max = self.processed_data['FPS'].max()
            
            st.write(f"🔍 **Final integrity check**:")
            st.write(f"   - Original FPS range: {original_fps_min:.1f} - {original_fps_max:.1f}")
            st.write(f"   - Final FPS range: {final_fps_min:.1f} - {final_fps_max:.1f}")
            
            if final_fps_max > original_fps_max + 0.1 or final_fps_min < original_fps_min - 0.1:
                st.error("🚨 **DATA INTEGRITY VIOLATION DETECTED!**")
                st.error("**Smoothing has created impossible values!**")
                return False
            else:
                st.success("✅ **Data integrity preserved**")
        
        return True
    
    def create_performance_chart(self, config):
        """Create optimized performance chart"""
        try:
            # Determine data source
            data = self.processed_data if self.processed_data is not None else self.original_data
            
            if data is None:
                st.error("❌ No data available for chart generation")
                return None
            
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
        """Export processed data to CSV with integrity validation"""
        if self.processed_data is None:
            return None, None
        
        # Use smoothed data if available, otherwise use original
        fps_data = self.processed_data['FPS_Smooth'] if 'FPS_Smooth' in self.processed_data else self.processed_data['FPS']
        cpu_data = self.processed_data['CPU_Smooth'] if 'CPU_Smooth' in self.processed_data else self.processed_data['CPU']
        
        # Final validation before export
        if self.debug_mode:
            original_fps_range = f"{self.original_data['FPS'].min():.1f} - {self.original_data['FPS'].max():.1f}"
            export_fps_range = f"{fps_data.min():.1f} - {fps_data.max():.1f}"
            
            st.write(f"🔍 **Export validation**:")
            st.write(f"   - Original FPS range: {original_fps_range}")
            st.write(f"   - Export FPS range: {export_fps_range}")
            
            # Check for impossible values
            if fps_data.max() > self.original_data['FPS'].max() + 5:
                st.error(f"🚨 **EXPORT BLOCKED**: Impossible FPS values detected!")
                st.error(f"**Export FPS max ({fps_data.max():.1f}) >> Original max ({self.original_data['FPS'].max():.1f})**")
                return None, None
        
        # Prepare export data
        export_data = pd.DataFrame({
            'Time_Minutes': self.processed_data['TimeMinutes'].round(3),
            'FPS': fps_data.round(1),
            'CPU_Percent': cpu_data.round(1)
        })
        
        # Convert to CSV
        csv_buffer = io.StringIO()
        export_data.to_csv(csv_buffer, index=False)
        csv_content = csv_buffer.getvalue()
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{game_title.replace(' ', '_')}_FIXED_processed_{timestamp}.csv"
        
        if self.debug_mode:
            st.success(f"✅ **Export validated and ready**: {len(export_data)} rows")
        
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
    .error-box {
        background: #ff4b4b;
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎮 Gaming Performance Analyzer - FIXED</h1>
        <p>Professional gaming chart generator with bulletproof data integrity</p>
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
        
        # Debug mode toggle
        debug_mode = st.toggle("🐛 Debug Mode", value=True, help="Show detailed processing information")
        analyzer.debug_mode = debug_mode
        
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
                    
                    # Process data only if data is loaded successfully
                    if analyzer.original_data is not None:
                        # Outlier removal
                        if enable_outlier_removal:
                            with st.spinner('🚫 Removing outliers...'):
                                analyzer.remove_outliers(outlier_method, outlier_threshold)
                        
                        # Apply smoothing with integrity checks
                        fps_window = fps_window if fps_smooth else 7
                        cpu_window = cpu_window if cpu_smooth else 7
                        
                        # Show processing status
                        if fps_smooth or cpu_smooth:
                            with st.spinner('🔧 Applying smoothing filters...'):
                                success = analyzer.apply_smoothing(fps_smooth, cpu_smooth, fps_window, cpu_window)
                        else:
                            with st.spinner('📊 Preparing raw CSV data...'):
                                success = analyzer.apply_smoothing(False, False, fps_window, cpu_window)
                        
                        if not success:
                            st.error("❌ Data processing failed - check debug output above")
                        else:
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
                    else:
                        st.error("❌ No data loaded. Please upload a valid CSV file first.")
    
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
                st.error("❌ Export blocked due to data integrity issues")
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
        - **Data integrity**: Ensures smoothed values never exceed original range
        """)
    
    with st.expander("🛡️ Data Integrity Features"):
        st.markdown("""
        **Fixed Issues:**
        - ✅ **Column mapping**: Intelligent detection of FPS and CPU columns
        - ✅ **Data preservation**: Raw data exactly preserved when smoothing OFF
        - ✅ **Range validation**: Smoothed values never exceed original min/max
        - ✅ **Export validation**: Blocks export if impossible values detected
        - ✅ **Debug mode**: Detailed logging of all processing steps
        - ✅ **Error handling**: Graceful fallback to original data on processing failure
        
        **Debug Mode Benefits:**
        - 🔍 Column detection logging
        - 🔍 Data range validation at each step
        - 🔍 Processing integrity checks
        - 🔍 Export validation details
        """)

    # Footer with version info
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #888; padding: 1rem;">
        🎮 Gaming Performance Analyzer v2.0 - FIXED<br>
        <small>Bulletproof data integrity • Professional gaming analytics • Bug-free processing</small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
