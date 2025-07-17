import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import chardet

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
        self.debug_mode = True
        self.available_columns = []
        self.numeric_columns = []
    
    def load_csv_data(self, file_data, filename):
        """Load and validate CSV data with smart encoding detection"""
        try:
            # Deteksi encoding otomatis pakai chardet
            encoding_guess = chardet.detect(file_data)['encoding']
            if self.debug_mode:
                st.write(f"🔍 Detected encoding: {encoding_guess}")

            # Coba beberapa delimiter
            delimiters = [',', ';', '\t', '|']
            for delimiter in delimiters:
                try:
                    # Baca langsung dari BytesIO + encoding hasil deteksi
                    df = pd.read_csv(io.BytesIO(file_data), encoding=encoding_guess, delimiter=delimiter)

                    # Validasi kolom
                    if len(df.columns) > 1 and len(df) > 0:
                        if self._validate_and_process_columns(df, delimiter, encoding_guess):
                            return True
                except Exception as e:
                    if self.debug_mode:
                        st.write(f"Debug: Failed {encoding_guess} + {delimiter}: {str(e)}")
                    continue

            st.error("❌ Gagal baca CSV. Coba cek format/encoding file.")
            return False
        except Exception as e:
            st.error(f"❌ Error baca file: {str(e)}")
            return False
    
    def _validate_and_process_columns(self, df, delimiter, encoding):
        """Validate columns and process data dynamically"""
        self.available_columns = list(df.columns)
        
        if self.debug_mode:
            st.write(f"🔍 **Debug - All columns found**: {self.available_columns}")
        
        # Convert all columns to numeric where possible and identify numeric columns
        processed_df = pd.DataFrame()
        self.numeric_columns = []
        
        for col in self.available_columns:
            # Try to convert to numeric
            numeric_data = pd.to_numeric(df[col], errors='coerce')
            
            # Ganti semua nilai negatif ekstrem (kayak -1, -9999) jadi NaN → biar ga ganggu chart
            numeric_data = numeric_data.where(numeric_data >= 0)
            
            # Check if at least 50% of data is valid numeric
            valid_numeric_ratio = (~numeric_data.isna()).sum() / len(df)
            
            if valid_numeric_ratio >= 0.5 and not numeric_data.isna().all() and np.isfinite(numeric_data.max()):
                processed_df[col] = numeric_data
                self.numeric_columns.append(col)
                if self.debug_mode:
                    st.write(f"✅ **Numeric column**: `{col}` (Range: {numeric_data.min():.1f} - {numeric_data.max():.1f})")
            else:
                processed_df[col] = df[col]
                if self.debug_mode:
                    st.write(f"📝 **Text/invalid column**: `{col}` (Sample: {df[col].iloc[0] if len(df) > 0 else 'N/A'})")

                     
        if len(self.numeric_columns) < 1:
            st.error("❌ No numeric columns found for analysis")
            return False
        
        # Remove rows where ALL numeric columns are NaN
        numeric_data_only = processed_df[self.numeric_columns]
        valid_mask = ~numeric_data_only.isna().all(axis=1)
        
        # Also remove negative values for performance metrics
        for col in self.numeric_columns:
            valid_mask = valid_mask & (processed_df[col] >= 0)
        
        invalid_count = len(df) - valid_mask.sum()
        
        if invalid_count > 0:
            st.warning(f"⚠️ Found {invalid_count} invalid rows (NaN, negative values)")
        
        # Create clean dataset
        clean_data = processed_df[valid_mask].copy().reset_index(drop=True)
        
        # Add time column
        clean_data['TimeMinutes'] = np.arange(len(clean_data)) / 60
        
        self.original_data = clean_data
        
        if self.debug_mode:
            st.write(f"✅ **Final data check**:")
            st.write(f"   - Total rows: {len(clean_data)}")
            st.write(f"   - Numeric columns: {self.numeric_columns}")
            st.write(f"   - All columns: {list(clean_data.columns)}")
        
        st.success(f"✅ Data loaded successfully using delimiter '{delimiter}' and encoding '{encoding}'")
        
        return True
    
    def get_column_display_name(self, col_name):
        """Generate user-friendly display names for columns"""
        display_names = {
            'fps': 'FPS',
            'cpu': 'CPU Usage (%)',
            'gpu': 'GPU Usage (%)', 
            'ram': 'RAM Usage (%)',
            'memory': 'Memory Usage (%)',
            'temp': 'Temperature (°C)',
            'temperature': 'Temperature (°C)',
            'power': 'Power Consumption (W)',
            'battery': 'Battery (%)',
            'jank': 'Jank Count',
            'frame_drops': 'Frame Drops',
            'latency': 'Latency (ms)',
            'ping': 'Ping (ms)',
            'network': 'Network Usage',
            'bandwidth': 'Bandwidth',
            'thermal': 'Thermal State'
        }
        
        col_lower = col_name.lower()
        for key, display in display_names.items():
            if key in col_lower:
                return display
        
        # If no match found, return formatted version of original
        return col_name.replace('_', ' ').title()
    
    def get_column_color_suggestion(self, col_name):
        """Suggest colors based on column type"""
        color_map = {
            'fps': '#00D4FF',      # Cyan for FPS
            'cpu': '#FF6B35',      # Orange for CPU
            'gpu': '#4ECDC4',      # Teal for GPU
            'ram': '#45B7D1',      # Blue for RAM
            'memory': '#45B7D1',   # Blue for Memory
            'temp': '#FF4757',     # Red for Temperature
            'temperature': '#FF4757', # Red for Temperature
            'power': '#FFA726',    # Amber for Power
            'battery': '#66BB6A',  # Green for Battery
            'jank': '#8E24AA',     # Purple for Jank
            'frame_drops': '#D32F2F', # Dark Red for Frame Drops
            'latency': '#FF9800',  # Orange for Latency
            'ping': '#FF9800',     # Orange for Ping
            'network': '#9C27B0',  # Purple for Network
            'bandwidth': '#9C27B0', # Purple for Bandwidth
            'thermal': '#E91E63'   # Pink for Thermal
        }
        
        col_lower = col_name.lower()
        for key, color in color_map.items():
            if key in col_lower:
                return color
        
        # Default colors for unknown columns
        default_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8']
        return default_colors[hash(col_name) % len(default_colors)]
    
    def remove_outliers(self, method='percentile', threshold=1, target_columns=None):
        """Remove outliers from specified columns"""
        if self.original_data is None:
            return False
        
        if target_columns is None:
            target_columns = self.numeric_columns
        
        original_count = len(self.original_data)
        data = self.original_data.copy()
        
        if self.debug_mode:
            st.write(f"🔍 **Outlier removal - Before**: {original_count} rows")
            for col in target_columns:
                if col in data.columns:
                    col_data = data[col]
                    st.write(f"   - {col}: {col_data.min():.1f} to {col_data.max():.1f} (avg: {col_data.mean():.1f})")
        
        # Apply outlier removal to each specified column
        combined_mask = pd.Series([True] * len(data))
        
        for col in target_columns:
            if col not in data.columns or col not in self.numeric_columns:
                continue
                
            col_data = data[col]
            
            if method == 'percentile':
                percentile_threshold = np.percentile(col_data.dropna(), threshold)
                keep_mask = col_data >= percentile_threshold
                
            elif method == 'iqr':
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                keep_mask = (col_data >= lower_bound) & (col_data <= upper_bound)
                
            elif method == 'zscore':
                z_scores = np.abs((col_data - col_data.mean()) / col_data.std())
                keep_mask = z_scores <= threshold
            
            # Combine masks (AND operation - must pass all column filters)
            combined_mask = combined_mask & keep_mask
            
            if self.debug_mode:
                removed_by_this_col = (~keep_mask).sum()
                st.write(f"   - {col}: removed {removed_by_this_col} outliers")
        
        # Apply combined mask
        self.processed_data = data[combined_mask].copy().reset_index(drop=True)
        self.removed_indices = data[~combined_mask].index.tolist()
        
        # Update time column
        self.processed_data['TimeMinutes'] = np.arange(len(self.processed_data)) / 60
        
        removed_count = len(self.removed_indices)
        removal_pct = (removed_count / original_count) * 100
        
        if self.debug_mode:
            st.write(f"🔍 **Outlier removal - After**: {len(self.processed_data)} rows")
            for col in target_columns:
                if col in self.processed_data.columns:
                    col_data = self.processed_data[col]
                    st.write(f"   - {col}: {col_data.min():.1f} to {col_data.max():.1f} (avg: {col_data.mean():.1f})")
        
        st.success(f"✅ Removed {removed_count} outliers ({removal_pct:.1f}%) using {method} method on {len(target_columns)} columns")
        
        # Show before/after comparison for key metrics
        if any('fps' in col.lower() for col in target_columns):
            fps_cols = [col for col in target_columns if 'fps' in col.lower()]
            if fps_cols:
                fps_col = fps_cols[0]
                original_fps = data[fps_col]
                processed_fps = self.processed_data[fps_col]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("📊 Original FPS Range", f"{original_fps.min():.0f} - {original_fps.max():.0f}")
                with col2:
                    st.metric("🔧 Processed FPS Range", f"{processed_fps.min():.0f} - {processed_fps.max():.0f}")
        
        return True
    
    def create_performance_chart(self, config):
        """Create performance chart with dynamic axes"""
        try:
            data = self.processed_data if self.processed_data is not None else self.original_data
            
            if data is None:
                st.error("❌ No data available for chart generation")
                return None
            
            # Get selected columns for plotting
            selected_columns = [col for col in config['selected_columns'] if not config.get(f'hide_{col}', False)]
            
            if not selected_columns:
                st.error("❌ No columns selected for plotting")
                return None
            
            # Create figure with dark theme
            plt.style.use('dark_background')
            fig, ax1 = plt.subplots(figsize=(16, 9))
            fig.patch.set_facecolor('#0E1117')
            
            time_data = data['TimeMinutes']
            
            # Create multiple y-axes for different metrics
            axes = [ax1]
            colors_used = []
            
            for i, col in enumerate(selected_columns):
                if col not in data.columns or col not in self.numeric_columns:
                    continue
                
                # Use appropriate axis
                if i == 0:
                    ax = ax1
                else:
                    ax = ax1.twinx()
                    # Offset additional axes
                    if i > 1:
                        ax.spines['right'].set_position(('outward', 60 * (i-1)))
                    axes.append(ax)
                
                col_data = data[col]
                col_color = config.get(f'{col}_color', self.get_column_color_suggestion(col))
                col_display = self.get_column_display_name(col)
                
                # Plot the line
                line = ax.plot(time_data, col_data, 
                              color=col_color, linewidth=2.5, 
                              label=col_display, alpha=0.8, zorder=len(selected_columns)-i)
                
                # Configure axis
                ax.set_ylabel(col_display, fontsize=12, color=col_color, fontweight='bold')
                ax.tick_params(axis='y', colors='white', labelsize=10)
                
                # Set appropriate limits with 0 as minimum
                if np.isfinite(col_data.max()) and not col_data.dropna().empty:
                    data_max = col_data.max()
                    padding = data_max * 0.1
                    ax.set_ylim(0, data_max + padding)
                else:
                    st.warning(f"⚠️ Skipping `{col}` because it contains no valid data.")
                    continue  # skip plotting this column

                
                colors_used.append((col_display, col_color))
            
            # Configure primary axis
            ax1.set_xlabel('Time (minutes)', fontsize=12, color='white', fontweight='bold')
            ax1.tick_params(axis='x', colors='white', labelsize=10)
            ax1.grid(True, alpha=0.3, linestyle='--', color='gray')
            ax1.set_facecolor('#0E1117')
            
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
            if config['smartphone_name'] or colors_used:
                legend_elements = []
                
                if config['smartphone_name']:
                    legend_elements.append(plt.Line2D([0], [0], color='white', label=config['smartphone_name']))
                
                for col_display, col_color in colors_used:
                    legend_elements.append(plt.Line2D([0], [0], color='white', 
                                                    linewidth=2.5, label=col_display))
                
                legend = ax1.legend(handles=legend_elements, loc='upper right', 
                                  framealpha=0.9, fancybox=True, bbox_to_anchor=(1.0, 1.0))
                legend.get_frame().set_facecolor('#262730')
                for text in legend.get_texts():
                    text.set_color('white')
                    if config['smartphone_name'] and text.get_text() == config['smartphone_name']:
                        text.set_fontweight('bold')
            
            # Remove spines for cleaner look
            for ax in axes:
                ax.spines['top'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['left'].set_visible(False)
                if ax != ax1:
                    ax.spines['right'].set_color(colors_used[axes.index(ax)-1][1] if axes.index(ax)-1 < len(colors_used) else 'white')
                else:
                    ax.spines['right'].set_visible(False)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            st.error(f"❌ Chart generation failed: {str(e)}")
            return None
    
    def get_performance_stats(self, selected_columns=None):
        """Calculate performance statistics for selected columns"""
        if self.original_data is None:
            return {}
        
        data = self.processed_data if self.processed_data is not None else self.original_data
        
        if selected_columns is None:
            selected_columns = self.numeric_columns
        
        stats = {
            'duration': len(data) / 60,
            'total_frames': len(data),
            'removed_frames': len(self.removed_indices)
        }
        
        # Calculate stats for each selected column
        for col in selected_columns:
            if col not in data.columns or col not in self.numeric_columns:
                continue
                
            col_data = data[col]
            col_display = self.get_column_display_name(col)
            
            stats[f'{col}_avg'] = col_data.mean()
            stats[f'{col}_min'] = col_data.min()
            stats[f'{col}_max'] = col_data.max()
            stats[f'{col}_std'] = col_data.std()
            stats[f'{col}_display'] = col_display
            
            # Special calculations for FPS
            if 'fps' in col.lower():
                stats['fps_60_plus'] = (col_data >= 60).sum() / len(col_data) * 100
                stats['frame_drops'] = (col_data < 30).sum()
                
                # Performance grading based on FPS
                avg_fps = col_data.mean()
                if avg_fps >= 90:
                    stats['grade'] = "🏆 Excellent"
                    stats['grade_color'] = "green"
                elif avg_fps >= 60:
                    stats['grade'] = "✅ Good"
                    stats['grade_color'] = "blue"
                elif avg_fps >= 30:
                    stats['grade'] = "⚠️ Playable"
                    stats['grade_color'] = "orange"
                else:
                    stats['grade'] = "❌ Poor"
                    stats['grade_color'] = "red"
        
        return stats
    
    def create_shadow_table(self, selected_columns=None):
        """Create shadow table for video chart creation"""
        data = self.processed_data if self.processed_data is not None else self.original_data
        
        if data is None:
            return None, None
        
        if selected_columns is None:
            selected_columns = self.numeric_columns
        
        # Detect frame rate from data frequency
        total_frames = len(data)
        total_time_minutes = data['TimeMinutes'].max()
        
        if total_time_minutes > 0:
            frames_per_minute = total_frames / total_time_minutes
            estimated_fps = frames_per_minute / 60
            
            if estimated_fps <= 35:
                detected_fps = 30
            elif estimated_fps <= 65:
                detected_fps = 60
            elif estimated_fps <= 95:
                detected_fps = 90
            elif estimated_fps <= 125:
                detected_fps = 120
            else:
                detected_fps = int(estimated_fps)
        else:
            detected_fps = 60
        
        # Create shadow table with selected columns
        shadow_table = pd.DataFrame({
            'Frame': range(1, len(data) + 1),
            'Time': (data.index / detected_fps / 60).round(4)
        })
        
        # Add selected numeric columns
        for col in selected_columns:
            if col in data.columns and col in self.numeric_columns:
                shadow_table[col] = data[col].round(1)
        
        return shadow_table, detected_fps
    
    def export_processed_data(self, game_title, selected_columns=None):
        """Export processed data to CSV"""
        data = self.processed_data if self.processed_data is not None else self.original_data
        
        if data is None:
            return None, None
        
        if selected_columns is None:
            selected_columns = self.numeric_columns
        
        # Prepare export data
        export_data = pd.DataFrame({
            'Time_Minutes': data['TimeMinutes'].round(3)
        })
        
        # Add selected columns
        for col in selected_columns:
            if col in data.columns and col in self.numeric_columns:
                export_data[col.replace(' ', '_')] = data[col].round(1)
        
        # Convert to CSV
        csv_buffer = io.StringIO()
        export_data.to_csv(csv_buffer, index=False)
        csv_content = csv_buffer.getvalue()
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{game_title.replace(' ', '_')}_processed_{timestamp}.csv"
        
        if self.debug_mode:
            st.success(f"✅ **Export ready**: {len(export_data)} rows, {len(selected_columns)} metrics")
        
        return csv_content, filename

def main():
    # Custom CSS
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
    .column-selector {
        background: #1E1E1E;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #333;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎮 Gaming Performance Analyzer</h1>
        <p>Dynamic multi-metric gaming chart generator</p>
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
        
        st.header("🔧 Quick Data Processing")
        
        enable_outlier_removal = st.toggle("🚫 Remove Outliers", value=False)
        if enable_outlier_removal:
            outlier_method = st.selectbox("Method", ['percentile', 'iqr', 'zscore'])
            if outlier_method == 'percentile':
                outlier_threshold = st.slider("Bottom Percentile", 0.1, 5.0, 1.0, 0.1)
            elif outlier_method == 'zscore':
                outlier_threshold = st.slider("Z-Score Threshold", 1.0, 4.0, 2.0, 0.1)
            else:
                outlier_threshold = 1.5
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # File upload
        st.subheader("📁 Upload Performance Data")
        uploaded_file = st.file_uploader(
            "Upload your gaming log CSV file", 
            type=['csv'],
            help="CSV should contain numeric performance columns"
        )
        
        if uploaded_file is not None:
            # Load data
            file_data = uploaded_file.read()
            
            with st.spinner('📊 Loading and analyzing data structure...'):
                if analyzer.load_csv_data(file_data, uploaded_file.name):
                    
                    if analyzer.original_data is not None and analyzer.numeric_columns:
                        
                        # Dynamic Column Selection
                        st.subheader("📊 Select Metrics to Display")
                        
                        with st.container():
                            st.markdown('<div class="column-selector">', unsafe_allow_html=True)
                            
                            # Multi-select for columns
                            default_columns = []
                            for col in analyzer.numeric_columns[:4]:  # Select first 4 by default
                                default_columns.append(col)
                            
                            selected_columns = st.multiselect(
                                "Choose metrics to analyze:",
                                analyzer.numeric_columns,
                                default=default_columns,
                                help="Select one or more numeric columns to display on the chart"
                            )
                            
                            if selected_columns:
                                st.write(f"**Selected metrics:** {', '.join([analyzer.get_column_display_name(col) for col in selected_columns])}")
                                
                                # Color and visibility controls for each selected column
                                st.write("**Customize appearance:**")
                                
                                chart_config = {
                                    'game_title': game_title,
                                    'game_settings': game_settings,
                                    'game_mode': game_mode,
                                    'smartphone_name': smartphone_name,
                                    'selected_columns': selected_columns
                                }
                                
                                # Create columns for color pickers
                                color_cols = st.columns(min(len(selected_columns), 4))
                                
                                for i, col in enumerate(selected_columns):
                                    col_display = analyzer.get_column_display_name(col)
                                    suggested_color = analyzer.get_column_color_suggestion(col)
                                    
                                    with color_cols[i % 4]:
                                        chart_config[f'{col}_color'] = st.color_picker(
                                            f"{col_display[:10]}...", 
                                            suggested_color,
                                            key=f"color_{col}_{i}",
                                            help=f"Color for {col_display}"
                                        )
                                        chart_config[f'hide_{col}'] = st.checkbox(
                                            f"Hide {col_display[:15]}...",
                                            value=False,
                                            key=f"hide_{col}_{i}",
                                            help=f"Hide {col_display} from chart"
                                        )
                            
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        if selected_columns:
                            # Data Processing Options
                            with st.expander("🔧 Advanced Processing Options"):
                                enable_advanced_outlier = st.toggle("🚫 Advanced Outlier Removal", value=False, key="advanced_outlier")
                                if enable_advanced_outlier:
                                    advanced_outlier_method = st.selectbox("Advanced Method", ['percentile', 'iqr', 'zscore'], key="advanced_method")
                                    
                                    # Select columns for outlier removal
                                    outlier_columns = st.multiselect(
                                        "Apply outlier removal to:",
                                        selected_columns,
                                        default=selected_columns,
                                        help="Choose which metrics to apply outlier removal",
                                        key="outlier_columns_select"
                                    )
                                    
                                    if advanced_outlier_method == 'percentile':
                                        advanced_outlier_threshold = st.slider("Advanced Bottom Percentile", 0.1, 10.0, 2.0, 0.1, key="advanced_threshold")
                                    elif advanced_outlier_method == 'zscore':
                                        advanced_outlier_threshold = st.slider("Advanced Z-Score Threshold", 1.0, 4.0, 2.5, 0.1, key="advanced_zscore")
                                    else:
                                        advanced_outlier_threshold = 1.5
                            
                            # Process data
                            processed = False
                            
                            # Priority 1: Advanced outlier removal (if enabled)
                            if 'enable_advanced_outlier' in locals() and enable_advanced_outlier and 'outlier_columns' in locals():
                                with st.spinner('🔧 Applying advanced outlier removal...'):
                                    analyzer.remove_outliers(advanced_outlier_method, advanced_outlier_threshold, outlier_columns)
                                    processed = True
                            
                            # Priority 2: Quick outlier removal from sidebar (if enabled and no advanced processing)
                            elif enable_outlier_removal and not processed:
                                with st.spinner('🔧 Removing outliers from sidebar settings...'):
                                    analyzer.remove_outliers(outlier_method, outlier_threshold, selected_columns)
                                    processed = True
                            
                            # Default: Use raw data
                            if not processed:
                                analyzer.processed_data = analyzer.original_data.copy()
                                st.info("📊 **Raw Mode**: Using original data without processing")
                            
                            # Create chart
                            st.subheader("📊 Performance Chart")
                            
                            with st.spinner('🎨 Generating dynamic chart...'):
                                chart_fig = analyzer.create_performance_chart(chart_config)
                                
                                if chart_fig:
                                    st.pyplot(chart_fig, use_container_width=True)
                                else:
                                    st.error("Failed to generate chart")
                        else:
                            st.warning("⚠️ Please select at least one metric to display")
    
    with col2:
        # Statistics panel
        if uploaded_file is not None and analyzer.original_data is not None:
            st.subheader("📈 Performance Statistics")
            
            if 'selected_columns' in locals() and selected_columns:
                stats = analyzer.get_performance_stats(selected_columns)
                
                # Show grade if FPS is available
                if any('fps' in col.lower() for col in selected_columns) and 'grade' in stats:
                    st.markdown(f"""
                    <div style="text-align: center; padding: 1rem; background: linear-gradient(45deg, #667eea, #764ba2); 
                                border-radius: 10px; margin-bottom: 1rem;">
                        <h3 style="color: white; margin: 0;">{stats['grade']}</h3>
                        <p style="color: white; margin: 0;">Overall Performance</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Display stats for each selected column
                for i, col in enumerate(selected_columns):
                    if f'{col}_avg' in stats:
                        col_display = stats.get(f'{col}_display', analyzer.get_column_display_name(col))
                        
                        col1_stat, col2_stat = st.columns(2)
                        with col1_stat:
                            st.metric(f"📊 Avg {col_display[:8]}...", f"{stats[f'{col}_avg']:.1f}")
                            if f'{col}_min' in stats:
                                st.metric(f"🔻 Min {col_display[:8]}...", f"{stats[f'{col}_min']:.1f}")
                        
                        with col2_stat:
                            if f'{col}_max' in stats:
                                st.metric(f"🔺 Max {col_display[:8]}...", f"{stats[f'{col}_max']:.1f}")
                            if f'{col}_std' in stats:
                                st.metric(f"📏 Std {col_display[:8]}...", f"{stats[f'{col}_std']:.1f}")
                
                # Special FPS metrics
                if 'fps_60_plus' in stats:
                    st.metric("🎮 60+ FPS Time", f"{stats['fps_60_plus']:.1f}%")
                if 'frame_drops' in stats:
                    st.metric("⚠️ Frame Drops", f"{stats['frame_drops']}")
                
                # General metrics
                st.metric("⏱️ Duration", f"{stats['duration']:.1f} min")
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
                
                # Shadow table preview and export
                shadow_table, detected_fps = analyzer.create_shadow_table(selected_columns)
                if shadow_table is not None:
                    st.subheader("🎬 Video Chart Table")
                    st.info(f"📹 **Detected Frame Rate**: {detected_fps} FPS")
                    
                    # Show preview
                    with st.expander("👀 Preview Video Chart Data (First 10 rows)"):
                        st.dataframe(shadow_table.head(10), use_container_width=True)
                    
                    # Export shadow table
                    shadow_csv = shadow_table.to_csv(index=False)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    shadow_filename = f"{game_title.replace(' ', '_')}_video_chart_{timestamp}.csv"
                    
                    st.download_button(
                        label="🎬 Download Video Chart CSV",
                        data=shadow_csv,
                        file_name=shadow_filename,
                        mime="text/csv",
                        use_container_width=True,
                        help="Perfect for creating animated video charts with all selected metrics"
                    )
                
                # Regular data export
                csv_content, csv_filename = analyzer.export_processed_data(game_title, selected_columns)
                if csv_content:
                    st.download_button(
                        label="📄 Download Processed Data",
                        data=csv_content,
                        file_name=csv_filename,
                        mime="text/csv",
                        use_container_width=True
                    )
            else:
                st.info("📊 Select metrics from the main panel to see statistics")
        else:
            st.info("📤 Upload CSV file to see performance statistics")
    
    # Documentation
    with st.expander("📋 CSV Format & Dynamic Features"):
        st.markdown("""
        **🚀 NEW: Dynamic Column Detection**
        - **Auto-detects** all numeric columns in your CSV
        - **Multi-metric support**: FPS, CPU, GPU, RAM, Temperature, etc.
        - **Custom axis creation**: Each metric gets its own Y-axis
        - **Smart color suggestions**: Based on metric type
        
        **📊 Supported Metric Types:**
        - **Performance**: FPS, Frame Drops, Jank Count
        - **Hardware**: CPU(%), GPU(%), RAM(%), Memory(%)
        - **Thermal**: Temperature, Thermal State
        - **Power**: Battery(%), Power Consumption
        - **Network**: Latency, Ping, Bandwidth
        - **Custom**: Any numeric column
        
        **📁 Example CSV Formats:**
        ```
        FPS,CPU(%),GPU(%),Temp,RAM(%)
        60,45.2,67.8,42.1,58.3
        58,48.1,70.2,43.5,59.1
        ```
        
        **✨ Features:**
        - **Dynamic axis scaling**: Automatic Y-axis adjustment
        - **Color customization**: Per-metric color control
        - **Show/Hide toggles**: Control visibility per metric
        - **Multi-column outlier removal**: Apply to selected metrics only
        """)
    
    with st.expander("🎨 Chart Customization Guide"):
        st.markdown("""
        **🎯 Metric Selection:**
        - Choose **1-6 metrics** for optimal readability
        - **Primary metrics** (FPS, CPU) get prominent positioning
        - **Secondary metrics** get additional Y-axes
        
        **🌈 Color Strategy:**
        - **FPS**: Cyan/Blue (performance focus)
        - **CPU/GPU**: Orange/Red (thermal colors)
        - **RAM/Memory**: Blue shades (data colors)
        - **Temperature**: Red/Pink (heat colors)
        - **Battery**: Green (energy colors)
        
        **📊 Multi-Axis Layout:**
        - **Left axis**: Primary metric (usually FPS)
        - **Right axis**: Secondary metric (usually CPU)
        - **Additional axes**: Offset to the right
        - **Legend**: Shows all active metrics
        """)
    
    with st.expander("🎬 Video Chart Export"):
        st.markdown("""
        **📹 Enhanced Video Chart Features:**
        - **Multi-metric support**: All selected metrics in one CSV
        - **Frame-perfect timing**: Synced to detected FPS
        - **Ready for animation**: Frame-by-frame progression
        
        **🎥 Recommended Workflow:**
        1. **Select metrics** you want in your video
        2. **Customize colors** for video consistency
        3. **Download Video Chart CSV**
        4. **Import to video editor** (After Effects, Premiere Pro)
        5. **Create animated charts** with multiple metrics
        
        **📊 Video Chart Columns:**
        - **Frame**: Sequential numbering (1, 2, 3...)
        - **Time**: Frame converted to minutes
        - **[Metric columns]**: All selected metrics with values
        
        **🎨 Animation Ideas:**
        - **Racing bar charts**: Show metric competition over time
        - **Multi-line graphs**: Animated line progression
        - **Gauge animations**: Circular metric displays
        - **Heatmaps**: Color-coded performance zones
        """)
    
    with st.expander("🔧 Advanced Processing"):
        st.markdown("""
        **🚫 Outlier Removal Options:**
        - **Per-metric control**: Choose which metrics to clean
        - **Multiple methods**: Percentile, IQR, Z-Score
        - **Selective processing**: Keep some metrics raw
        
        **📊 Processing Strategies:**
        - **FPS cleaning**: Remove frame drops and stutters
        - **CPU/GPU smoothing**: Remove usage spikes
        - **Temperature filtering**: Remove sensor glitches
        - **Mixed approach**: Different methods per metric
        
        **⚡ Performance Tips:**
        - **Start simple**: 2-3 metrics for first analysis
        - **Color contrast**: Ensure visibility on dark background
        - **Axis scaling**: Check auto-scaling results
        - **Export timing**: Process before video creation
        """)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #888; padding: 1rem;">
        🎮 Gaming Performance Analyzer v4.0 - Dynamic Multi-Metric Edition<br>
        <small>📊 Dynamic Columns • 🎨 Multi-Axis Charts • 🎬 Video Export • 📈 Advanced Analytics</small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
