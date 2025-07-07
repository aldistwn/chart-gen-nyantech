import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io
from datetime import datetime
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
        self.debug_mode = True
        self.available_columns = {}
    
    def load_csv_data(self, file_data, filename):
        """Load and validate CSV data with multiple format support"""
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
        """Validate required columns and process data"""
        columns = list(df.columns)
        
        if self.debug_mode:
            st.write(f"🔍 **Debug - Columns found**: {columns}")
        
        # Find all available columns and their types
        available_columns = {}
        
        # Find FPS column
        fps_col = None
        for col in columns:
            if 'fps' in col.lower() or col.strip().upper() == 'FPS':
                fps_col = col
                available_columns['FPS'] = col
                break
        
        # Find CPU column
        cpu_col = None
        for col in columns:
            if 'cpu' in col.lower() and '%' in col:
                cpu_col = col
                available_columns['CPU'] = col
                break
        
        # Find other potential gaming metrics
        for col in columns:
            col_lower = col.lower()
            if 'gpu' in col_lower and '%' in col:
                available_columns['GPU'] = col
            elif 'ram' in col_lower or 'memory' in col_lower:
                available_columns['RAM'] = col
            elif 'temp' in col_lower:
                available_columns['Temperature'] = col
            elif 'battery' in col_lower:
                available_columns['Battery'] = col
            elif 'jank' in col_lower or 'stutter' in col_lower:
                available_columns['Jank'] = col
            elif 'ping' in col_lower or 'network' in col_lower or 'latency' in col_lower:
                available_columns['Network'] = col
        
        # Store available columns for toggle feature
        self.available_columns = available_columns
        
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
            if len(available_columns) > 2:
                st.write(f"🎯 **Additional columns found**: {list(available_columns.keys())}")
        
        # Convert to numeric and clean data
        clean_data_dict = {}
        
        for metric, col_name in available_columns.items():
            data = pd.to_numeric(df[col_name], errors='coerce')
            clean_data_dict[metric] = data
            
            if self.debug_mode:
                st.write(f"🔍 **{metric} range**: {data.min():.1f} - {data.max():.1f}")
        
        # Remove invalid data (but keep track)
        valid_mask = pd.Series(True, index=df.index)
        for metric, data in clean_data_dict.items():
            valid_mask &= ~(data.isna() | (data < 0))
        
        invalid_count = len(df) - valid_mask.sum()
        
        if invalid_count > 0:
            st.warning(f"⚠️ Found {invalid_count} invalid rows (NaN, negative values)")
        
        # Create clean dataset
        clean_data = pd.DataFrame({
            'TimeMinutes': np.arange(valid_mask.sum()) / 60
        })
        
        for metric, data in clean_data_dict.items():
            clean_data[metric] = data[valid_mask].values
        
        self.original_data = clean_data.copy()
        
        if self.debug_mode:
            st.write(f"✅ **Final data check**:")
            st.write(f"   - Total rows: {len(clean_data)}")
            st.write(f"   - Available metrics: {list(self.available_columns.keys())}")
        
        st.success(f"✅ Data loaded successfully using delimiter '{delimiter}' and encoding '{encoding}'")
        
        # Data summary
        cols = st.columns(min(4, len(available_columns)))
        for i, (metric, col_name) in enumerate(available_columns.items()):
            with cols[i % len(cols)]:
                if metric == 'FPS':
                    st.metric(f"🎯 {metric}", f"{clean_data[metric].min():.0f} - {clean_data[metric].max():.0f}")
                elif metric == 'CPU':
                    st.metric(f"🖥️ {metric}", f"{clean_data[metric].min():.0f}% - {clean_data[metric].max():.0f}%")
                else:
                    st.metric(f"📊 {metric}", f"{clean_data[metric].min():.0f} - {clean_data[metric].max():.0f}")
        
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
            percentile_threshold = np.percentile(fps_data, threshold)
            keep_mask = fps_data >= percentile_threshold
            
        elif method == 'iqr':
            Q1 = fps_data.quantile(0.25)
            Q3 = fps_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            keep_mask = fps_data >= lower_bound
            
        elif method == 'zscore':
            z_scores = np.abs((fps_data - fps_data.mean()) / fps_data.std())
            keep_mask = z_scores <= threshold
        
        # Apply mask
        self.processed_data = self.original_data[keep_mask].copy().reset_index(drop=True)
        self.removed_indices = self.original_data[~keep_mask].index.tolist()
        
        # Update time column
        self.processed_data['TimeMinutes'] = np.arange(len(self.processed_data)) / 60
        
        removed_count = len(self.removed_indices)
        removal_pct = (removed_count / original_count) * 100
        
        if self.debug_mode:
            st.write(f"🔍 **Outlier removal - After**: {len(self.processed_data)} rows, FPS range: {self.processed_data['FPS'].min():.1f} - {self.processed_data['FPS'].max():.1f}")
        
        st.info(f"🚫 Removed {removed_count} outliers ({removal_pct:.1f}%) using {method} method")
        
        return True
    
    def create_performance_chart(self, config):
        """Create performance chart with dynamic toggles"""
        try:
            data = self.processed_data if self.processed_data is not None else self.original_data
            
            if data is None:
                st.error("❌ No data available for chart generation")
                return None
            
            # Create figure with dark theme
            plt.style.use('dark_background')
            fig, ax1 = plt.subplots(figsize=(16, 9))
            fig.patch.set_facecolor('#0E1117')
            
            # Time data
            time_data = data['TimeMinutes']
            
            # Color scheme for different metrics
            colors = {
                'FPS': config.get('fps_color', '#00D4FF'),
                'CPU': config.get('cpu_color', '#FF6B35'),
                'GPU': '#00FF7F',
                'RAM': '#FFD700',
                'Temperature': '#FF1493',
                'Battery': '#32CD32',
                'Jank': '#FF4500',
                'Network': '#9370DB'
            }
            
            # Plot CPU first (background) if enabled
            if not config.get('hide_cpu', False) and 'CPU' in data.columns:
                ax1.set_xlabel('Time (minutes)', fontsize=12, color='white', fontweight='bold')
                ax1.set_ylabel('CPU Usage (%)', fontsize=12, color=colors['CPU'], fontweight='bold')
                ax1.tick_params(axis='both', colors='white', labelsize=10)
                ax1.set_ylim(0, 100)
                
                cpu_data = data['CPU']
                ax1.plot(time_data, cpu_data, 
                        color=colors['CPU'], linewidth=2.0, 
                        label='CPU Usage', alpha=0.6, zorder=1)
            
            # Create secondary axis for FPS (foreground)
            ax2 = ax1.twinx()
            
            if not config.get('hide_fps', False) and 'FPS' in data.columns:
                ax2.set_ylabel('FPS', fontsize=12, color=colors['FPS'], fontweight='bold')
                ax2.tick_params(axis='y', colors='white', labelsize=10)
                
                fps_data = data['FPS']
                fps_max = fps_data.max() * 1.1
                ax2.set_ylim(0, fps_max)
                
                ax2.plot(time_data, fps_data, 
                        color=colors['FPS'], linewidth=3.0, 
                        label='FPS', alpha=0.9, zorder=3)
            
            # Plot additional metrics if available and enabled
            additional_axes = {}
            y_position = 0.05
            
            for metric in ['GPU', 'RAM', 'Temperature', 'Battery', 'Jank', 'Network']:
                if (metric in data.columns and 
                    not config.get(f'hide_{metric.lower()}', False)):
                    
                    # Create additional axis for this metric
                    if metric not in additional_axes:
                        if len(additional_axes) == 0:
                            ax_new = ax1.twinx()
                        else:
                            ax_new = ax1.twinx()
                            # Offset the spine
                            ax_new.spines['right'].set_position(('outward', 60 * len(additional_axes)))
                        
                        additional_axes[metric] = ax_new
                    
                    ax_metric = additional_axes[metric]
                    metric_data = data[metric]
                    
                    # Set appropriate y-limits based on metric type
                    if metric in ['GPU', 'RAM', 'Battery']:
                        ax_metric.set_ylim(0, 100)
                    elif metric == 'Temperature':
                        ax_metric.set_ylim(20, 100)
                    else:
                        ax_metric.set_ylim(metric_data.min() * 0.9, metric_data.max() * 1.1)
                    
                    ax_metric.plot(time_data, metric_data,
                                  color=colors[metric], linewidth=2.0,
                                  label=metric, alpha=0.7, zorder=2)
                    
                    ax_metric.set_ylabel(f'{metric}', fontsize=10, color=colors[metric])
                    ax_metric.tick_params(axis='y', colors=colors[metric], labelsize=8)
            
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
            
            # Legend - FPS first (more prominent)
            if config['smartphone_name']:
                legend_elements = [plt.Line2D([0], [0], color='none', label=config['smartphone_name'])]
                
                # Add legend items for visible metrics
                if not config.get('hide_fps', False) and 'FPS' in data.columns:
                    legend_elements.append(plt.Line2D([0], [0], color=colors['FPS'], 
                                                    linewidth=3.0, label='FPS'))
                if not config.get('hide_cpu', False) and 'CPU' in data.columns:
                    legend_elements.append(plt.Line2D([0], [0], color=colors['CPU'], 
                                                    linewidth=2, label='CPU Usage'))
                
                for metric in ['GPU', 'RAM', 'Temperature', 'Battery', 'Jank', 'Network']:
                    if (metric in data.columns and 
                        not config.get(f'hide_{metric.lower()}', False)):
                        legend_elements.append(plt.Line2D([0], [0], color=colors[metric], 
                                                        linewidth=2, label=metric))
                
                legend = ax2.legend(handles=legend_elements, loc='upper right', 
                                  framealpha=0.9, fancybox=True)
                legend.get_frame().set_facecolor('#262730')
                for text in legend.get_texts():
                    text.set_color('white')
                    if text.get_text() == config['smartphone_name']:
                        text.set_fontweight('bold')
                    elif text.get_text() == 'FPS':
                        text.set_fontweight('bold')  # Make FPS bold in legend
            
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
        """Calculate performance statistics"""
        if self.original_data is None:
            return {}
        
        data = self.processed_data if self.processed_data is not None else self.original_data
        fps_data = data['FPS']
        cpu_data = data['CPU']
        
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
        
        # Frame statistics
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
    
    def create_shadow_table(self):
        """Create shadow table for video chart creation"""
        data = self.processed_data if self.processed_data is not None else self.original_data
        
        if data is None:
            return None
        
        # Detect frame rate from data frequency
        total_frames = len(data)
        total_time_minutes = data['TimeMinutes'].max()
        
        # Calculate estimated FPS based on data frequency
        if total_time_minutes > 0:
            frames_per_minute = total_frames / total_time_minutes
            estimated_fps = frames_per_minute / 60  # Convert to frames per second
            
            # Round to common gaming FPS values
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
            detected_fps = 60  # Default fallback
        
        # Create shadow table
        shadow_table = pd.DataFrame({
            'Frame': range(1, len(data) + 1),
            'Time': (data.index / detected_fps / 60).round(4),  # Convert frame to minutes
            'FPS': data['FPS'].round(1),
            'CPU%': data['CPU'].round(1)
        })
        
        return shadow_table, detected_fps
    
    def export_processed_data(self, game_title):
        """Export processed data to CSV"""
        data = self.processed_data if self.processed_data is not None else self.original_data
        
        if data is None:
            return None, None
        
        # Prepare export data
        export_data = pd.DataFrame({
            'Time_Minutes': data['TimeMinutes'].round(3),
            'FPS': data['FPS'].round(1),
            'CPU_Percent': data['CPU'].round(1)
        })
        
        # Convert to CSV
        csv_buffer = io.StringIO()
        export_data.to_csv(csv_buffer, index=False)
        csv_content = csv_buffer.getvalue()
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{game_title.replace(' ', '_')}_processed_{timestamp}.csv"
        
        if self.debug_mode:
            st.success(f"✅ **Export ready**: {len(export_data)} rows")
        
        return csv_content, filename
    
    def export_shadow_table(self, game_title):
        """Export shadow table for video chart creation"""
        shadow_table, detected_fps = self.create_shadow_table()
        
        if shadow_table is None:
            return None, None, None
        
        # Convert to CSV
        csv_buffer = io.StringIO()
        shadow_table.to_csv(csv_buffer, index=False)
        csv_content = csv_buffer.getvalue()
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{game_title.replace(' ', '_')}_video_chart_{timestamp}.csv"
        
        return csv_content, filename, detected_fps

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
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎮 Gaming Performance Analyzer</h1>
        <p>Professional gaming chart generator</p>
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
            help="CSV should contain FPS and CPU usage columns"
        )
        
        if uploaded_file is not None:
            # Load data
            file_data = uploaded_file.read()
            
            with st.spinner('📊 Loading and validating data...'):
                if analyzer.load_csv_data(file_data, uploaded_file.name):
                    
                    if analyzer.original_data is not None:
                        
                        # Process data if outlier removal is enabled
                        if enable_outlier_removal:
                            with st.spinner('🔧 Removing outliers...'):
                                analyzer.remove_outliers(outlier_method, outlier_threshold)
                        else:
                            # Use original data directly
                            analyzer.processed_data = analyzer.original_data.copy()
                            st.info("📊 **Raw Mode**: Using original data without processing")
                        
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
            
            # Shadow table preview and export
            shadow_table, detected_fps = analyzer.create_shadow_table()
            if shadow_table is not None:
                st.subheader("🎬 Video Chart Table")
                st.info(f"📹 **Detected Frame Rate**: {detected_fps} FPS")
                
                # Show preview
                with st.expander("👀 Preview Video Chart Data (First 10 rows)"):
                    st.dataframe(shadow_table.head(10), use_container_width=True)
                
                # Export shadow table
                shadow_csv, shadow_filename, _ = analyzer.export_shadow_table(game_title)
                if shadow_csv:
                    st.download_button(
                        label="🎬 Download Video Chart CSV",
                        data=shadow_csv,
                        file_name=shadow_filename,
                        mime="text/csv",
                        use_container_width=True,
                        help="Perfect for creating animated video charts"
                    )
            
            # Regular data export
            csv_content, csv_filename = analyzer.export_processed_data(game_title)
            if csv_content:
                st.download_button(
                    label="📄 Download Processed Data",
                    data=csv_content,
                    file_name=csv_filename,
                    mime="text/csv",
                    use_container_width=True
                )
        else:
            st.info("📤 Upload CSV file to see performance statistics")
    
    # Documentation
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
        
        **Raw Mode:**
        - Shows original CSV data without any processing
        - Fastest chart generation
        - Maximum accuracy to source data
        """)
    
    with st.expander("🎬 Video Chart Table"):
        st.markdown("""
        **Video Chart CSV Features:**
        - **Frame Column**: Sequential frame numbers (1, 2, 3...)
        - **Time Column**: Frame converted to minutes based on detected FPS
        - **FPS Column**: Raw FPS data from your CSV
        - **CPU% Column**: Raw CPU usage data from your CSV
        
        **Frame Rate Detection:**
        - **Auto-detects** from your data frequency
        - **Common rates**: 30, 60, 90, 120 FPS
        - **Perfect for**: After Effects, Premiere Pro, DaVinci Resolve
        
        **Use Cases:**
        - 🎥 **Animated Charts**: Frame-by-frame data progression
        - 📊 **Video Presentations**: Time-synced performance data
        - 🎬 **Content Creation**: Professional gaming analysis videos
        - 📈 **Social Media**: Engaging performance visualizations
        
        **Workflow:**
        1. Upload your gaming CSV
        2. Download Video Chart CSV
        3. Import to your video editor
        4. Create animated performance charts
        """)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #888; padding: 1rem;">
        🎮 Gaming Performance Analyzer v3.1 - Video Chart Ready<br>
        <small>📊 Raw Data Focus • 🎬 Video Chart Export • 📈 Performance Analytics</small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
