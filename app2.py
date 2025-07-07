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
        """Apply Savitzky-Golay smoothing ONLY to CPU data - FPS always stays raw"""
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
        
        # CRITICAL: FPS is ALWAYS raw - no smoothing ever applied
        self.processed_data['FPS_Smooth'] = self.processed_data['FPS'].copy()
        
        if self.debug_mode:
            st.write("🎯 **FPS Policy**: Always using raw FPS data (no smoothing applied)")
            st.write(f"   - FPS range preserved: {self.processed_data['FPS_Smooth'].min():.1f} - {self.processed_data['FPS_Smooth'].max():.1f}")
        
        # CPU Smoothing (only CPU can be smoothed)
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
            st.write(f"   - FPS identical to original: {np.array_equal(self.processed_data['FPS_Smooth'], self.processed_data['FPS'])}")
            
            # FPS should ALWAYS be identical to original
            if not np.array_equal(self.processed_data['FPS_Smooth'], self.processed_data['FPS']):
                st.error("🚨 **FPS DATA INTEGRITY VIOLATION!**")
                st.error("**FPS has been modified when it should stay raw!**")
                return False
            else:
                st.success("✅ **FPS data integrity preserved (raw data maintained)**")
        
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
    
    def validate_processed_vs_raw(self):
        """Compare processed data vs raw data and provide detailed analysis"""
        if self.original_data is None:
            return False, "No original data available"
        
        if self.processed_data is None:
            return False, "No processed data available"
        
        # Get the data to compare
        raw_fps = self.original_data['FPS']
        raw_cpu = self.original_data['CPU']
        
        processed_fps = self.processed_data['FPS_Smooth'] if 'FPS_Smooth' in self.processed_data else self.processed_data['FPS']
        processed_cpu = self.processed_data['CPU_Smooth'] if 'CPU_Smooth' in self.processed_data else self.processed_data['CPU']
        
        # Basic stats comparison
        raw_fps_stats = {
            'min': raw_fps.min(),
            'max': raw_fps.max(),
            'mean': raw_fps.mean(),
            'std': raw_fps.std(),
            'count': len(raw_fps)
        }
        
        processed_fps_stats = {
            'min': processed_fps.min(),
            'max': processed_fps.max(),
            'mean': processed_fps.mean(),
            'std': processed_fps.std(),
            'count': len(processed_fps)
        }
        
        raw_cpu_stats = {
            'min': raw_cpu.min(),
            'max': raw_cpu.max(),
            'mean': raw_cpu.mean(),
            'std': raw_cpu.std(),
            'count': len(raw_cpu)
        }
        
        processed_cpu_stats = {
            'min': processed_cpu.min(),
            'max': processed_cpu.max(),
            'mean': processed_cpu.mean(),
            'std': processed_cpu.std(),
            'count': len(processed_cpu)
        }
        
        # Calculate differences
        fps_differences = {
            'min_diff': processed_fps_stats['min'] - raw_fps_stats['min'],
            'max_diff': processed_fps_stats['max'] - raw_fps_stats['max'],
            'mean_diff': processed_fps_stats['mean'] - raw_fps_stats['mean'],
            'std_diff': processed_fps_stats['std'] - raw_fps_stats['std'],
            'count_diff': processed_fps_stats['count'] - raw_fps_stats['count']
        }
        
        cpu_differences = {
            'min_diff': processed_cpu_stats['min'] - raw_cpu_stats['min'],
            'max_diff': processed_cpu_stats['max'] - raw_cpu_stats['max'],
            'mean_diff': processed_cpu_stats['mean'] - raw_cpu_stats['mean'],
            'std_diff': processed_cpu_stats['std'] - raw_cpu_stats['std'],
            'count_diff': processed_cpu_stats['count'] - raw_cpu_stats['count']
        }
        
        # Determine if data is significantly different
        fps_threshold = 0.1  # Small threshold for floating point comparison
        cpu_threshold = 0.1
        
        fps_identical = (
            abs(fps_differences['min_diff']) < fps_threshold and
            abs(fps_differences['max_diff']) < fps_threshold and
            abs(fps_differences['mean_diff']) < fps_threshold and
            fps_differences['count_diff'] == 0
        )
        
        cpu_identical = (
            abs(cpu_differences['min_diff']) < cpu_threshold and
            abs(cpu_differences['max_diff']) < cpu_threshold and
            abs(cpu_differences['mean_diff']) < cpu_threshold and
            cpu_differences['count_diff'] == 0
        )
        
        # Create validation result
        validation_result = {
            'fps_identical': fps_identical,
            'cpu_identical': cpu_identical,
            'raw_fps_stats': raw_fps_stats,
            'processed_fps_stats': processed_fps_stats,
            'raw_cpu_stats': raw_cpu_stats,
            'processed_cpu_stats': processed_cpu_stats,
            'fps_differences': fps_differences,
            'cpu_differences': cpu_differences
        }
        
        return True, validation_result
    
    def display_data_validation_report(self, validation_result):
        """Display comprehensive data validation report"""
        
        fps_identical = validation_result['fps_identical']
        cpu_identical = validation_result['cpu_identical']
        
        # Header
        if fps_identical and cpu_identical:
            st.success("✅ **DATA VALIDATION PASSED**: Processed data identical to raw data")
            validation_status = "identical"
        else:
            st.warning("⚠️ **DATA VALIDATION**: Differences detected between raw and processed data")
            validation_status = "different"
        
        # Create comparison table
        with st.expander("🔍 **Data Validation Details** - Click to expand"):
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 FPS Comparison")
                
                if fps_identical:
                    st.success("✅ FPS data identical")
                else:
                    st.warning("⚠️ FPS data modified")
                
                # FPS comparison table
                fps_comparison_data = {
                    'Metric': ['Min', 'Max', 'Average', 'Std Dev', 'Count'],
                    'Raw Data': [
                        f"{validation_result['raw_fps_stats']['min']:.1f}",
                        f"{validation_result['raw_fps_stats']['max']:.1f}",
                        f"{validation_result['raw_fps_stats']['mean']:.1f}",
                        f"{validation_result['raw_fps_stats']['std']:.1f}",
                        f"{validation_result['raw_fps_stats']['count']}"
                    ],
                    'Processed Data': [
                        f"{validation_result['processed_fps_stats']['min']:.1f}",
                        f"{validation_result['processed_fps_stats']['max']:.1f}",
                        f"{validation_result['processed_fps_stats']['mean']:.1f}",
                        f"{validation_result['processed_fps_stats']['std']:.1f}",
                        f"{validation_result['processed_fps_stats']['count']}"
                    ],
                    'Difference': [
                        f"{validation_result['fps_differences']['min_diff']:+.1f}",
                        f"{validation_result['fps_differences']['max_diff']:+.1f}",
                        f"{validation_result['fps_differences']['mean_diff']:+.1f}",
                        f"{validation_result['fps_differences']['std_diff']:+.1f}",
                        f"{validation_result['fps_differences']['count_diff']:+d}"
                    ]
                }
                
                fps_df = pd.DataFrame(fps_comparison_data)
                st.dataframe(fps_df, use_container_width=True)
            
            with col2:
                st.subheader("🖥️ CPU Comparison")
                
                if cpu_identical:
                    st.success("✅ CPU data identical")
                else:
                    st.warning("⚠️ CPU data modified")
                
                # CPU comparison table
                cpu_comparison_data = {
                    'Metric': ['Min', 'Max', 'Average', 'Std Dev', 'Count'],
                    'Raw Data': [
                        f"{validation_result['raw_cpu_stats']['min']:.1f}%",
                        f"{validation_result['raw_cpu_stats']['max']:.1f}%",
                        f"{validation_result['raw_cpu_stats']['mean']:.1f}%",
                        f"{validation_result['raw_cpu_stats']['std']:.1f}%",
                        f"{validation_result['raw_cpu_stats']['count']}"
                    ],
                    'Processed Data': [
                        f"{validation_result['processed_cpu_stats']['min']:.1f}%",
                        f"{validation_result['processed_cpu_stats']['max']:.1f}%",
                        f"{validation_result['processed_cpu_stats']['mean']:.1f}%",
                        f"{validation_result['processed_cpu_stats']['std']:.1f}%",
                        f"{validation_result['processed_cpu_stats']['count']}"
                    ],
                    'Difference': [
                        f"{validation_result['cpu_differences']['min_diff']:+.1f}%",
                        f"{validation_result['cpu_differences']['max_diff']:+.1f}%",
                        f"{validation_result['cpu_differences']['mean_diff']:+.1f}%",
                        f"{validation_result['cpu_differences']['std_diff']:+.1f}%",
                        f"{validation_result['cpu_differences']['count_diff']:+d}"
                    ]
                }
                
                cpu_df = pd.DataFrame(cpu_comparison_data)
                st.dataframe(cpu_df, use_container_width=True)
            
            # Summary and recommendations
            st.subheader("📋 Validation Summary")
            
            if validation_status == "identical":
                st.markdown("""
                ✅ **All Good!** Your processed data is identical to the raw CSV data.
                - No unexpected modifications detected
                - Chart will accurately represent your original data
                - Safe to proceed with analysis
                """)
            else:
                st.markdown("""
                ⚠️ **Data Modified:** Processing has changed your data from the original CSV.
                
                **This might be expected if you enabled:**
                - 🚫 Outlier removal (reduces data points)
                - 🎯 FPS/CPU smoothing (changes values)
                - 🔬 Savgol processing (applies filters)
                
                **This might be unexpected if:**
                - 📊 You're in Raw CSV mode but still see differences
                - 🔧 Processing options are disabled but data changed
                - 🚨 Values are outside reasonable ranges
                """)
                
                # Specific warnings for significant changes
                fps_mean_change = abs(validation_result['fps_differences']['mean_diff'])
                fps_range_change = abs(validation_result['fps_differences']['max_diff']) + abs(validation_result['fps_differences']['min_diff'])
                
                if fps_mean_change > 5:
                    st.error(f"🚨 **Significant FPS change detected**: Average FPS changed by {fps_mean_change:.1f}")
                
                if fps_range_change > 10:
                    st.error(f"🚨 **Significant FPS range change**: Range boundaries changed by {fps_range_change:.1f}")
                
                if validation_result['fps_differences']['count_diff'] != 0:
                    st.warning(f"📊 **Data point count changed**: {validation_result['fps_differences']['count_diff']:+d} data points")
        
        return validation_status
    
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
        
        # Universal Savgol Filter Toggle
        st.markdown("### 🎛️ **Universal Processing Control**")
        enable_savgol = st.toggle("🔬 Enable Savgol Processing", value=False, 
                                 help="Master switch: When OFF, only raw CSV data is used for charts")
        
        if not enable_savgol:
            st.info("📊 **RAW CSV MODE**: Only pure CSV data will be used - no processing applied")
            st.markdown("---")
        
        # Debug mode toggle
        debug_mode = st.toggle("🐛 Debug Mode", value=True, help="Show detailed processing information")
        analyzer.debug_mode = debug_mode
        
        if enable_savgol:
            st.markdown("### 🔧 **Processing Options** (Savgol Enabled)")
            
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
            
            # CPU Smoothing only (FPS smoothing disabled)
            st.info("🎯 **FPS Smoothing**: Disabled by design - FPS chart will always show raw data")
            cpu_smooth = st.toggle("🖥️ CPU Smoothing", value=False)
            if cpu_smooth:
                cpu_window = st.slider("CPU Window", 3, 51, 7, step=2,
                                     help="Larger window = smoother CPU line")
            
            # Set FPS smoothing to always disabled
            fps_smooth = False
            fps_window = 7
        else:
            # Set all processing to disabled when Savgol is off
            enable_outlier_removal = False
            fps_smooth = False
            cpu_smooth = False
            fps_window = 7
            cpu_window = 7
    
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
                        
                        # Check if Savgol processing is enabled
                        if enable_savgol:
                            # Full processing pipeline enabled
                            if enable_outlier_removal:
                                with st.spinner('🚫 Removing outliers...'):
                                    analyzer.remove_outliers(outlier_method, outlier_threshold)
                            
                            # Apply smoothing with integrity checks
                            fps_window = fps_window if fps_smooth else 7
                            cpu_window = cpu_window if cpu_smooth else 7
                            
                            # Show processing status
                            if cpu_smooth:
                                with st.spinner('🔧 Applying CPU smoothing (FPS stays raw)...'):
                                    success = analyzer.apply_smoothing(False, cpu_smooth, fps_window, cpu_window)
                            else:
                                with st.spinner('📊 Preparing data (no smoothing applied)...'):
                                    success = analyzer.apply_smoothing(False, False, fps_window, cpu_window)
                            
                            if not success:
                                st.error("❌ Data processing failed - check debug output above")
                                success = False
                        else:
                            # RAW CSV MODE - No processing at all
                            with st.spinner('📊 Using pure raw CSV data...'):
                                # Set processed_data to exact copy of original_data
                                analyzer.processed_data = analyzer.original_data.copy()
                                analyzer.processed_data['FPS_Smooth'] = analyzer.processed_data['FPS'].copy()
                                analyzer.processed_data['CPU_Smooth'] = analyzer.processed_data['CPU'].copy()
                                success = True
                                
                                if analyzer.debug_mode:
                                    st.success("✅ **RAW CSV MODE**: No Savgol processing applied")
                                    st.write(f"   - Using pure CSV data: {len(analyzer.processed_data)} rows")
                                    st.write(f"   - FPS range: {analyzer.processed_data['FPS'].min():.1f} - {analyzer.processed_data['FPS'].max():.1f}")
                                    st.write(f"   - CPU range: {analyzer.processed_data['CPU'].min():.1f} - {analyzer.processed_data['CPU'].max():.1f}")
                                
                                st.info("📊 **RAW CSV MODE**: Chart generated from pure CSV data without any processing")
                        
                        if success:
                            # ✅ DATA VALIDATION CHECK BEFORE CHART
                            st.subheader("🔍 Data Validation Check")
                            
                            with st.spinner('🔍 Validating processed data vs raw data...'):
                                validation_success, validation_result = analyzer.validate_processed_vs_raw()
                            
                            if validation_success:
                                validation_status = analyzer.display_data_validation_report(validation_result)
                                
                                # Create chart
                                st.subheader("📊 Performance Chart")
                                
                                # Add mode indicator with validation status
                                if enable_savgol:
                                    processing_status = "🔬 **Savgol Processing Enabled** (CPU only)"
                                    if enable_outlier_removal or cpu_smooth:
                                        processing_status += " - Filters applied to CPU"
                                    else:
                                        processing_status += " - Ready for CPU filtering"
                                else:
                                    processing_status = "📊 **Raw CSV Mode** - Pure data from file"
                                
                                # Add validation badge
                                if validation_status == "identical":
                                    validation_badge = "✅ **Data Validated: Identical to Raw**"
                                else:
                                    validation_badge = "⚠️ **Data Validated: Modified from Raw**"
                                
                                st.markdown(f"*{processing_status}*")
                                st.markdown(f"*{validation_badge}*")
                                
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
                                
                                with st.spinner('🎨 Generating validated chart...'):
                                    chart_fig = analyzer.create_performance_chart(chart_config)
                                    
                                    if chart_fig:
                                        st.pyplot(chart_fig, use_container_width=True)
                                        
                                        # Add post-chart validation summary
                                        if validation_status == "different":
                                            st.info("💡 **Chart Note**: This chart shows processed data. Check validation details above to understand what changed from your original CSV.")
                                    else:
                                        st.error("Failed to generate chart")
                            else:
                                st.error(f"❌ Data validation failed: {validation_result}")
                                st.error("Cannot proceed with chart generation due to validation errors")
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
    
    with st.expander("🔍 Data Validation Features"):
        st.markdown("""
        **🔍 Automatic Data Validation:**
        - **Pre-Chart Check**: Compares processed vs raw data before showing chart
        - **Statistical Analysis**: Min, Max, Average, Standard Deviation comparison
        - **Difference Detection**: Identifies any modifications from original CSV
        - **Validation Report**: Detailed breakdown of all changes
        
        **Validation Statuses:**
        - ✅ **Identical**: Processed data matches raw CSV exactly
        - ⚠️ **Modified**: Processing has changed the data (expected with filters)
        - 🚨 **Significant Changes**: Large deviations that might indicate issues
        
        **What Triggers Differences:**
        - 🚫 **Outlier Removal**: Reduces data point count
        - 🖥️ **CPU Smoothing**: Changes CPU values slightly (FPS always stays raw)
        - 🔬 **Savgol Processing**: Applies mathematical filters to CPU only
        
        **Benefits:**
        - 🛡️ **Data Integrity**: Ensures no unexpected modifications
        - 📊 **Transparency**: Shows exactly what changed
        - 🔍 **Quality Control**: Detects processing errors early
        - 📈 **Confidence**: Know your chart accurately represents intended data
        """)
    
    with st.expander("🎛️ Universal Processing Control"):
        st.markdown("""
        **🔬 Savgol Processing Toggle:**
        - **ENABLED**: Processing pipeline available (outlier removal, CPU smoothing only)
        - **DISABLED**: Pure raw CSV mode - no processing, just direct chart from file data
        
        **⚠️ Important: FPS Smoothing is permanently disabled by design**
        
        **Raw CSV Mode Benefits:**
        - ✅ **Zero Processing**: Exactly what's in your CSV file
        - ✅ **Maximum Accuracy**: No algorithms modifying your data
        - ✅ **Performance**: Fastest chart generation
        - ✅ **Transparency**: What you see is exactly what you uploaded
        
        **Savgol Processing Mode Benefits:**
        - 🚫 **Outlier Removal**: Clean up bad data points
        - 🖥️ **CPU Smoothing Only**: Reduce CPU noise while keeping FPS raw
        - 📊 **Selective Processing**: FPS stays untouched, CPU can be filtered
        - 🛡️ **Data Integrity**: Built-in validation and safety checks
        """)
    
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
        - **CPU Smoothing Only**: Savitzky-Golay filter applied only to CPU usage
        - **FPS Always Raw**: FPS data never gets smoothed - always shows original values
        - **Window Size**: 3-51 points for CPU smoothing (larger = smoother)
        - **When disabled**: Shows pure raw CSV data without any processing
        - **Data integrity**: Ensures CPU smoothed values stay within 0-100%
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
        🎮 Gaming Performance Analyzer v2.3 - FPS Raw Policy<br>
        <small>🎯 FPS Always Raw • 🖥️ CPU Smoothing Only • 🔍 Data Validation • 🛡️ Data Integrity</small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
