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
        self.available_columns = []
        self.numeric_columns = []
        
        # Comparison data
        self.original_data_2 = None
        self.processed_data_2 = None
        self.removed_indices_2 = []
        self.available_columns_2 = []
        self.numeric_columns_2 = []
    
    def load_csv_data(self, file_data, filename, dataset_num=1):
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
                            if self._validate_and_process_columns(df, delimiter, encoding, dataset_num):
                                return True
                    except Exception as e:
                        if self.debug_mode:
                            st.write(f"Debug Dataset {dataset_num}: Failed {encoding} + {delimiter}: {str(e)}")
                        continue
            
            st.error(f"❌ Could not parse CSV file {dataset_num}. Please check the format.")
            return False
            
        except Exception as e:
            st.error(f"❌ Error loading CSV {dataset_num}: {str(e)}")
            return False
    
    def _validate_and_process_columns(self, df, delimiter, encoding, dataset_num=1):
        """Validate columns and process data dynamically"""
        if dataset_num == 1:
            self.available_columns = list(df.columns)
            target_numeric_columns = 'numeric_columns'
            target_original_data = 'original_data'
        else:
            self.available_columns_2 = list(df.columns)
            target_numeric_columns = 'numeric_columns_2'
            target_original_data = 'original_data_2'
        
        current_columns = self.available_columns if dataset_num == 1 else self.available_columns_2
        
        if self.debug_mode:
            st.write(f"🔍 **Debug Dataset {dataset_num} - All columns found**: {current_columns}")
        
        # Convert all columns to numeric where possible and identify numeric columns
        processed_df = pd.DataFrame()
        numeric_cols = []
        
        for col in current_columns:
            # Try to convert to numeric
            numeric_data = pd.to_numeric(df[col], errors='coerce')
            
            # Check if at least 50% of data is numeric and not all NaN
            valid_numeric_ratio = (~numeric_data.isna()).sum() / len(df)
            
            if valid_numeric_ratio >= 0.5 and not numeric_data.isna().all():
                processed_df[col] = numeric_data
                numeric_cols.append(col)
                if self.debug_mode:
                    st.write(f"✅ **Dataset {dataset_num} Numeric column**: `{col}` (Range: {numeric_data.min():.1f} - {numeric_data.max():.1f})")
            else:
                # Keep as text/categorical data
                processed_df[col] = df[col]
                if self.debug_mode:
                    st.write(f"📝 **Dataset {dataset_num} Text column**: `{col}` (Sample: {df[col].iloc[0] if len(df) > 0 else 'N/A'})")
        
        if len(numeric_cols) < 1:
            st.error(f"❌ No numeric columns found for analysis in dataset {dataset_num}")
            return False
        
        # Set the appropriate attributes
        setattr(self, target_numeric_columns, numeric_cols)
        
        # Remove rows where ALL numeric columns are NaN
        numeric_data_only = processed_df[numeric_cols]
        valid_mask = ~numeric_data_only.isna().all(axis=1)
        
        # Also remove negative values and infinite values for performance metrics
        for col in numeric_cols:
            col_series = processed_df[col]
            valid_mask = valid_mask & (col_series >= 0) & ~np.isinf(col_series) & ~pd.isna(col_series)
        
        invalid_count = len(df) - valid_mask.sum()
        
        if invalid_count > 0:
            st.warning(f"⚠️ Found {invalid_count} invalid rows in dataset {dataset_num} (NaN, negative values)")
        
        # Create clean dataset
        clean_data = processed_df[valid_mask].copy().reset_index(drop=True)
        
        # Add time column
        clean_data['TimeMinutes'] = np.arange(len(clean_data)) / 60
        
        setattr(self, target_original_data, clean_data)
        
        if self.debug_mode:
            st.write(f"✅ **Dataset {dataset_num} Final data check**:")
            st.write(f"   - Total rows: {len(clean_data)}")
            st.write(f"   - Numeric columns: {numeric_cols}")
            st.write(f"   - All columns: {list(clean_data.columns)}")
        
        st.success(f"✅ Dataset {dataset_num} loaded successfully using delimiter '{delimiter}' and encoding '{encoding}'")
        
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
    
    def get_column_color_suggestion(self, col_name, dataset_num=1):
        """Suggest colors based on column type"""
        color_map = {
            'fps': '#00D4FF' if dataset_num == 1 else '#0099CC',      # Cyan variants for FPS
            'cpu': '#FF6B35' if dataset_num == 1 else '#CC5529',      # Orange variants for CPU
            'gpu': '#4ECDC4' if dataset_num == 1 else '#3BA39D',      # Teal variants for GPU
            'ram': '#45B7D1' if dataset_num == 1 else '#3692A7',      # Blue variants for RAM
            'memory': '#45B7D1' if dataset_num == 1 else '#3692A7',   # Blue variants for Memory
            'temp': '#FF4757' if dataset_num == 1 else '#CC3946',     # Red variants for Temperature
            'temperature': '#FF4757' if dataset_num == 1 else '#CC3946', # Red variants for Temperature
            'power': '#FFA726' if dataset_num == 1 else '#CC851E',    # Amber variants for Power
            'battery': '#66BB6A' if dataset_num == 1 else '#529655',  # Green variants for Battery
            'jank': '#8E24AA' if dataset_num == 1 else '#711D88',     # Purple variants for Jank
            'frame_drops': '#D32F2F' if dataset_num == 1 else '#A92626', # Dark Red variants for Frame Drops
            'latency': '#FF9800' if dataset_num == 1 else '#CC7A00',  # Orange variants for Latency
            'ping': '#FF9800' if dataset_num == 1 else '#CC7A00',     # Orange variants for Ping
            'network': '#9C27B0' if dataset_num == 1 else '#7D1F8D',  # Purple variants for Network
            'bandwidth': '#9C27B0' if dataset_num == 1 else '#7D1F8D', # Purple variants for Bandwidth
            'thermal': '#E91E63' if dataset_num == 1 else '#BA1A4F'   # Pink variants for Thermal
        }
        
        col_lower = col_name.lower()
        for key, color in color_map.items():
            if key in col_lower:
                return color
        
        # Default colors for unknown columns
        default_colors_1 = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8']
        default_colors_2 = ['#CC5555', '#3EA39D', '#3692A7', '#78A591', '#CCBB85', '#B088B8', '#7AA4A0']
        
        colors = default_colors_1 if dataset_num == 1 else default_colors_2
        return colors[hash(col_name) % len(colors)]
    
    def remove_outliers(self, method='percentile', threshold=1, target_columns=None, dataset_num=1):
        """Remove outliers from specified columns"""
        original_data = self.original_data if dataset_num == 1 else self.original_data_2
        numeric_columns = self.numeric_columns if dataset_num == 1 else self.numeric_columns_2
        
        if original_data is None:
            return False
        
        if target_columns is None:
            target_columns = numeric_columns
        
        original_count = len(original_data)
        data = original_data.copy()
        
        if self.debug_mode:
            st.write(f"🔍 **Dataset {dataset_num} Outlier removal - Before**: {original_count} rows")
            for col in target_columns:
                if col in data.columns:
                    col_data = data[col]
                    st.write(f"   - {col}: {col_data.min():.1f} to {col_data.max():.1f} (avg: {col_data.mean():.1f})")
        
        # Apply outlier removal to each specified column
        combined_mask = pd.Series([True] * len(data))
        
        for col in target_columns:
            if col not in data.columns or col not in numeric_columns:
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
        processed_data = data[combined_mask].copy().reset_index(drop=True)
        removed_indices = data[~combined_mask].index.tolist()
        
        # Update time column
        processed_data['TimeMinutes'] = np.arange(len(processed_data)) / 60
        
        # Set the appropriate attributes
        if dataset_num == 1:
            self.processed_data = processed_data
            self.removed_indices = removed_indices
        else:
            self.processed_data_2 = processed_data
            self.removed_indices_2 = removed_indices
        
        removed_count = len(removed_indices)
        removal_pct = (removed_count / original_count) * 100
        
        if self.debug_mode:
            st.write(f"🔍 **Dataset {dataset_num} Outlier removal - After**: {len(processed_data)} rows")
            for col in target_columns:
                if col in processed_data.columns:
                    col_data = processed_data[col]
                    st.write(f"   - {col}: {col_data.min():.1f} to {col_data.max():.1f} (avg: {col_data.mean():.1f})")
        
        st.success(f"✅ Dataset {dataset_num}: Removed {removed_count} outliers ({removal_pct:.1f}%) using {method} method on {len(target_columns)} columns")
        
        return True
    
    def create_performance_chart(self, config):
        """Create performance chart with FPS and CPU focus (dual y-axis)"""
        try:
            data = self.processed_data if self.processed_data is not None else self.original_data
            
            if data is None:
                st.error("❌ No data available for chart generation")
                return None
            
            # Find FPS and CPU columns
            fps_col = None
            cpu_col = None
            
            for col in self.numeric_columns:
                if 'fps' in col.lower() and fps_col is None:
                    fps_col = col
                elif 'cpu' in col.lower() and cpu_col is None:
                    cpu_col = col
            
            if not fps_col and not cpu_col:
                st.error("❌ No FPS or CPU columns found in the data")
                return None
            
            # Create figure with dark theme
            plt.style.use('dark_background')
            fig, ax1 = plt.subplots(figsize=(16, 9))
            fig.patch.set_facecolor('#0E1117')
            
            time_data = data['TimeMinutes']
            
            # Plot FPS on primary axis (left)
            if fps_col:
                fps_data = data[fps_col]
                fps_color = '#00D4FF'  # Cyan for FPS
                line1 = ax1.plot(time_data, fps_data, color=fps_color, linewidth=3, 
                               label='FPS', alpha=0.9, zorder=2)
                
                ax1.set_xlabel('Time (minutes)', fontsize=14, color='white', fontweight='bold')
                ax1.set_ylabel('FPS', fontsize=14, color=fps_color, fontweight='bold')
                ax1.tick_params(axis='y', labelcolor=fps_color, labelsize=12, colors=fps_color)
                ax1.tick_params(axis='x', colors='white', labelsize=12)
                
                # Set FPS limits (0 to max + 10%)
                fps_max = fps_data.max() * 1.1 if not pd.isna(fps_data.max()) else 120
                ax1.set_ylim(0, fps_max)
                
                # Add horizontal reference lines for FPS
                ax1.axhline(y=60, color='green', linestyle='--', alpha=0.5, linewidth=1)
                ax1.axhline(y=30, color='orange', linestyle='--', alpha=0.5, linewidth=1)
                ax1.text(time_data.iloc[-1] * 0.02, 62, '60 FPS', color='green', fontsize=10)
                ax1.text(time_data.iloc[-1] * 0.02, 32, '30 FPS', color='orange', fontsize=10)
            
            # Plot CPU on secondary axis (right)
            if cpu_col:
                ax2 = ax1.twinx()
                cpu_data = data[cpu_col]
                cpu_color = '#FF6B35'  # Orange for CPU
                line2 = ax2.plot(time_data, cpu_data, color=cpu_color, linewidth=3, 
                               label='CPU Usage (%)', alpha=0.9, zorder=1)
                
                ax2.set_ylabel('CPU Usage (%)', fontsize=14, color=cpu_color, fontweight='bold')
                ax2.tick_params(axis='y', labelcolor=cpu_color, labelsize=12, colors=cpu_color)
                
                # Set CPU limits (0 to 100%)
                ax2.set_ylim(0, 100)
                
                # Add horizontal reference line for CPU
                ax2.axhline(y=80, color='red', linestyle='--', alpha=0.3, linewidth=1)
                ax2.text(time_data.iloc[-1] * 0.02, 82, '80% CPU', color='red', fontsize=10)
            
            # Configure primary axis styling
            ax1.grid(True, alpha=0.3, linestyle='--', color='gray')
            ax1.set_facecolor('#0E1117')
            
            # Title
            title_lines = []
            if config['game_title']:
                title_lines.append(config['game_title'])
            if config['game_settings']:
                title_lines.append(config['game_settings'])
            if config['game_mode']:
                title_lines.append(config['game_mode'])
            
            if title_lines:
                plt.suptitle('\n'.join(title_lines), 
                           fontsize=20, fontweight='bold', 
                           color='white', y=0.95)
            
            # Create legend
            legend_elements = []
            if config['smartphone_name']:
                legend_elements.append(plt.Line2D([0], [0], color='white', 
                                                label=config['smartphone_name'], linewidth=3))
            
            if fps_col:
                legend_elements.append(plt.Line2D([0], [0], color=fps_color, 
                                                linewidth=3, label='FPS'))
            if cpu_col:
                legend_elements.append(plt.Line2D([0], [0], color=cpu_color, 
                                                linewidth=3, label='CPU Usage (%)'))
            
            if legend_elements:
                legend = ax1.legend(handles=legend_elements, loc='upper right', 
                                  framealpha=0.9, fancybox=True, bbox_to_anchor=(1.0, 1.0))
                legend.get_frame().set_facecolor('#262730')
                for text in legend.get_texts():
                    text.set_color('white')
                    if config['smartphone_name'] and text.get_text() == config['smartphone_name']:
                        text.set_fontweight('bold')
            
            # Clean up spines
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            ax1.spines['bottom'].set_color('white')
            ax1.spines['left'].set_color(fps_color if fps_col else 'white')
            
            if cpu_col:
                ax2.spines['top'].set_visible(False)
                ax2.spines['left'].set_visible(False)
                ax2.spines['right'].set_color(cpu_color)
                ax2.spines['bottom'].set_color('white')
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            st.error(f"❌ Chart generation failed: {str(e)}")
            return None
    
    def get_performance_stats(self, selected_columns=None):
        """Calculate performance statistics for selected columns (single dataset)"""
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
    
    def create_comparison_chart(self, config):
        """Create comparison chart with FPS and CPU focus (dual y-axis)"""
        try:
            # Get data from both datasets
            data1 = self.processed_data if self.processed_data is not None else self.original_data
            data2 = self.processed_data_2 if self.processed_data_2 is not None else self.original_data_2
            
            if data1 is None or data2 is None:
                st.error("❌ Need both datasets for comparison")
                return None
            
            # Find FPS and CPU columns in both datasets
            fps_col = None
            cpu_col = None
            
            # Find common FPS and CPU columns
            for col in self.numeric_columns:
                if col in self.numeric_columns_2:
                    if 'fps' in col.lower() and fps_col is None:
                        fps_col = col
                    elif 'cpu' in col.lower() and cpu_col is None:
                        cpu_col = col
            
            if not fps_col and not cpu_col:
                st.error("❌ No common FPS or CPU columns found between datasets")
                return None
            
            # Create figure with dark theme
            plt.style.use('dark_background')
            fig, ax1 = plt.subplots(figsize=(16, 9))
            fig.patch.set_facecolor('#0E1117')
            
            time_data1 = data1['TimeMinutes']
            time_data2 = data2['TimeMinutes']
            
            # Plot FPS on primary axis (left)
            if fps_col:
                fps_data1 = data1[fps_col]
                fps_data2 = data2[fps_col]
                fps_color1 = '#00D4FF'  # Bright cyan for dataset 1
                fps_color2 = '#0099CC'  # Darker cyan for dataset 2
                
                line1 = ax1.plot(time_data1, fps_data1, color=fps_color1, linewidth=3, 
                               label=f'FPS - {config["device_name_1"]}', alpha=0.9, zorder=2)
                line2 = ax1.plot(time_data2, fps_data2, color=fps_color2, linewidth=3, 
                               linestyle='--', label=f'FPS - {config["device_name_2"]}', alpha=0.9, zorder=2)
                
                ax1.set_xlabel('Time (minutes)', fontsize=14, color='white', fontweight='bold')
                ax1.set_ylabel('FPS', fontsize=14, color='#00D4FF', fontweight='bold')
                ax1.tick_params(axis='y', labelcolor='#00D4FF', labelsize=12, colors='#00D4FF')
                ax1.tick_params(axis='x', colors='white', labelsize=12)
                
                # Set FPS limits (0 to max + 10%)
                fps_max = max(fps_data1.max(), fps_data2.max()) * 1.1 if not pd.isna(fps_data1.max()) and not pd.isna(fps_data2.max()) else 120
                ax1.set_ylim(0, fps_max)
                
                # Add horizontal reference lines for FPS
                ax1.axhline(y=60, color='green', linestyle='--', alpha=0.5, linewidth=1)
                ax1.axhline(y=30, color='orange', linestyle='--', alpha=0.5, linewidth=1)
                ax1.text(max(time_data1.iloc[-1], time_data2.iloc[-1]) * 0.02, 62, '60 FPS', color='green', fontsize=10)
                ax1.text(max(time_data1.iloc[-1], time_data2.iloc[-1]) * 0.02, 32, '30 FPS', color='orange', fontsize=10)
            
            # Plot CPU on secondary axis (right)
            if cpu_col:
                ax2 = ax1.twinx()
                cpu_data1 = data1[cpu_col]
                cpu_data2 = data2[cpu_col]
                cpu_color1 = '#FF6B35'  # Bright orange for dataset 1
                cpu_color2 = '#CC5529'  # Darker orange for dataset 2
                
                line3 = ax2.plot(time_data1, cpu_data1, color=cpu_color1, linewidth=3, 
                               label=f'CPU - {config["device_name_1"]}', alpha=0.9, zorder=1)
                line4 = ax2.plot(time_data2, cpu_data2, color=cpu_color2, linewidth=3, 
                               linestyle='--', label=f'CPU - {config["device_name_2"]}', alpha=0.9, zorder=1)
                
                ax2.set_ylabel('CPU Usage (%)', fontsize=14, color='#FF6B35', fontweight='bold')
                ax2.tick_params(axis='y', labelcolor='#FF6B35', labelsize=12, colors='#FF6B35')
                
                # Set CPU limits (0 to 100%)
                ax2.set_ylim(0, 100)
                
                # Add horizontal reference line for CPU
                ax2.axhline(y=80, color='red', linestyle='--', alpha=0.3, linewidth=1)
                ax2.text(max(time_data1.iloc[-1], time_data2.iloc[-1]) * 0.02, 82, '80% CPU', color='red', fontsize=10)
            
            # Configure primary axis styling
            ax1.grid(True, alpha=0.3, linestyle='--', color='gray')
            ax1.set_facecolor('#0E1117')
            
            # Title
            title_lines = [f"🆚 Performance Comparison: {config['device_name_1']} vs {config['device_name_2']}"]
            if config['game_title']:
                title_lines.append(config['game_title'])
            if config['game_settings']:
                title_lines.append(config['game_settings'])
            
            plt.suptitle('\n'.join(title_lines), 
                        fontsize=18, fontweight='bold', 
                        color='white', y=0.95)
            
            # Create legend
            legend_elements = []
            
            if fps_col:
                legend_elements.append(plt.Line2D([0], [0], color=fps_color1, 
                                                linewidth=3, label=f'FPS - {config["device_name_1"]}'))
                legend_elements.append(plt.Line2D([0], [0], color=fps_color2, 
                                                linewidth=3, linestyle='--', label=f'FPS - {config["device_name_2"]}'))
            if cpu_col:
                legend_elements.append(plt.Line2D([0], [0], color=cpu_color1, 
                                                linewidth=3, label=f'CPU - {config["device_name_1"]}'))
                legend_elements.append(plt.Line2D([0], [0], color=cpu_color2, 
                                                linewidth=3, linestyle='--', label=f'CPU - {config["device_name_2"]}'))
            
            if legend_elements:
                legend = ax1.legend(handles=legend_elements, loc='upper right', 
                                  framealpha=0.9, fancybox=True, bbox_to_anchor=(1.0, 1.0))
                legend.get_frame().set_facecolor('#262730')
                for text in legend.get_texts():
                    text.set_color('white')
            
            # Clean up spines
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            ax1.spines['bottom'].set_color('white')
            ax1.spines['left'].set_color('#00D4FF' if fps_col else 'white')
            
            if cpu_col:
                ax2.spines['top'].set_visible(False)
                ax2.spines['left'].set_visible(False)
                ax2.spines['right'].set_color('#FF6B35')
                ax2.spines['bottom'].set_color('white')
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            st.error(f"❌ Comparison chart generation failed: {str(e)}")
            return None
    
    def get_comparison_stats(self, selected_columns=None):
        """Calculate comparison statistics for both datasets"""
        if self.original_data is None or self.original_data_2 is None:
            return {}
        
        data1 = self.processed_data if self.processed_data is not None else self.original_data
        data2 = self.processed_data_2 if self.processed_data_2 is not None else self.original_data_2
        
        if selected_columns is None:
            # Find common columns
            selected_columns = list(set(self.numeric_columns) & set(self.numeric_columns_2))
        
        stats = {
            'duration_1': len(data1) / 60,
            'duration_2': len(data2) / 60,
            'total_frames_1': len(data1),
            'total_frames_2': len(data2),
            'removed_frames_1': len(self.removed_indices),
            'removed_frames_2': len(self.removed_indices_2)
        }
        
        # Calculate stats for each selected column
        for col in selected_columns:
            if col in data1.columns and col in data2.columns:
                col_data1 = data1[col]
                col_data2 = data2[col]
                col_display = self.get_column_display_name(col)
                
                stats[f'{col}_avg_1'] = col_data1.mean()
                stats[f'{col}_avg_2'] = col_data2.mean()
                stats[f'{col}_max_1'] = col_data1.max()
                stats[f'{col}_max_2'] = col_data2.max()
                stats[f'{col}_min_1'] = col_data1.min()
                stats[f'{col}_min_2'] = col_data2.min()
                stats[f'{col}_display'] = col_display
                
                # Calculate improvement/difference
                if col_data1.mean() > 0:
                    improvement = ((col_data2.mean() - col_data1.mean()) / col_data1.mean()) * 100
                    stats[f'{col}_improvement'] = improvement
        
        return stats

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
    .comparison-header {
        text-align: center;
        background: linear-gradient(90deg, #ff6b6b 0%, #4ecdc4 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1rem;
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
        <p>Dynamic multi-metric gaming chart generator with comparison mode</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize analyzer
    analyzer = GamingPerformanceAnalyzer()
    
    # Mode selection
    analysis_mode = st.radio(
        "🔧 Choose Analysis Mode:",
        ["📊 Single CSV Analysis", "🆚 CSV Comparison Mode"],
        horizontal=True
    )
    
    if analysis_mode == "🆚 CSV Comparison Mode":
        # Comparison Mode
        st.markdown("""
        <div class="comparison-header">
            <h2>🆚 CSV Comparison Mode</h2>
            <p>Compare performance between two different datasets</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Device configuration
        col_config1, col_config2 = st.columns(2)
        
        with col_config1:
            st.subheader("📱 Dataset 1 Configuration")
            device_name_1 = st.text_input("Device/Setup Name 1", value="iPhone 15 Pro Max", key="device1")
            game_title = st.text_input("🎯 Game Title", value="Mobile Legends: Bang Bang", key="game_comparison")
        
        with col_config2:
            st.subheader("📱 Dataset 2 Configuration") 
            device_name_2 = st.text_input("Device/Setup Name 2", value="Samsung Galaxy S24 Ultra", key="device2")
            game_settings = st.text_input("⚙️ Settings/Mode", value="Ultra - 120 FPS", key="settings_comparison")
        
        # File uploads
        col_upload1, col_upload2 = st.columns(2)
        
        with col_upload1:
            st.subheader("📁 Upload Dataset 1")
            uploaded_file_1 = st.file_uploader(
                "Upload first CSV file", 
                type=['csv'],
                key="upload1",
                help="First dataset for comparison"
            )
        
        with col_upload2:
            st.subheader("📁 Upload Dataset 2")
            uploaded_file_2 = st.file_uploader(
                "Upload second CSV file", 
                type=['csv'],
                key="upload2",
                help="Second dataset for comparison"
            )
        
        # Process both files if uploaded
        datasets_loaded = 0
        
        if uploaded_file_1 is not None:
            file_data_1 = uploaded_file_1.read()
            with st.spinner('📊 Loading Dataset 1...'):
                if analyzer.load_csv_data(file_data_1, uploaded_file_1.name, 1):
                    datasets_loaded += 1
        
        if uploaded_file_2 is not None:
            file_data_2 = uploaded_file_2.read()
            with st.spinner('📊 Loading Dataset 2...'):
                if analyzer.load_csv_data(file_data_2, uploaded_file_2.name, 2):
                    datasets_loaded += 1
        
        if datasets_loaded == 2:
            # Find common columns
            common_columns = list(set(analyzer.numeric_columns) & set(analyzer.numeric_columns_2))
            
            if common_columns:
                st.success(f"✅ Both datasets loaded! Found {len(common_columns)} common metrics for comparison")
                
                # Column selection for comparison
                st.subheader("📊 Select Metrics for Comparison")
                
                selected_columns = st.multiselect(
                    "Choose common metrics to compare:",
                    common_columns,
                    default=common_columns[:3] if len(common_columns) >= 3 else common_columns,
                    help="Only columns present in both datasets can be compared"
                )
                
                if selected_columns:
                    # Outlier removal options
                    with st.expander("🔧 Data Processing Options"):
                        process_dataset1 = st.checkbox("🚫 Remove outliers from Dataset 1", key="outlier1")
                        process_dataset2 = st.checkbox("🚫 Remove outliers from Dataset 2", key="outlier2")
                        
                        if process_dataset1 or process_dataset2:
                            outlier_method = st.selectbox("Outlier Method", ['percentile', 'iqr', 'zscore'], key="outlier_method_comp")
                            if outlier_method == 'percentile':
                                outlier_threshold = st.slider("Bottom Percentile", 0.1, 5.0, 1.0, 0.1, key="outlier_threshold_comp")
                            elif outlier_method == 'zscore':
                                outlier_threshold = st.slider("Z-Score Threshold", 1.0, 4.0, 2.0, 0.1, key="outlier_zscore_comp")
                            else:
                                outlier_threshold = 1.5
                    
                    # Process datasets
                    if process_dataset1:
                        analyzer.remove_outliers(outlier_method, outlier_threshold, selected_columns, 1)
                    else:
                        analyzer.processed_data = analyzer.original_data.copy()
                    
                    if process_dataset2:
                        analyzer.remove_outliers(outlier_method, outlier_threshold, selected_columns, 2)
                    else:
                        analyzer.processed_data_2 = analyzer.original_data_2.copy()
                    
                    # Create comparison chart
                    st.subheader("📊 Performance Comparison Chart")
                    
                    chart_config = {
                        'game_title': game_title,
                        'game_settings': game_settings,
                        'device_name_1': device_name_1,
                        'device_name_2': device_name_2,
                        'selected_columns': selected_columns
                    }
                    
                    # Add color configurations for each dataset
                    color_cols = st.columns(len(selected_columns))
                    for i, col in enumerate(selected_columns):
                        col_display = analyzer.get_column_display_name(col)
                        
                        with color_cols[i]:
                            chart_config[f'{col}_color_1'] = st.color_picker(
                                f"{col_display[:8]}... D1", 
                                analyzer.get_column_color_suggestion(col, 1),
                                key=f"color_{col}_1",
                                help=f"Color for {col_display} - Dataset 1"
                            )
                            chart_config[f'{col}_color_2'] = st.color_picker(
                                f"{col_display[:8]}... D2", 
                                analyzer.get_column_color_suggestion(col, 2),
                                key=f"color_{col}_2",
                                help=f"Color for {col_display} - Dataset 2"
                            )
                    
                    with st.spinner('🎨 Generating comparison chart...'):
                        chart_fig = analyzer.create_comparison_chart(chart_config)
                        
                        if chart_fig:
                            st.pyplot(chart_fig, use_container_width=True)
                            
                            # Export comparison chart
                            img_buffer = io.BytesIO()
                            chart_fig.savefig(img_buffer, format='png', dpi=300, 
                                            bbox_inches='tight', facecolor='#0E1117')
                            img_buffer.seek(0)
                            
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            png_filename = f"comparison_{device_name_1.replace(' ', '_')}_vs_{device_name_2.replace(' ', '_')}_{timestamp}.png"
                            
                            st.download_button(
                                label="📸 Download Comparison Chart",
                                data=img_buffer.getvalue(),
                                file_name=png_filename,
                                mime="image/png",
                                use_container_width=True
                            )
                        else:
                            st.error("Failed to generate comparison chart")
                    
                    # Comparison Statistics
                    st.subheader("📈 Performance Comparison Statistics")
                    
                    stats = analyzer.get_comparison_stats(selected_columns)
                    
                    # Create comparison metrics
                    for col in selected_columns:
                        if f'{col}_avg_1' in stats and f'{col}_avg_2' in stats:
                            col_display = stats.get(f'{col}_display', analyzer.get_column_display_name(col))
                            
                            st.write(f"**{col_display} Comparison:**")
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric(
                                    f"📱 {device_name_1}", 
                                    f"{stats[f'{col}_avg_1']:.1f}",
                                    help=f"Average {col_display} for {device_name_1}"
                                )
                            
                            with col2:
                                st.metric(
                                    f"📱 {device_name_2}", 
                                    f"{stats[f'{col}_avg_2']:.1f}",
                                    help=f"Average {col_display} for {device_name_2}"
                                )
                            
                            with col3:
                                if f'{col}_improvement' in stats:
                                    improvement = stats[f'{col}_improvement']
                                    delta_color = "normal" if abs(improvement) < 5 else ("inverse" if improvement < 0 else "normal")
                                    
                                    st.metric(
                                        "📊 Difference", 
                                        f"{improvement:+.1f}%",
                                        delta=f"{improvement:+.1f}%",
                                        delta_color=delta_color,
                                        help=f"Performance difference: Dataset 2 vs Dataset 1"
                                    )
                            
                            st.divider()
                    
                    # Summary insights
                    if any('fps' in col.lower() for col in selected_columns):
                        fps_cols = [col for col in selected_columns if 'fps' in col.lower()]
                        if fps_cols:
                            fps_col = fps_cols[0]
                            fps_avg_1 = stats.get(f'{fps_col}_avg_1', 0)
                            fps_avg_2 = stats.get(f'{fps_col}_avg_2', 0)
                            
                            if fps_avg_1 > 0 and fps_avg_2 > 0:
                                winner = device_name_1 if fps_avg_1 > fps_avg_2 else device_name_2
                                winner_fps = max(fps_avg_1, fps_avg_2)
                                loser_fps = min(fps_avg_1, fps_avg_2)
                                fps_advantage = ((winner_fps - loser_fps) / loser_fps) * 100
                                
                                st.markdown(f"""
                                <div style="text-align: center; padding: 1rem; background: linear-gradient(45deg, #667eea, #764ba2); 
                                            border-radius: 10px; margin: 1rem 0;">
                                    <h3 style="color: white; margin: 0;">🏆 FPS Winner: {winner}</h3>
                                    <p style="color: white; margin: 0;">{fps_advantage:.1f}% better average FPS performance</p>
                                </div>
                                """, unsafe_allow_html=True)
                
                else:
                    st.warning("⚠️ Please select at least one metric for comparison")
            else:
                st.error("❌ No common columns found between the two datasets")
        
        elif datasets_loaded == 1:
            st.info("📤 Please upload both CSV files to enable comparison mode")
        
        elif uploaded_file_1 is not None or uploaded_file_2 is not None:
            st.info("📊 Loading datasets...")
    
    else:
        # Single CSV Analysis Mode
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
                    if analyzer.load_csv_data(file_data, uploaded_file.name, 1):
                        
                        if analyzer.original_data is not None and analyzer.numeric_columns:
                            
                            # Dynamic Column Selection - SIMPLIFIED TO FPS & CPU ONLY
                            st.subheader("📊 Gaming Performance Chart")
                            st.info("🎯 **Focus Mode**: Automatically displays FPS and CPU Usage for optimal gaming analysis")
                            
                            chart_config = {
                                'game_title': game_title,
                                'game_settings': game_settings,
                                'game_mode': game_mode,
                                'smartphone_name': smartphone_name
                            } enumerate(selected_columns):
                                        col_display = analyzer.get_column_display_name(col)
                                        suggested_color = analyzer.get_column_color_suggestion(col, 1)
                                        
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
                                if enable_advanced_outlier and outlier_columns:
                                    with st.spinner('🔧 Applying advanced outlier removal...'):
                                        analyzer.remove_outliers(advanced_outlier_method, advanced_outlier_threshold, outlier_columns, 1)
                                        processed = True
                                
                                # Priority 2: Quick outlier removal from sidebar (if enabled and no advanced processing)
                                elif enable_outlier_removal and not processed:
                                    with st.spinner('🔧 Removing outliers from sidebar settings...'):
                                        analyzer.remove_outliers(outlier_method, outlier_threshold, selected_columns, 1)
                                        processed = True
                                
                                # Default: Use raw data
                                if not processed:
                                    analyzer.processed_data = analyzer.original_data.copy()
                                    st.info("📊 **Raw Mode**: Using original data without processing")
                                
                                # Create chart (using single dataset method)
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
            # Statistics panel (single dataset)
            if uploaded_file is not None and analyzer.original_data is not None:
                st.subheader("📈 Performance Statistics")
                
                if 'selected_columns' in locals() and analyzer.original_data is not None:
                    # Find FPS and CPU columns for stats
                    fps_col = None
                    cpu_col = None
                    for col in analyzer.numeric_columns:
                        if 'fps' in col.lower() and fps_col is None:
                            fps_col = col
                        elif 'cpu' in col.lower() and cpu_col is None:
                            cpu_col = col
                    
                    available_columns = [col for col in [fps_col, cpu_col] if col is not None]
                    
                    if available_columns:
                        stats = analyzer.get_performance_stats(available_columns)
                    
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
                else:
                    st.info("📊 Select metrics from the main panel to see statistics")
            else:
                st.info("📤 Upload CSV file to see performance statistics")
    
    # Documentation
    with st.expander("📋 CSV Comparison Features"):
        st.markdown("""
        **🆚 NEW: CSV Comparison Mode**
        - **Side-by-side analysis**: Compare two different datasets
        - **Common metrics**: Only columns present in both files
        - **Visual comparison**: Solid vs dashed lines
        - **Performance insights**: Automatic winner detection
        - **Export options**: Save comparison charts
        
        **🎯 Use Cases:**
        - **Device comparison**: iPhone vs Android performance
        - **Settings testing**: Ultra vs High graphics
        - **Before/After**: Game optimization results
        - **Hardware analysis**: Different SoC performance
        - **Game versions**: Update impact analysis
        
        **📊 Comparison Features:**
        - **Percentage differences**: Clear improvement metrics
        - **Winner detection**: Automatic best performer
        - **Color coding**: Different shades per dataset
        - **Line styles**: Solid (Dataset 1) vs Dashed (Dataset 2)
        """)
    
    with st.expander("🎨 Comparison Chart Guide"):
        st.markdown("""
        **🎯 Reading Comparison Charts:**
        - **Solid lines**: First dataset (Device 1)
        - **Dashed lines**: Second dataset (Device 2)
        - **Higher is better**: FPS, Battery %
        - **Lower is better**: CPU %, Temperature
        
        **🌈 Color Strategy:**
        - **Bright colors**: First dataset
        - **Darker shades**: Second dataset
        - **Same metric family**: Similar color base
        
        **📊 Best Practices:**
        - **Sync time ranges**: Ensure similar test duration
        - **Same conditions**: Consistent game settings
        - **Clean data**: Apply outlier removal consistently
        - **Focus metrics**: Choose 2-4 key comparisons
        """)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #888; padding: 1rem;">
        🎮 Gaming Performance Analyzer v5.0 - Comparison Edition<br>
        <small>📊 Dynamic Columns • 🆚 CSV Comparison • 🎨 Multi-Axis Charts • 🎬 Video Export • 📈 Advanced Analytics</small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
