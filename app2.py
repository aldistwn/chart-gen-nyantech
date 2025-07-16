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
            st.write(f"📊 **Total rows in original CSV**: {len(df)}")
        
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
                    nan_count = numeric_data.isna().sum()
                    negative_count = (numeric_data < 0).sum()
                    st.write(f"✅ **Dataset {dataset_num} Numeric column**: `{col}`")
                    st.write(f"   - Range: {numeric_data.min():.1f} - {numeric_data.max():.1f}")
                    st.write(f"   - NaN values: {nan_count} ({nan_count/len(df)*100:.1f}%)")
                    st.write(f"   - Negative values: {negative_count} ({negative_count/len(df)*100:.1f}%)")
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
        
        # Create initial valid mask (all True)
        valid_mask = pd.Series([True] * len(processed_df))
        
        # Track reasons for removal
        removal_reasons = {
            'all_nan': 0,
            'negative_values': {},
            'infinite_values': {},
            'nan_values': {}
        }
        
        # First check: Remove rows where ALL numeric columns are NaN
        numeric_data_only = processed_df[numeric_cols]
        all_nan_mask = numeric_data_only.isna().all(axis=1)
        removal_reasons['all_nan'] = all_nan_mask.sum()
        valid_mask = valid_mask & ~all_nan_mask
        
        # Check each numeric column for invalid values
        for col in numeric_cols:
            col_series = processed_df[col]
            
            # Track negative values
            negative_mask = col_series < 0
            removal_reasons['negative_values'][col] = negative_mask.sum()
            
            # Track infinite values
            inf_mask = np.isinf(col_series)
            removal_reasons['infinite_values'][col] = inf_mask.sum()
            
            # Track NaN values
            nan_mask = pd.isna(col_series)
            removal_reasons['nan_values'][col] = nan_mask.sum()
            
            # Update valid mask - ONLY remove infinite values
            # Keep negative values and NaN for now (can be filtered later)
            valid_mask = valid_mask & ~inf_mask
        
        # Show detailed removal statistics
        if self.debug_mode:
            st.write(f"\n📊 **Data Quality Report for Dataset {dataset_num}:**")
            st.write(f"- Rows with all columns NaN: {removal_reasons['all_nan']}")
            
            if any(removal_reasons['negative_values'].values()):
                st.write("- Negative values by column:")
                for col, count in removal_reasons['negative_values'].items():
                    if count > 0:
                        st.write(f"  - {col}: {count} rows")
            
            if any(removal_reasons['infinite_values'].values()):
                st.write("- Infinite values by column:")
                for col, count in removal_reasons['infinite_values'].items():
                    if count > 0:
                        st.write(f"  - {col}: {count} rows")
            
            if any(removal_reasons['nan_values'].values()):
                st.write("- NaN values by column:")
                for col, count in removal_reasons['nan_values'].items():
                    if count > 0:
                        st.write(f"  - {col}: {count} rows ({count/len(df)*100:.1f}%)")
        
        invalid_count = len(df) - valid_mask.sum()
        
        if invalid_count > 0:
            st.warning(f"⚠️ Found {invalid_count} rows with infinite values or all columns NaN (removed)")
            st.info("💡 **Note**: Individual NaN values and negative values are KEPT for flexibility. Use outlier removal if needed.")
        
        # Create clean dataset
        clean_data = processed_df[valid_mask].copy().reset_index(drop=True)
        
        # Check if we have any data left
        if len(clean_data) == 0:
            st.error(f"❌ No valid data remaining after cleaning for dataset {dataset_num}")
            return False
        
        # Add time column
        clean_data['TimeMinutes'] = np.arange(len(clean_data)) / 60
        
        setattr(self, target_original_data, clean_data)
        
        if self.debug_mode:
            st.write(f"\n✅ **Dataset {dataset_num} Final data check**:")
            st.write(f"   - Original rows: {len(df)}")
            st.write(f"   - Valid rows after cleaning: {len(clean_data)}")
            st.write(f"   - Rows removed: {len(df) - len(clean_data)} ({(len(df) - len(clean_data))/len(df)*100:.1f}%)")
            st.write(f"   - Numeric columns: {numeric_cols}")
            st.write(f"   - All columns: {list(clean_data.columns)}")
            
            # Show sample of data
            with st.expander("🔍 View sample data (first 5 rows)"):
                st.dataframe(clean_data.head())
        
        st.success(f"✅ Dataset {dataset_num} loaded successfully using delimiter '{delimiter}' and encoding '{encoding}'")
        
        return True
    
    def create_performance_chart(self, config):
        """Create performance chart with FPS and CPU focus (dual y-axis)"""
        try:
            data = self.processed_data if self.processed_data is not None else self.original_data
            
            if data is None:
                st.error("❌ No data available for chart generation")
                return None
            
            # Debug: Check data shape
            if self.debug_mode:
                st.write(f"📊 **Chart Debug - Data shape**: {data.shape}")
                st.write(f"📊 **Available columns**: {list(data.columns)}")
            
            # Find FPS and CPU columns with more flexible matching
            fps_col = None
            cpu_col = None
            
            for col in self.numeric_columns:
                col_lower = col.lower()
                # More flexible FPS matching
                if fps_col is None and any(term in col_lower for term in ['fps', 'frame', 'framerate']):
                    fps_col = col
                    if self.debug_mode:
                        st.write(f"📊 Found FPS column: {col}")
                # More flexible CPU matching
                elif cpu_col is None and any(term in col_lower for term in ['cpu', 'processor']):
                    cpu_col = col
                    if self.debug_mode:
                        st.write(f"📊 Found CPU column: {col}")
            
            if not fps_col and not cpu_col:
                st.error("❌ No FPS or CPU columns found in the data")
                st.info("💡 Looking for columns containing: 'fps', 'frame', 'cpu', 'processor'")
                st.write("Available numeric columns:", self.numeric_columns)
                return None
            
            # CREATE FILTERED DATAFRAME WITH ONLY FPS AND CPU
            columns_to_keep = ['TimeMinutes']
            if fps_col:
                columns_to_keep.append(fps_col)
            if cpu_col:
                columns_to_keep.append(cpu_col)
            
            # Filter data to only include needed columns
            filtered_data = data[columns_to_keep].copy()
            
            # Remove rows where BOTH FPS and CPU are NaN (if both columns exist)
            if fps_col and cpu_col:
                valid_mask = ~(filtered_data[fps_col].isna() & filtered_data[cpu_col].isna())
                filtered_data = filtered_data[valid_mask].reset_index(drop=True)
                if self.debug_mode:
                    st.write(f"📊 Removed {(~valid_mask).sum()} rows where both FPS and CPU are NaN")
            
            # Update time column after filtering
            filtered_data['TimeMinutes'] = np.arange(len(filtered_data)) / 60
            
            if self.debug_mode:
                st.write(f"📊 **Filtered data for chart**: {filtered_data.shape}")
                st.write(f"📊 **Using only columns**: {list(filtered_data.columns)}")
            
            # Create figure with dark theme
            plt.style.use('dark_background')
            fig, ax1 = plt.subplots(figsize=(16, 9))
            fig.patch.set_facecolor('#0E1117')
            
            time_data = filtered_data['TimeMinutes']
            
            # Check time data validity
            if len(time_data) == 0:
                st.error("❌ No valid data remaining after filtering")
                return None
            
            # Plot FPS on primary axis (left)
            if fps_col and fps_col in filtered_data.columns:
                fps_data = filtered_data[fps_col].fillna(0)  # Fill NaN with 0 for plotting
                fps_color = '#00D4FF'  # Cyan for FPS
                
                if self.debug_mode:
                    st.write(f"📊 Plotting FPS data: {len(fps_data)} points")
                    st.write(f"   - Min: {fps_data.min():.1f}, Max: {fps_data.max():.1f}, Avg: {fps_data.mean():.1f}")
                
                line1 = ax1.plot(time_data, fps_data, color=fps_color, linewidth=3, 
                               label='FPS', alpha=0.9, zorder=2)
                
                ax1.set_xlabel('Time (minutes)', fontsize=14, color='white', fontweight='bold')
                ax1.set_ylabel('FPS', fontsize=14, color=fps_color, fontweight='bold')
                ax1.tick_params(axis='y', labelcolor=fps_color, labelsize=12, colors=fps_color)
                ax1.tick_params(axis='x', colors='white', labelsize=12)
                
                # Set FPS limits (0 to max + 10%)
                fps_max = fps_data.max() * 1.1 if not pd.isna(fps_data.max()) and fps_data.max() > 0 else 120
                ax1.set_ylim(0, fps_max)
                
                # Add horizontal reference lines for FPS
                ax1.axhline(y=60, color='green', linestyle='--', alpha=0.5, linewidth=1)
                ax1.axhline(y=30, color='orange', linestyle='--', alpha=0.5, linewidth=1)
                
                # Safe text positioning
                text_x_pos = time_data.iloc[-1] * 0.02 if len(time_data) > 0 else 0.02
                ax1.text(text_x_pos, 62, '60 FPS', color='green', fontsize=10)
                ax1.text(text_x_pos, 32, '30 FPS', color='orange', fontsize=10)
            
            # Plot CPU on secondary axis (right)
            if cpu_col and cpu_col in filtered_data.columns:
                ax2 = ax1.twinx()
                cpu_data = filtered_data[cpu_col].fillna(0)  # Fill NaN with 0 for plotting
                cpu_color = '#FF6B35'  # Orange for CPU
                
                if self.debug_mode:
                    st.write(f"📊 Plotting CPU data: {len(cpu_data)} points")
                    st.write(f"   - Min: {cpu_data.min():.1f}, Max: {cpu_data.max():.1f}, Avg: {cpu_data.mean():.1f}")
                
                line2 = ax2.plot(time_data, cpu_data, color=cpu_color, linewidth=3, 
                               label='CPU Usage (%)', alpha=0.9, zorder=1)
                
                ax2.set_ylabel('CPU Usage (%)', fontsize=14, color=cpu_color, fontweight='bold')
                ax2.tick_params(axis='y', labelcolor=cpu_color, labelsize=12, colors=cpu_color)
                
                # Set CPU limits (0 to 100%)
                ax2.set_ylim(0, 100)
                
                # Add horizontal reference line for CPU
                ax2.axhline(y=80, color='red', linestyle='--', alpha=0.3, linewidth=1)
                text_x_pos = time_data.iloc[-1] * 0.02 if len(time_data) > 0 else 0.02
                ax2.text(text_x_pos, 82, '80% CPU', color='red', fontsize=10)
            
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
            if self.debug_mode:
                st.exception(e)
            return None
    
    # ... (rest of the methods remain the same)
    
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
                if not pd.isna(fps_data1.max()) and not pd.isna(fps_data2.max()):
                    fps_max = max(fps_data1.max(), fps_data2.max()) * 1.1
                else:
                    fps_max = 120
