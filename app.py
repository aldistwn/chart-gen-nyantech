import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io
from datetime import datetime
from scipy.signal import savgol_filter

# Page configuration
st.set_page_config(
    page_title="🎮 Gaming Chart Generator",
    page_icon="🎮",
    layout="wide"
)

class FinalOptimizedGamingChartGenerator:
    def __init__(self):
        self.original_data = None
        self.processed_data = None
        self.removed_indices = []
        self.column_mapping = {}
    
    def validate_data(self, data, column_name):
        """Validate data quality and provide feedback"""
        try:
            numeric_data = pd.to_numeric(data, errors='coerce')
            valid_count = numeric_data.notna().sum()
            total_count = len(data)
            valid_percentage = (valid_count / total_count) * 100
            
            if valid_percentage < 80:
                st.warning(f"⚠️ {column_name}: Low data quality ({valid_percentage:.1f}% valid)")
            elif valid_percentage < 95:
                st.info(f"ℹ️ {column_name}: Good data quality ({valid_percentage:.1f}% valid)")
            else:
                st.success(f"✅ {column_name}: Excellent data quality ({valid_percentage:.1f}% valid)")
            
            return numeric_data, valid_percentage
            
        except Exception as e:
            st.error(f"❌ {column_name} validation failed: {e}")
            return data, 0
    
    def load_csv_data(self, uploaded_file):
        """Enhanced CSV loading with strict column requirements"""
        try:
            # Try different delimiters
            delimiters = [',', ';', '\t', '|']
            
            for delimiter in delimiters:
                try:
                    uploaded_file.seek(0)
                    self.original_data = pd.read_csv(uploaded_file, delimiter=delimiter)
                    if len(self.original_data.columns) > 1:
                        st.success(f"✅ CSV parsed successfully with delimiter: '{delimiter}'")
                        break
                except:
                    continue
            
            if self.original_data is None or len(self.original_data.columns) <= 1:
                st.error("❌ Cannot parse CSV file. Please check format.")
                return False
            
            # Show basic info
            st.info(f"📊 Dataset: {len(self.original_data)} rows × {len(self.original_data.columns)} columns")
            
            # STRICT column detection - only exact matches
            columns = list(self.original_data.columns)
            
            # Debug: Show all columns
            with st.expander("🔍 DEBUG - All detected columns"):
                for i, col in enumerate(columns):
                    st.write(f"Column {i+1}: '{col}'")
            
            # Check for EXACT required columns
            if 'FPS' not in columns:
                st.error("❌ Required 'FPS' column not found!")
                st.info("💡 Please ensure your CSV has a column labeled exactly: **FPS**")
                st.info(f"Available columns: {', '.join(columns)}")
                return False
            
            if 'CPU(%)' not in columns:
                st.error("❌ Required 'CPU(%)' column not found!")
                st.info("💡 Please ensure your CSV has a column labeled exactly: **CPU(%)**")
                st.info(f"Available columns: {', '.join(columns)}")
                return False
            
            # SUCCESS: Both columns found
            st.success("✅ FPS column detected: **FPS**")
            st.success("✅ CPU column detected: **CPU(%)**")
            
            # Validate and extract data
            fps_data, fps_quality = self.validate_data(self.original_data['FPS'], "FPS")
            cpu_data, cpu_quality = self.validate_data(self.original_data['CPU(%)'], "CPU")
            
            if fps_quality < 50 or cpu_quality < 50:
                st.error("❌ Data quality too low. Please check your data.")
                return False
            
            # DEBUG: Show exact data ranges
            st.markdown("### 📊 Data Validation & Preview")
            
            # Critical debug: Check for anomalous values
            high_fps_count = len(fps_data[fps_data > 70])
            very_high_fps_count = len(fps_data[fps_data > 80])
            
            if high_fps_count > 0:
                st.warning(f"⚠️ Found {high_fps_count} FPS values > 70")
            if very_high_fps_count > 0:
                st.error(f"🚨 Found {very_high_fps_count} FPS values > 80 - This is unusual!")
            
            # Create standardized dataset
            self.original_data['FPS'] = fps_data
            self.original_data['CPU(%)'] = cpu_data
            self.original_data['TimeMinutes'] = [i / 60 for i in range(len(self.original_data))]
            
            # Show data preview
            col1, col2 = st.columns(2)
            with col1:
                st.metric("📈 FPS Range", f"{fps_data.min():.1f} - {fps_data.max():.1f}")
                st.metric("📊 FPS Average", f"{fps_data.mean():.1f}")
            with col2:
                st.metric("🖥️ CPU Range", f"{cpu_data.min():.1f}% - {cpu_data.max():.1f}%")
                st.metric("📊 CPU Average", f"{cpu_data.mean():.1f}%")
            
            # Data preview table
            with st.expander("📋 Data Preview (First 10 rows)"):
                preview_df = pd.DataFrame({
                    'Time (min)': self.original_data['TimeMinutes'].head(10),
                    'FPS': self.original_data['FPS'].head(10),
                    'CPU (%)': self.original_data['CPU(%)'].head(10)
                })
                st.dataframe(preview_df, use_container_width=True)
            
            return True
            
        except Exception as e:
            st.error(f"❌ Error loading CSV: {e}")
            return False
    
    def remove_fps_outliers_optimized(self, sensitivity='moderate'):
        """Optimized outlier removal with proper index tracking"""
        try:
            if self.original_data is None:
                return False
            
            fps_data = self.original_data['FPS'].dropna()
            if len(fps_data) < 10:
                st.warning("⚠️ Not enough data points for outlier removal")
                return False
            
            # Calculate thresholds
            percentile_1 = fps_data.quantile(0.01)
            percentile_5 = fps_data.quantile(0.05)
            
            thresholds = {
                'conservative': percentile_1,
                'moderate': percentile_1 * 1.1,
                'aggressive': percentile_5
            }
            threshold = thresholds.get(sensitivity, percentile_1)
            
            # Get indices to keep (CONSISTENT across all columns)
            keep_mask = fps_data >= threshold
            keep_indices = fps_data[keep_mask].index.tolist()
            removed_indices = fps_data[~keep_mask].index.tolist()
            
            # Store for tracking
            self.removed_indices = removed_indices
            
            # Create processed dataset with SAME indices for all columns
            self.processed_data = pd.DataFrame({
                'FPS': self.original_data.loc[keep_indices, 'FPS'].values,
                'CPU(%)': self.original_data.loc[keep_indices, 'CPU(%)'].values,
                'TimeMinutes': [i / 60 for i in range(len(keep_indices))]
            })
            
            # DEBUG: Verify processed data ranges
            st.info(f"🔍 After outlier removal: FPS range {self.processed_data['FPS'].min():.1f} - {self.processed_data['FPS'].max():.1f}")
            
            # Check for any anomalous values after processing
            high_fps_after = len(self.processed_data['FPS'][self.processed_data['FPS'] > 70])
            if high_fps_after > 0:
                st.warning(f"⚠️ Still {high_fps_after} values > 70 after outlier removal")
            
            # Feedback
            removal_count = len(removed_indices)
            removal_pct = (removal_count / len(fps_data)) * 100
            
            st.success(f"✅ Removed {removal_count} frames ({removal_pct:.1f}%)")
            st.info(f"📊 Threshold: {threshold:.1f} FPS | Kept: {len(keep_indices)} frames")
            
            return True
            
        except Exception as e:
            st.error(f"❌ Outlier removal failed: {e}")
            return False
    
    def apply_processing(self, fps_window=5, fps_poly=1, cpu_window=5, cpu_poly=1,
                        enable_fps_smooth=False, enable_cpu_smooth=False,
                        enable_outlier_removal=False, outlier_sensitivity='moderate'):
        """Streamlined processing pipeline with debugging"""
        
        st.info(f"🔍 Original data range: FPS {self.original_data['FPS'].min():.1f} - {self.original_data['FPS'].max():.1f}")
        
        # Initialize with original data
        if enable_outlier_removal:
            if not self.remove_fps_outliers_optimized(outlier_sensitivity):
                st.warning("⚠️ Outlier removal failed, using original data")
                self.processed_data = self.original_data.copy()
        else:
            self.processed_data = self.original_data.copy()
        
        # Apply smoothing filters
        try:
            # FPS Smoothing
            if enable_fps_smooth:
                data_length = len(self.processed_data)
                fps_window = min(max(fps_window, 5), data_length)
                if fps_window % 2 == 0:
                    fps_window -= 1
                
                if data_length >= fps_window:
                    # DEBUG: Before smoothing
                    before_max = self.processed_data['FPS'].max()
                    
                    smoothed_fps = savgol_filter(
                        self.processed_data['FPS'],
                        window_length=fps_window,
                        polyorder=min(fps_poly, fps_window-1)
                    )
                    self.processed_data['FPS_Smooth'] = smoothed_fps
                    
                    # DEBUG: After smoothing
                    after_max = smoothed_fps.max()
                    st.info(f"🎯 FPS smoothed: {before_max:.1f} → {after_max:.1f} max")
                    
                    # Check for smoothing anomalies
                    if after_max > before_max + 5:  # If smoothing increases max by more than 5
                        st.warning(f"⚠️ Smoothing increased max FPS significantly: {before_max:.1f} → {after_max:.1f}")
                else:
                    self.processed_data['FPS_Smooth'] = self.processed_data['FPS']
            else:
                self.processed_data['FPS_Smooth'] = self.processed_data['FPS']
            
            # CPU Smoothing
            if enable_cpu_smooth:
                data_length = len(self.processed_data)
                cpu_window = min(max(cpu_window, 5), data_length)
                if cpu_window % 2 == 0:
                    cpu_window -= 1
                
                if data_length >= cpu_window:
                    smoothed_cpu = savgol_filter(
                        self.processed_data['CPU(%)'],
                        window_length=cpu_window,
                        polyorder=min(cpu_poly, cpu_window-1)
                    )
                    self.processed_data['CPU_Smooth'] = smoothed_cpu
                    st.info(f"🖥️ CPU smoothed (window: {cpu_window}, poly: {cpu_poly})")
                else:
                    self.processed_data['CPU_Smooth'] = self.processed_data['CPU(%)']
            else:
                self.processed_data['CPU_Smooth'] = self.processed_data['CPU(%)']
            
            # FINAL DEBUG: Show final data ranges
            if 'FPS_Smooth' in self.processed_data:
                final_fps_max = self.processed_data['FPS_Smooth'].max()
                st.info(f"🔍 Final processed FPS max: {final_fps_max:.1f}")
                
                if final_fps_max > 80:
                    st.error(f"🚨 ANOMALY DETECTED: Final FPS max is {final_fps_max:.1f} - This should not happen!")
            
            return True
            
        except Exception as e:
            st.error(f"❌ Processing failed: {e}")
            return False
    
    def create_optimized_chart(self, game_title, game_settings, game_mode, smartphone_name,
                             fps_color, cpu_color, show_original=True, show_processed=True):
        """Simplified and consistent chart creation"""
        
        # Create figure
        fig, ax1 = plt.subplots(figsize=(19.2, 10.8))
        fig.patch.set_facecolor('none')
        
        # Setup axes
        ax1.set_xlabel('Time (minutes)', fontsize=12, fontweight='bold', color='black')
        ax1.set_ylabel('FPS', color='black', fontsize=12, fontweight='bold')
        ax1.tick_params(axis='y', labelcolor='black', labelsize=10)
        ax1.tick_params(axis='x', labelcolor='black', labelsize=10)
        
        ax2 = ax1.twinx()
        ax2.set_ylabel('CPU Usage (%)', color='black', fontsize=12, fontweight='bold')
        ax2.tick_params(axis='y', labelcolor='black', labelsize=10)
        ax2.set_ylim(0, 100)
        
        # Determine primary dataset
        primary_data = self.processed_data if self.processed_data is not None else self.original_data
        
        # DEBUG: Log what data is being used for chart
        if 'FPS_Smooth' in primary_data:
            chart_fps_max = primary_data['FPS_Smooth'].max()
            st.info(f"🔍 Chart using FPS_Smooth with max: {chart_fps_max:.1f}")
        else:
            chart_fps_max = primary_data['FPS'].max()
            st.info(f"🔍 Chart using FPS with max: {chart_fps_max:.1f}")
        
        # Plot original data (if requested and different from processed)
        if show_original and self.processed_data is not None and len(self.processed_data) != len(self.original_data):
            orig_length = len(self.processed_data)
            orig_time = self.original_data['TimeMinutes'][:orig_length]
            orig_fps = self.original_data['FPS'][:orig_length]
            orig_cpu = self.original_data['CPU(%)'][:orig_length]
            
            ax1.plot(orig_time, orig_fps, color=fps_color, linewidth=1, 
                    alpha=0.3, linestyle='--', zorder=2)
            ax2.plot(orig_time, orig_cpu, color=cpu_color, linewidth=1,
                    alpha=0.3, linestyle='--', zorder=1)
        
        # Plot processed/main data
        if show_processed and self.processed_data is not None:
            time_data = self.processed_data['TimeMinutes']
            fps_data = self.processed_data['FPS_Smooth']
            cpu_data = self.processed_data['CPU_Smooth']
            
            # Main lines with labels
            ax1.plot(time_data, fps_data, color=fps_color, linewidth=2.5,
                    label='FPS', alpha=0.9, zorder=4)
            ax2.plot(time_data, cpu_data, color=cpu_color, linewidth=2.5,
                    label='CPU', alpha=0.9, zorder=3)
        elif show_original:
            # Fallback to original data
            ax1.plot(self.original_data['TimeMinutes'], self.original_data['FPS'],
                    color=fps_color, linewidth=2.5, label='FPS', alpha=0.9)
            ax2.plot(self.original_data['TimeMinutes'], self.original_data['CPU(%)'],
                    color=cpu_color, linewidth=2.5, label='CPU', alpha=0.9)
        
        # Set limits - CRITICAL: Use actual data max, not inflated
        if 'FPS_Smooth' in primary_data:
            fps_max = max(primary_data['FPS_Smooth']) * 1.1
        else:
            fps_max = max(primary_data['FPS']) * 1.1
        
        ax1.set_ylim(0, fps_max)
        
        # DEBUG: Log axis limits
        st.info(f"🔍 Chart Y-axis set to: 0 - {fps_max:.1f}")
        
        # Title and styling
        title_text = f"{game_title}\n{game_settings}\n{game_mode}"
        plt.suptitle(title_text, fontsize=24, fontweight='bold', y=0.98, color='white')
        plt.subplots_adjust(top=0.85)
        
        # Grid and styling
        ax1.grid(True, alpha=0.3, linestyle='--', color='white')
        ax1.set_facecolor('none')
        
        # Legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        
        if lines1 or lines2:
            legend_lines = [plt.Line2D([0], [0], color='none')] + lines1 + lines2
            legend_labels = [smartphone_name] + labels1 + labels2
            
            legend = ax1.legend(legend_lines, legend_labels,
                              loc='upper right', framealpha=0.8, fancybox=True,
                              facecolor='white', edgecolor='gray')
            
            for i, text in enumerate(legend.get_texts()):
                text.set_color('black')
                text.set_horizontalalignment('left')
                if i == 0:
                    text.set_fontweight('bold')
                    text.set_fontsize(11)
                else:
                    text.set_fontsize(10)
        
        # Hide spines
        for spine in ax1.spines.values():
            spine.set_visible(False)
        for spine in ax2.spines.values():
            spine.set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def get_statistics(self, use_processed=True):
        """Get performance statistics"""
        if self.original_data is None:
            return {}
        
        # Choose dataset
        if use_processed and self.processed_data is not None:
            fps_data = self.processed_data['FPS_Smooth'].dropna()
            cpu_data = self.processed_data['CPU_Smooth'].dropna()
            data_type = "Processed"
            duration = len(self.processed_data) / 60
        else:
            fps_data = self.original_data['FPS'].dropna()
            cpu_data = self.original_data['CPU(%)'].dropna()
            data_type = "Original"
            duration = len(self.original_data) / 60
        
        # Performance grading
        avg_fps = fps_data.mean()
        if avg_fps >= 90:
            grade = "🏆 Excellent (90+ FPS)"
        elif avg_fps >= 60:
            grade = "✅ Good (60+ FPS)"
        elif avg_fps >= 30:
            grade = "⚠️ Playable (30+ FPS)"
        else:
            grade = "❌ Poor (<30 FPS)"
        
        return {
            'grade': grade,
            'data_type': data_type,
            'duration': round(duration, 1),
            'avg_fps': round(avg_fps, 1),
            'min_fps': round(fps_data.min(), 1),
            'max_fps': round(fps_data.max(), 1),
            'avg_cpu': round(cpu_data.mean(), 1),
            'max_cpu': round(cpu_data.max(), 1),
            'fps_above_60': round((len(fps_data[fps_data >= 60]) / len(fps_data)) * 100, 1),
            'frame_drops': len(fps_data[fps_data < 30]),
            'removed_frames': len(self.removed_indices) if hasattr(self, 'removed_indices') else 0
        }
    
    def generate_processed_csv(self, game_title):
        """Generate CSV file with processed FPS and CPU data"""
        if self.processed_data is None:
            return None
        
        # Create clean DataFrame for export
        export_data = pd.DataFrame({
            'Time_Minutes': self.processed_data['TimeMinutes'],
            'FPS': self.processed_data['FPS_Smooth'] if 'FPS_Smooth' in self.processed_data else self.processed_data['FPS'],
            'CPU_Percent': self.processed_data['CPU_Smooth'] if 'CPU_Smooth' in self.processed_data else self.processed_data['CPU(%)']
        })
        
        # Round to reasonable precision
        export_data['Time_Minutes'] = export_data['Time_Minutes'].round(3)
        export_data['FPS'] = export_data['FPS'].round(1)
        export_data['CPU_Percent'] = export_data['CPU_Percent'].round(1)
        
        # Convert to CSV
        csv_buffer = io.StringIO()
        export_data.to_csv(csv_buffer, index=False)
        csv_content = csv_buffer.getvalue()
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{game_title.replace(' ', '_')}_processed_data_{timestamp}.csv"
        
        return csv_content, filename

def main():
    # Header
    st.title("🎮 Final Gaming Chart Generator")
    st.markdown("Transform gaming logs with **strict column detection** and **CSV export**")
    
    # Initialize
    generator = FinalOptimizedGamingChartGenerator()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("🎮 Game Configuration")
        game_title = st.text_input("Game Title", value="MOBILE LEGENDS")
        game_settings = st.text_input("Graphics Settings", value="ULTRA - 120 FPS")
        game_mode = st.text_input("Performance Mode", value="BOOST MODE")
        smartphone_name = st.text_input("Smartphone Model", value="iPhone 15 Pro Max")
        
        st.header("🎨 Chart Colors")
        fps_color = st.color_picker("FPS Color", "#4A90E2")  # Blue default
        cpu_color = st.color_picker("CPU Color", "#FF6600")   # Orange default
        
        st.header("📊 Display Options")
        show_original = st.checkbox("Show Original Data", value=False,
                                   help="Show original data as faded background")
        show_processed = st.checkbox("Show Processed Data", value=True,
                                    help="Show processed/smoothed data as main line")
        
        st.header("🔧 Data Processing")
        
        # Outlier Removal
        enable_outlier_removal = st.toggle("🚫 Remove Worst FPS Frames", value=False,
                                          help="Remove only the worst performing frames")
        
        if enable_outlier_removal:
            outlier_sensitivity = st.select_slider(
                "Removal Sensitivity",
                options=['conservative', 'moderate', 'aggressive'],
                value='moderate'
            )
        else:
            outlier_sensitivity = 'moderate'
        
        # Smoothing
        col1, col2 = st.columns(2)
        with col1:
            enable_fps_smooth = st.toggle("🎯 FPS Smoothing", value=False)
            if enable_fps_smooth:
                fps_window = st.slider("FPS Window", 5, 21, 7, step=2)
                fps_poly = st.slider("FPS Poly", 1, 3, 1)
            else:
                fps_window, fps_poly = 7, 1
        
        with col2:
            enable_cpu_smooth = st.toggle("🖥️ CPU Smoothing", value=False)
            if enable_cpu_smooth:
                cpu_window = st.slider("CPU Window", 5, 21, 7, step=2)
                cpu_poly = st.slider("CPU Poly", 1, 3, 1)
            else:
                cpu_window, cpu_poly = 7, 1
    
    # File upload
    st.header("📁 Upload Gaming Log CSV")
    uploaded_file = st.file_uploader("Drop your CSV file here", type=['csv'])
    
    if uploaded_file:
        # Load and validate data
        if generator.load_csv_data(uploaded_file):
            
            # Process data
            with st.spinner('🔧 Processing data...'):
                if generator.apply_processing(
                    fps_window, fps_poly, cpu_window, cpu_poly,
                    enable_fps_smooth, enable_cpu_smooth,
                    enable_outlier_removal, outlier_sensitivity
                ):
                    st.success("✅ Data processing completed!")
                else:
                    st.warning("⚠️ Processing failed, using original data")
            
            # Show quick stats
            display_data = generator.processed_data if generator.processed_data is not None else generator.original_data
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📊 Data Points", f"{len(display_data):,}")
            with col2:
                st.metric("⏱️ Duration", f"{len(display_data)/60:.1f} min")
            with col3:
                st.metric("🎯 Avg FPS", f"{display_data['FPS'].mean():.1f}")
            with col4:
                st.metric("🖥️ Avg CPU", f"{display_data['CPU(%)'].mean():.1f}%")
            
            # Generate chart
            st.header("📊 Performance Chart")
            
            if not show_original and not show_processed:
                st.warning("⚠️ Please select at least one display option")
            else:
                with st.spinner('🎨 Creating chart...'):
                    chart_fig = generator.create_optimized_chart(
                        game_title, game_settings, game_mode, smartphone_name,
                        fps_color, cpu_color, show_original, show_processed
                    )
                    st.pyplot(chart_fig)
            
            # Performance statistics
            stats = generator.get_statistics(use_processed=True)
            
            st.header(f"📈 Performance Analysis ({stats['data_type']} Data)")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Performance Grade", stats['grade'])
            with col2:
                st.metric("FPS Range", f"{stats['min_fps']}-{stats['max_fps']}")
            with col3:
                st.metric("60+ FPS Time", f"{stats['fps_above_60']}%")
            with col4:
                st.metric("Removed Frames", stats['removed_frames'])
            
            # Export section
            st.header("💾 Export Results")
            
            col1, col2 = st.columns(2)
            
            # PNG Export
            with col1:
                st.subheader("📸 Chart Export")
                if 'chart_fig' in locals():
                    img_buffer = io.BytesIO()
                    chart_fig.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight',
                                     facecolor='none', edgecolor='none', transparent=True)
                    img_buffer.seek(0)
                    
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    png_filename = f"{game_title.replace(' ', '_')}_chart_{timestamp}.png"
                    
                    st.download_button(
                        label="📸 Download Chart (PNG)",
                        data=img_buffer.getvalue(),
                        file_name=png_filename,
                        mime="image/png",
                        use_container_width=True
                    )
            
            # CSV Export
            with col2:
                st.subheader("📄 Data Export")
                if generator.processed_data is not None:
                    csv_content, csv_filename = generator.generate_processed_csv(game_title)
                    
                    if csv_content:
                        st.download_button(
                            label="📄 Download Processed CSV",
                            data=csv_content,
                            file_name=csv_filename,
                            mime="text/csv",
                            use_container_width=True
                        )
                        
                        # Show preview of export data
                        with st.expander("👀 Preview Export Data"):
                            preview_df = pd.read_csv(io.StringIO(csv_content))
                            st.dataframe(preview_df.head(10))
                            st.info(f"📊 Export contains {len(preview_df)} rows with Time, FPS, and CPU data")
    
    else:
        # Help section
        st.info("📤 Upload your gaming log CSV to get started!")
        
        with st.expander("📋 Required CSV Format"):
            st.markdown("""
            **STRICT Requirements:**
            - Must have a column labeled exactly: **FPS**
            - Must have a column labeled exactly: **CPU(%)**
            
            **Example CSV structure:**
            ```
            FPS,CPU(%),JANK,BigJANK
            60,45.2,0,0
            58,48.1,1,0
            62,42.8,0,0
            ```
            
            **Important Notes:**
            - Column names are case-sensitive
            - No variations accepted (fps, Fps, cpu usage, etc.)
            - Exact spelling required: FPS and CPU(%)
            """)
        
        with st.expander("💾 Export Features"):
            st.markdown("""
            **Chart Export (PNG):**
            - High-resolution (300 DPI)
            - Transparent background
            - Professional gaming chart format
            - Ready for presentations/reports
            
            **Data Export (CSV):**
            - Clean processed FPS and CPU data
            - Time_Minutes, FPS, CPU_Percent columns
            - Rounded values for clarity
            - Ready for further analysis
            - Compatible with Excel/Google Sheets
            """)

if __name__ == "__main__":
    main()
