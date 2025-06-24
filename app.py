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

class GamingChartGenerator:
    def __init__(self):
        self.data = None
        self.smoothed_data = None
    
    def load_csv_data(self, uploaded_file):
        """Load CSV with smart detection"""
        try:
            # Try different delimiters
            delimiters = [',', ';', '\t', '|']
            
            for delimiter in delimiters:
                try:
                    uploaded_file.seek(0)
                    self.data = pd.read_csv(uploaded_file, delimiter=delimiter)
                    if len(self.data.columns) > 1:
                        break
                except:
                    continue
            
            if self.data is None or len(self.data.columns) <= 1:
                st.error("❌ Cannot parse CSV file. Please check format.")
                return False
            
            # Smart column detection
            columns = list(self.data.columns)
            fps_col = None
            cpu_col = None
            
            # Find FPS column
            for col in columns:
                if 'fps' in col.lower():
                    fps_col = col
                    break
            
            # Find CPU column  
            for col in columns:
                if 'cpu' in col.lower() and '%' in col:
                    cpu_col = col
                    break
            
            if not fps_col or not cpu_col:
                st.error("❌ Required columns not found. Need FPS and CPU(%) data.")
                st.info(f"Available columns: {', '.join(columns)}")
                return False
            
            # Standardize column names
            self.data['FPS'] = self.data[fps_col]
            self.data['CPU(%)'] = self.data[cpu_col]
            
            # Create time index
            self.data['TimeMinutes'] = [i / 60 for i in range(len(self.data))]
            
            return True
            
        except Exception as e:
            st.error(f"❌ Error loading CSV: {e}")
            return False
    
    def remove_fps_outliers(self, data, sensitivity='moderate'):
        """Remove FPS outliers - ONLY removes bottom 1% worst frame drops"""
        try:
            # Convert to pandas Series if not already
            if not isinstance(data, pd.Series):
                data = pd.Series(data)
            
            # Remove NaN values for calculation
            clean_data = data.dropna()
            if len(clean_data) < 10:  # Need minimum data points
                st.warning("⚠️ Not enough data points for FPS outlier removal")
                return data
            
            # Calculate percentiles for 1% worst frames
            percentile_1 = clean_data.quantile(0.01)  # Bottom 1%
            percentile_5 = clean_data.quantile(0.05)  # Bottom 5% for reference
            
            # Set threshold based on sensitivity - ONLY remove bottom frames
            thresholds = {
                'conservative': percentile_1,      # Remove only bottom 1%
                'moderate': percentile_1 * 1.1,   # Remove bottom 1% + slightly above
                'aggressive': percentile_5        # Remove bottom 5%
            }
            threshold = thresholds.get(sensitivity, percentile_1)
            
            # Only remove frames BELOW threshold (worst performance)
            outlier_mask = clean_data < threshold
            outlier_count = outlier_mask.sum()
            
            # Show FPS outlier detection info
            st.info(f"🎯 FPS Analysis: Min={clean_data.min():.1f}, Max={clean_data.max():.1f}")
            st.info(f"📊 1% Percentile: {percentile_1:.1f} FPS")
            st.info(f"📊 Removal Threshold: {threshold:.1f} FPS")
            st.info(f"🧹 Worst frames found: {outlier_count} ({outlier_count/len(clean_data)*100:.1f}%)")
            
            if outlier_count == 0:
                st.success("✅ No bad FPS frames detected - performance is consistent!")
                return data
            
            # Remove only the worst frames
            valid_mask = ~outlier_mask.reindex(data.index, fill_value=False)
            result = data[valid_mask].copy()
            
            # Reset index untuk membuat data berurutan tanpa gap
            result = result.reset_index(drop=True)
            
            # Show results
            original_length = len(data)
            new_length = len(result)
            removed_count = original_length - new_length
            
            st.success(f"✅ Removed {removed_count} worst FPS frames ({removed_count/original_length*100:.1f}%)")
            st.info(f"📈 Data points: {original_length} → {new_length}")
            st.info(f"🎯 New FPS range: {result.min():.1f} - {result.max():.1f}")
            
            return result
            
        except Exception as e:
            st.error(f"❌ FPS outlier removal failed: {e}")
            st.info("🔄 Using original FPS data instead")
            return data
    
    def apply_savgol_filter(self, fps_window=21, fps_poly=3, cpu_window=21, cpu_poly=3, 
                           enable_fps=True, enable_cpu=True, enable_outlier_removal=False, outlier_sensitivity='moderate'):
        """Apply Savitzky-Golay filter dan IQR outlier removal untuk smoothing data FPS dan CPU secara terpisah"""
        if self.data is None:
            return False
        
        try:
            # Copy original data
            self.smoothed_data = self.data.copy()
            original_length = len(self.data)
            
            # Apply outlier removal if enabled
            if enable_outlier_removal:
                st.info("🧹 Applying FPS outlier removal (removing 1% worst frames only)...")
                
                # Remove outliers ONLY from FPS
                st.markdown("**FPS Outlier Removal:**")
                fps_filtered = self.remove_fps_outliers(self.data['FPS'], outlier_sensitivity)
                
                # Keep CPU data as-is (no outlier removal for CPU)
                st.markdown("**CPU Data:**")
                st.info("✅ CPU data kept unchanged (no outlier removal applied)")
                cpu_data = self.data['CPU(%)']
                
                # Adjust CPU data length to match filtered FPS
                min_length = len(fps_filtered)
                cpu_adjusted = cpu_data.iloc[:min_length] if len(cpu_data) > min_length else cpu_data
                
                # Create new dataframe with filtered FPS and original CPU
                self.smoothed_data = pd.DataFrame({
                    'FPS': fps_filtered.values,
                    'CPU(%)': cpu_adjusted.values,
                    'TimeMinutes': [i / 60 for i in range(min_length)]  # Recalculate time
                })
                
                st.success(f"✅ FPS outlier removal completed! Data: {original_length} → {min_length} points")
            
            # Apply FPS filter if enabled
            if enable_fps:
                # Pastikan window size ganjil dan tidak lebih besar dari data
                current_data_length = len(self.smoothed_data)
                fps_window = min(fps_window, current_data_length)
                if fps_window % 2 == 0:
                    fps_window -= 1
                fps_window = max(fps_window, 5)  # Minimum window size
                
                if len(self.smoothed_data['FPS'].dropna()) >= fps_window:
                    st.info(f"🎯 Applying FPS smoothing (window={fps_window}, poly={fps_poly})...")
                    self.smoothed_data['FPS_Smooth'] = savgol_filter(
                        self.smoothed_data['FPS'], 
                        window_length=fps_window, 
                        polyorder=min(fps_poly, fps_window-1)
                    )
                else:
                    self.smoothed_data['FPS_Smooth'] = self.smoothed_data['FPS']
            else:
                # Use FPS data (with or without outlier removal)
                self.smoothed_data['FPS_Smooth'] = self.smoothed_data['FPS']
            
            # Apply CPU filter if enabled
            if enable_cpu:
                # Pastikan window size ganjil dan tidak lebih besar dari data
                current_data_length = len(self.smoothed_data)
                cpu_window = min(cpu_window, current_data_length)
                if cpu_window % 2 == 0:
                    cpu_window -= 1
                cpu_window = max(cpu_window, 5)  # Minimum window size
                
                if len(self.smoothed_data['CPU(%)'].dropna()) >= cpu_window:
                    st.info(f"🖥️ Applying CPU smoothing (window={cpu_window}, poly={cpu_poly})...")
                    self.smoothed_data['CPU_Smooth'] = savgol_filter(
                        self.smoothed_data['CPU(%)'], 
                        window_length=cpu_window, 
                        polyorder=min(cpu_poly, cpu_window-1)
                    )
                else:
                    self.smoothed_data['CPU_Smooth'] = self.smoothed_data['CPU(%)']
            else:
                # Use CPU data (with or without outlier removal)
                self.smoothed_data['CPU_Smooth'] = self.smoothed_data['CPU(%)']
            
            return True
            
        except Exception as e:
            st.error(f"❌ Error applying filters: {e}")
            return False
    
    def create_chart(self, game_title, game_settings, game_mode, smartphone_name, fps_color, cpu_color, 
                    show_original=True, show_smoothed=True, enable_fps_filter=True, enable_cpu_filter=True):
        """Generate professional gaming chart dengan legend sederhana (hanya FPS dan CPU)"""
        
        # Create figure with 1920x1080 resolution (Full HD)
        fig, ax1 = plt.subplots(figsize=(19.2, 10.8))  # 1920x1080 pixels at 100 DPI
        fig.patch.set_facecolor('none')  # Transparent figure background
        
        # Setup axes
        ax1.set_xlabel('Time (minutes)', fontsize=12, fontweight='bold', color='black')
        ax1.set_ylabel('FPS', color='black', fontsize=12, fontweight='bold')
        ax1.tick_params(axis='y', labelcolor='black', labelsize=10)
        ax1.tick_params(axis='x', labelcolor='black', labelsize=10)
        
        ax2 = ax1.twinx()
        ax2.set_ylabel('CPU Usage (%)', color='black', fontsize=12, fontweight='bold')
        ax2.tick_params(axis='y', labelcolor='black', labelsize=10)
        ax2.set_ylim(0, 100)
        
        # Determine which data to use for time axis
        if show_smoothed and self.smoothed_data is not None:
            time_data = self.smoothed_data['TimeMinutes']
            data_length = len(self.smoothed_data)
        else:
            time_data = self.data['TimeMinutes']
            data_length = len(self.data)
        
        # Variable to track if we've added labels (to avoid duplicate legend entries)
        fps_labeled = False
        cpu_labeled = False
        
        # Original data (lebih transparan jika smoothed ditampilkan)
        if show_original:
            # Jika outlier removal diaktifkan, original data mungkin berbeda panjang
            if show_smoothed and self.smoothed_data is not None and len(self.smoothed_data) != len(self.data):
                # Adjust original data display untuk outlier removal case
                original_time = self.data['TimeMinutes'][:len(time_data)] if len(self.data) > len(time_data) else self.data['TimeMinutes']
                original_fps = self.data['FPS'][:len(time_data)] if len(self.data) > len(time_data) else self.data['FPS']
                original_cpu = self.data['CPU(%)'][:len(time_data)] if len(self.data) > len(time_data) else self.data['CPU(%)']
            else:
                original_time = self.data['TimeMinutes']
                original_fps = self.data['FPS']
                original_cpu = self.data['CPU(%)']
            
            alpha_original = 0.3 if show_smoothed else 0.9
            linestyle_original = '--' if show_smoothed else '-'
            
            # Only add label if this is the only line being shown for this metric
            fps_label = 'FPS' if not show_smoothed else None
            cpu_label = 'CPU' if not show_smoothed else None
            
            ax1.plot(original_time, original_fps, 
                    color=fps_color, linewidth=1, label=fps_label, 
                    alpha=alpha_original, zorder=2, linestyle=linestyle_original)
            ax2.plot(original_time, original_cpu, 
                    color=cpu_color, linewidth=1, label=cpu_label, 
                    alpha=alpha_original, zorder=1, linestyle=linestyle_original)
            
            if not show_smoothed:
                fps_labeled = True
                cpu_labeled = True
        
        # Smoothed data - dengan label sederhana
        if show_smoothed and self.smoothed_data is not None:
            # FPS smoothed line - hanya label 'FPS'
            fps_label = 'FPS' if not fps_labeled else None
            ax1.plot(time_data, self.smoothed_data['FPS_Smooth'], 
                    color=fps_color, linewidth=2.5, label=fps_label, 
                    alpha=0.9, zorder=4)
            
            # CPU smoothed line - hanya label 'CPU'
            cpu_label = 'CPU' if not cpu_labeled else None
            ax2.plot(time_data, self.smoothed_data['CPU_Smooth'], 
                    color=cpu_color, linewidth=2.5, label=cpu_label, 
                    alpha=0.9, zorder=3)
        
        # Set FPS axis limits based on the data being displayed
        if show_smoothed and self.smoothed_data is not None:
            fps_max = max(self.smoothed_data['FPS_Smooth']) * 1.1
        else:
            fps_max = max(self.data['FPS']) * 1.1
        ax1.set_ylim(0, fps_max)
        
        # Professional 3-line title (without smartphone name)
        title_text = f"{game_title}\n{game_settings}\n{game_mode}"
        plt.suptitle(title_text, fontsize=24, fontweight='bold', y=0.98, color='white')
        plt.subplots_adjust(top=0.85)
        
        # Styling
        ax1.grid(True, alpha=0.3, linestyle='--', color='white')
        ax1.set_facecolor('none')
        
        # Legend - menampilkan smartphone name, FPS dan CPU dengan alignment kiri
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        
        # Filter out None labels
        filtered_lines = []
        filtered_labels = []
        for line, label in zip(lines1 + lines2, labels1 + labels2):
            if label is not None:
                filtered_lines.append(line)
                filtered_labels.append(label)
        
        if filtered_lines:  # Only create legend if there are labels
            # Add smartphone name as first entry (invisible line)
            legend_lines = [plt.Line2D([0], [0], color='none')] + filtered_lines
            legend_labels = [smartphone_name] + filtered_labels
            
            legend = ax1.legend(legend_lines, legend_labels, 
                              loc='upper right', framealpha=0.8, fancybox=True,
                              facecolor='white', edgecolor='gray')
            
            # Style legend text with left alignment
            for i, text in enumerate(legend.get_texts()):
                text.set_color('black')
                text.set_horizontalalignment('left')  # Rata kiri
                if i == 0:  # Smartphone name - make it bold
                    text.set_fontweight('bold')
                    text.set_fontsize(11)
                else:  # FPS and CPU labels
                    text.set_fontsize(10)
        
        # Hide spines
        for spine in ax1.spines.values():
            spine.set_visible(False)
        for spine in ax2.spines.values():
            spine.set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def calculate_stats(self, use_processed=False):
        """Calculate gaming performance statistics"""
        if self.data is None:
            return {}
        
        # Pilih data yang akan digunakan untuk statistik
        if use_processed and self.smoothed_data is not None:
            fps_data = self.smoothed_data['FPS_Smooth'].dropna()
            cpu_data = self.smoothed_data['CPU_Smooth'].dropna()
            data_type = "(Processed)"
            duration = len(self.smoothed_data) / 60
        else:
            fps_data = self.data['FPS'].dropna()
            cpu_data = self.data['CPU(%)'].dropna()
            data_type = "(Original)"
            duration = len(self.data) / 60
        
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
            'frame_drops': len(fps_data[fps_data < 30])
        }

def main():
    # Header
    st.title("🎮 Gaming Performance Chart Generator")
    st.markdown("Transform your gaming logs into professional performance charts with **Savitzky-Golay smoothing**")
    
    # Initialize
    generator = GamingChartGenerator()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("🎮 Game Configuration")
        game_title = st.text_input("Game Title", value="MOBILE LEGENDS")
        game_settings = st.text_input("Graphics Settings", value="ULTRA - 120 FPS")
        game_mode = st.text_input("Performance Mode", value="BOOST MODE")
        smartphone_name = st.text_input("Smartphone Model", value="iPhone 15 Pro Max")
        
        st.header("🎨 Chart Colors")
        fps_color = st.color_picker("FPS Color", "#FF6600")
        cpu_color = st.color_picker("CPU Color", "#4A90E2")
        
        st.header("📊 Display Options")
        show_original = st.checkbox("Show Original Data", value=False)
        show_smoothed = st.checkbox("Show Smoothed Data", value=False)
        
        st.header("🔧 Savitzky-Golay Filter")
        st.markdown("**FPS Smoothing:**")
        enable_fps_filter = st.toggle("🎯 Enable FPS Smoothing", value=False,
                                     help="Turn on/off FPS data smoothing")
        
        if enable_fps_filter:
            fps_window = st.slider("FPS Window Size", min_value=5, max_value=51, value=5, step=2,
                                  help="Larger values = more smoothing")
            fps_poly = st.slider("FPS Polynomial Order", min_value=1, max_value=5, value=1,
                                help="Higher order = better fit to curves")
        else:
            fps_window = 5
            fps_poly = 1
            st.info("ℹ️ FPS smoothing disabled - using processed FPS data")
        
        st.markdown("**CPU Smoothing:**")
        enable_cpu_filter = st.toggle("🖥️ Enable CPU Smoothing", value=False,
                                     help="Turn on/off CPU usage data smoothing")
        
        if enable_cpu_filter:
            cpu_window = st.slider("CPU Window Size", min_value=5, max_value=51, value=5, step=2,
                                  help="Larger values = more smoothing")
            cpu_poly = st.slider("CPU Polynomial Order", min_value=1, max_value=5, value=1,
                                help="Higher order = better fit to curves")
        else:
            cpu_window = 5
            cpu_poly = 1
            st.info("ℹ️ CPU smoothing disabled - using processed CPU data")
        
        st.header("🧹 Enhanced FPS Outlier Removal")
        enable_outlier_removal = st.toggle("🚫 Remove 1% Worst FPS Drops", value=False,
                                          help="Remove only the worst 1% FPS drops - CPU data remains unchanged")
        
        if enable_outlier_removal:
            outlier_sensitivity = st.select_slider(
                "FPS Removal Sensitivity",
                options=['conservative', 'moderate', 'aggressive'],
                value='moderate',
                help="Conservative = 1% worst only, Moderate = 1% + slightly above, Aggressive = 5% worst"
            )
            
            # Explanation based on sensitivity
            if outlier_sensitivity == 'conservative':
                st.info("🛡️ **Conservative**: Remove only bottom 1% FPS frames")
            elif outlier_sensitivity == 'moderate':
                st.info("⚖️ **Moderate**: Remove bottom 1% + slightly above threshold")
            else:
                st.info("🔥 **Aggressive**: Remove bottom 5% FPS frames for cleanest result")
            
            st.warning("⚠️ **Note**: Only worst FPS frames will be removed. CPU data remains unchanged.")
            st.info("💡 **Example**: If FPS range is 30-120, only frames below ~35 FPS will be removed, keeping 80-120 FPS intact.")
        else:
            outlier_sensitivity = 'moderate'
            st.info("ℹ️ FPS outlier removal disabled - keeping all original data")
        
        # Statistics option - only show if at least one filter is enabled
        if enable_fps_filter or enable_cpu_filter or enable_outlier_removal:
            use_smoothed_stats = st.checkbox("Use Processed Data for Statistics", value=True,
                                            help="Use data after outlier removal and/or smoothing for statistics calculation")
        else:
            use_smoothed_stats = False
    
    # File upload
    st.header("📁 Upload Gaming Log CSV")
    uploaded_file = st.file_uploader("Drop your CSV file here", type=['csv'])
    
    if uploaded_file:
        with st.spinner('🔄 Analyzing gaming data...'):
            if generator.load_csv_data(uploaded_file):
                
                # Apply Savitzky-Golay filter dengan kontrol terpisah + outlier removal
                with st.spinner('🔧 Applying enhanced outlier removal and smoothing filters...'):
                    if generator.apply_savgol_filter(fps_window, fps_poly, cpu_window, cpu_poly, 
                                                   enable_fps_filter, enable_cpu_filter, enable_outlier_removal, outlier_sensitivity):
                        # Custom success message berdasarkan filter yang aktif
                        filter_messages = []
                        if enable_outlier_removal:
                            filter_messages.append(f"FPS outlier removal ({outlier_sensitivity})")
                        if enable_fps_filter:
                            filter_messages.append("FPS smoothing")
                        if enable_cpu_filter:
                            filter_messages.append("CPU smoothing")
                        if filter_messages:
                            st.success(f"🎉 Gaming log processed with {', '.join(filter_messages)}!")
                        else:
                            st.success("🎉 Gaming log loaded! (No processing applied)")
                    else:
                        st.warning("⚠️ Gaming log loaded but processing failed. Using original data only.")
                        show_smoothed = False  # Force disable smoothed display if filter failed
                
                # Quick stats - use appropriate data
                display_data = generator.smoothed_data if (enable_outlier_removal and generator.smoothed_data is not None) else generator.data
                
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
                
                if not show_original and not show_smoothed:
                    st.warning("⚠️ Please select at least one display option (Original or Smoothed)")
                else:
                    with st.spinner('🎨 Creating professional chart...'):
                        chart_fig = generator.create_chart(game_title, game_settings, game_mode, smartphone_name,
                                                         fps_color, cpu_color, show_original, show_smoothed,
                                                         enable_fps_filter, enable_cpu_filter)
                        st.pyplot(chart_fig)
                
                # Performance analysis
                stats = generator.calculate_stats(use_smoothed_stats and (enable_fps_filter or enable_cpu_filter or enable_outlier_removal))
                
                # Dynamic stats label
                active_processing = []
                if use_smoothed_stats:
                    if enable_outlier_removal:
                        active_processing.append("1% Frame Drops Removed")
                    if enable_fps_filter:
                        active_processing.append("FPS Smoothed")
                    if enable_cpu_filter:
                        active_processing.append("CPU Smoothed")
                
                if active_processing:
                    stats_label = f"({', '.join(active_processing)})"
                else:
                    stats_label = "(Original Data)"
                
                st.header(f"📈 Performance Analysis {stats_label}")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Performance Grade", stats['grade'])
                with col2:
                    st.metric("FPS Range", f"{stats['min_fps']}-{stats['max_fps']}")
                with col3:
                    st.metric("60+ FPS Time", f"{stats['fps_above_60']}%")
                with col4:
                    st.metric("Frame Drops", stats['frame_drops'])
                
                # Filter info - show information for active filters
                if (enable_fps_filter or enable_cpu_filter or enable_outlier_removal) and generator.smoothed_data is not None:
                    with st.expander("🔧 Active Processing Filters"):
                        
                        # Enhanced outlier removal info
                        if enable_outlier_removal:
                            st.markdown("**🧹 Enhanced FPS Outlier Removal (ACTIVE):**")
                            st.write(f"• Method: Pure Percentile-based Removal")
                            st.write(f"• Sensitivity: {outlier_sensitivity.title()}")
                            st.write(f"• **Target**: Remove only worst FPS frames (CPU unchanged)")
                            st.write(f"• **Effect**: Dataset becomes shorter, time axis is recalculated")
                            
                            # Show data reduction stats
                            original_length = len(generator.data)
                            processed_length = len(generator.smoothed_data)
                            reduction_pct = ((original_length - processed_length) / original_length) * 100
                            
                            st.write(f"• **Data Reduction**: {original_length} → {processed_length} points ({reduction_pct:.1f}% removed)")
                            st.write(f"• **Time Reduction**: {original_length/60:.1f} → {processed_length/60:.1f} minutes")
                            st.write(f"• **CPU Data**: Kept unchanged (no outlier removal)")
                            st.write("")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if enable_fps_filter:
                                st.markdown("**🎯 FPS Filter (ACTIVE):**")
                                st.write(f"• Window Size: {fps_window}")
                                st.write(f"• Polynomial Order: {fps_poly}")
                            else:
                                st.markdown("**🎯 FPS Filter (DISABLED)**")
                                st.write("• Using processed FPS data")
                        
                        with col2:
                            if enable_cpu_filter:
                                st.markdown("**🖥️ CPU Filter (ACTIVE):**")
                                st.write(f"• Window Size: {cpu_window}")
                                st.write(f"• Polynomial Order: {cpu_poly}")
                            else:
                                st.markdown("**🖥️ CPU Filter (DISABLED)**")
                                st.write("• Using processed CPU data")
                        
                        st.info("💡 **Enhanced Processing Order**: 1) Remove only worst FPS frames (CPU untouched) → 2) Apply Savitzky-Golay smoothing. This method preserves good FPS performance while removing only true frame drops.")
                else:
                    st.info("ℹ️ No processing filters active - displaying original data")
                
                # Download section
                st.header("💾 Export Results")
                
                # PNG download
                if 'chart_fig' in locals():
                    img_buffer = io.BytesIO()
                    chart_fig.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight',
                                     facecolor='none', edgecolor='none', transparent=True)
                    img_buffer.seek(0)
                    
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    
                    # Dynamic filename based on active filters
                    filter_suffix = ""
                    filters_active = []
                    if enable_outlier_removal:
                        filters_active.append(f"1pct_removed_{outlier_sensitivity}")
                    if enable_fps_filter:
                        filters_active.append("fps_smooth")
                    if enable_cpu_filter:
                        filters_active.append("cpu_smooth")
                    
                    if filters_active:
                        filter_suffix = "_" + "_".join(filters_active)
                    
                    png_filename = f"{game_title.replace(' ', '_')}{filter_suffix}_chart_{timestamp}.png"
                    
                    st.download_button(
                        label="📸 Download Chart (PNG)",
                        data=img_buffer.getvalue(),
                        file_name=png_filename,
                        mime="image/png",
                        use_container_width=True
                    )
    
    else:
        # Help section
        st.info("📤 Upload your gaming log CSV to get started!")
        
        with st.expander("📋 Supported CSV Format"):
            st.markdown("""
            **Required columns:**
            - FPS data (any column with 'fps' in name)
            - CPU usage data (any column with 'cpu' and '%')
            
            **Example CSV structure:**
            ```
            FPS,CPU(%),JANK,BigJANK
            60,45.2,0,0
            58,48.1,1,0
            62,42.8,0,0
            ```
            """)
        
        with st.expander("🧹 About Enhanced FPS Outlier Removal"):
            st.markdown("""
            **Enhanced FPS Method** menghilangkan hanya worst FPS frames:
            - ✅ **FOCUS**: Only removes worst FPS performance frames
            - ✅ **PRESERVE**: Keeps good FPS (80-120) completely intact
            - ✅ **CPU SAFE**: CPU data remains 100% unchanged
            - ✅ **ACCURATE**: Better represents actual gaming experience
            
            **Key Advantages:**
            1. **Percentile-based**: Uses pure percentile thresholds, not IQR
            2. **FPS-only**: Only processes FPS data, leaves CPU untouched
            3. **Conservative**: Removes true frame drops, preserves good performance
            4. **Sequential Data**: No gaps in data (a,b,c,d,e → a,b,d,e if c is bad frame)
            
            **How it works:**
            1. Calculate FPS percentiles (1st, 5th percentile)
            2. Set removal threshold based on sensitivity
            3. Remove only frames BELOW threshold (worst performance)
            4. Keep CPU data exactly as-is
            5. Reset index to create continuous sequence
            6. Recalculate time axis for remaining data
            
            **Sensitivity Levels:**
            - **Conservative**: Remove only bottom 1% FPS frames
            - **Moderate**: Remove bottom 1% + slightly above threshold
            - **Aggressive**: Remove bottom 5% FPS frames for cleanest result
            
            **Example Impact:**
            - FPS Range: 30-120 FPS
            - Conservative: Removes frames <31 FPS (keeps 80+ FPS intact)
            - Moderate: Removes frames <35 FPS
            - Aggressive: Removes frames <45 FPS
            - CPU: Always unchanged regardless of FPS filtering
            """)
        
        with st.expander("🔧 About Savitzky-Golay Filter"):
            st.markdown("""
            **Savitzky-Golay Filter** adalah metode smoothing yang:
            - ✅ Mengurangi noise dalam data
            - ✅ Mempertahankan bentuk kurva asli
            - ✅ Cocok untuk data gaming performance
            - ✅ Dapat disesuaikan secara terpisah untuk FPS dan CPU
            
            **Parameter:**
            - **Window Size**: Jumlah data point yang digunakan (lebih besar = lebih smooth)
            - **Polynomial Order**: Tingkat polynomial untuk fitting (1-5)
            
            **Tips:**
            - Window size ganjil dan minimal 5
            - Polynomial order harus lebih kecil dari window size
            - Untuk data noisy: window size besar, poly order rendah
            - Untuk mempertahankan detail: window size kecil, poly order tinggi
            
            **Works best with Enhanced FPS Removal:**
            - First remove only worst FPS frames
            - Then smooth the remaining clean data
            - Result: Professional-grade performance charts with preserved good performance
            """)
        
        # Footer
        st.markdown("---")
        st.markdown("Made with ❤️ for gaming performance analysis | Enhanced with Smart FPS Outlier Removal + Savitzky-Golay smoothing")

if __name__ == "__main__":
    main()
