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
    
    def apply_savgol_filter(self, fps_window=21, fps_poly=3, cpu_window=21, cpu_poly=3, 
                           enable_fps=True, enable_cpu=True, enable_outlier_removal=False, outlier_sensitivity='moderate'):
        """Apply Savitzky-Golay filter dan IQR outlier removal untuk smoothing data FPS dan CPU secara terpisah"""
        if self.data is None:
            return False
        
        try:
            # Copy original data
            self.smoothed_data = self.data.copy()
            
            # Apply outlier removal if enabled
            if enable_outlier_removal:
                self.smoothed_data['FPS'] = self.remove_outliers_iqr(self.data['FPS'], outlier_sensitivity)
                self.smoothed_data['CPU(%)'] = self.remove_outliers_iqr(self.data['CPU(%)'], outlier_sensitivity)
            
            # Apply FPS filter if enabled
            if enable_fps:
                # Pastikan window size ganjil dan tidak lebih besar dari data
                fps_window = min(fps_window, len(self.smoothed_data))
                if fps_window % 2 == 0:
                    fps_window -= 1
                fps_window = max(fps_window, 5)  # Minimum window size
                
                if len(self.smoothed_data['FPS'].dropna()) >= fps_window:
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
                cpu_window = min(cpu_window, len(self.smoothed_data))
                if cpu_window % 2 == 0:
                    cpu_window -= 1
                cpu_window = max(cpu_window, 5)  # Minimum window size
                
                if len(self.smoothed_data['CPU(%)'].dropna()) >= cpu_window:
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
    
    def remove_outliers_iqr(self, data, sensitivity='moderate'):
        """Remove outliers using IQR method"""
        try:
            # Remove NaN values
            clean_data = data.dropna()
            if len(clean_data) < 10:  # Need minimum data points
                return data
            
            # Calculate IQR
            q1 = clean_data.quantile(0.25)
            q3 = clean_data.quantile(0.75)
            iqr = q3 - q1
            
            # Set multiplier based on sensitivity
            multipliers = {
                'conservative': 2.0,  # Keep more data
                'moderate': 1.5,      # Balanced
                'aggressive': 1.0     # Remove more outliers
            }
            multiplier = multipliers.get(sensitivity, 1.5)
            
            # Calculate bounds
            lower_bound = q1 - multiplier * iqr
            upper_bound = q3 + multiplier * iqr
            
            # Replace outliers with interpolated values
            result = data.copy()
            outlier_mask = (data < lower_bound) | (data > upper_bound)
            
            # Simple linear interpolation for outliers
            if outlier_mask.any():
                result = result.interpolate(method='linear')
                # If still NaN at edges, use forward/backward fill
                result = result.fillna(method='ffill').fillna(method='bfill')
            
            return result
            
        except Exception as e:
            st.warning(f"⚠️ Outlier removal failed: {e}")
            return data
    
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
        
        # Plot data
        time_data = self.data['TimeMinutes']
        
        # Variable to track if we've added labels (to avoid duplicate legend entries)
        fps_labeled = False
        cpu_labeled = False
        
        # Original data (lebih transparan jika smoothed ditampilkan)
        if show_original:
            alpha_original = 0.3 if show_smoothed else 0.9
            linestyle_original = '--' if show_smoothed else '-'
            
            # Only add label if this is the only line being shown for this metric
            fps_label = 'FPS' if not show_smoothed else None
            cpu_label = 'CPU' if not show_smoothed else None
            
            ax1.plot(time_data, self.data['FPS'], 
                    color=fps_color, linewidth=1, label=fps_label, 
                    alpha=alpha_original, zorder=2, linestyle=linestyle_original)
            ax2.plot(time_data, self.data['CPU(%)'], 
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
        
        # Set FPS axis limits
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
        else:
            fps_data = self.data['FPS'].dropna()
            cpu_data = self.data['CPU(%)'].dropna()
            data_type = "(Original)"
        
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
            'duration': round(len(self.data) / 60, 1),
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
        show_original = st.checkbox("Show Original Data", value=True)
        show_smoothed = st.checkbox("Show Smoothed Data", value=True)
        
        st.header("🔧 Savitzky-Golay Filter")
        st.markdown("**FPS Smoothing:**")
        enable_fps_filter = st.toggle("🎯 Enable FPS Smoothing", value=True,
                                     help="Turn on/off FPS data smoothing")
        
        if enable_fps_filter:
            fps_window = st.slider("FPS Window Size", min_value=5, max_value=51, value=21, step=2,
                                  help="Larger values = more smoothing")
            fps_poly = st.slider("FPS Polynomial Order", min_value=1, max_value=5, value=3,
                                help="Higher order = better fit to curves")
        else:
            fps_window = 21
            fps_poly = 3
            st.info("ℹ️ FPS smoothing disabled - using original FPS data")
        
        st.markdown("**CPU Smoothing:**")
        enable_cpu_filter = st.toggle("🖥️ Enable CPU Smoothing", value=True,
                                     help="Turn on/off CPU usage data smoothing")
        
        if enable_cpu_filter:
            cpu_window = st.slider("CPU Window Size", min_value=5, max_value=51, value=21, step=2,
                                  help="Larger values = more smoothing")
            cpu_poly = st.slider("CPU Polynomial Order", min_value=1, max_value=5, value=3,
                                help="Higher order = better fit to curves")
        else:
            cpu_window = 21
            cpu_poly = 3
            st.info("ℹ️ CPU smoothing disabled - using original CPU data")
        
        st.header("🧹 Outlier Removal (IQR Method)")
        enable_outlier_removal = st.toggle("🚫 Remove Extreme FPS Drops", value=False,
                                          help="Remove extreme FPS drops and spikes using IQR statistical method")
        
        if enable_outlier_removal:
            outlier_sensitivity = st.select_slider(
                "Outlier Sensitivity",
                options=['conservative', 'moderate', 'aggressive'],
                value='moderate',
                help="Conservative = keep more data, Aggressive = remove more outliers"
            )
            
            # Explanation based on sensitivity
            if outlier_sensitivity == 'conservative':
                st.info("🛡️ **Conservative**: Only removes very extreme outliers (2.0 × IQR)")
            elif outlier_sensitivity == 'moderate':
                st.info("⚖️ **Moderate**: Balanced removal of outliers (1.5 × IQR)")
            else:
                st.info("🔥 **Aggressive**: Removes more outliers for cleaner chart (1.0 × IQR)")
        else:
            outlier_sensitivity = 'moderate'
            st.info("ℹ️ Outlier removal disabled - keeping all original data")
        
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
                with st.spinner('🔧 Applying smoothing filters and outlier removal...'):
                    if generator.apply_savgol_filter(fps_window, fps_poly, cpu_window, cpu_poly, 
                                                   enable_fps_filter, enable_cpu_filter, enable_outlier_removal, outlier_sensitivity):
                        # Custom success message berdasarkan filter yang aktif
                        filter_messages = []
                        if enable_outlier_removal:
                            filter_messages.append(f"outlier removal ({outlier_sensitivity})")
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
                
                # Quick stats
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📊 Data Points", f"{len(generator.data):,}")
                with col2:
                    st.metric("⏱️ Duration", f"{len(generator.data)/60:.1f} min")
                with col3:
                    st.metric("🎯 Avg FPS", f"{generator.data['FPS'].mean():.1f}")
                with col4:
                    st.metric("🖥️ Avg CPU", f"{generator.data['CPU(%)'].mean():.1f}%")
                
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
                        active_processing.append("Outlier Removed")
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
                        
                        # Outlier removal info
                        if enable_outlier_removal:
                            st.markdown("**🧹 Outlier Removal (ACTIVE):**")
                            st.write(f"• Method: IQR (Interquartile Range)")
                            st.write(f"• Sensitivity: {outlier_sensitivity.title()}")
                            
                            # Calculate and show outlier stats
                            if 'FPS' in generator.data.columns:
                                original_fps = generator.data['FPS'].dropna()
                                processed_fps = generator.smoothed_data['FPS'].dropna() if 'FPS' in generator.smoothed_data.columns else original_fps
                                
                                # Simple outlier detection for display
                                q1 = original_fps.quantile(0.25)
                                q3 = original_fps.quantile(0.75)
                                iqr = q3 - q1
                                multipliers = {'conservative': 2.0, 'moderate': 1.5, 'aggressive': 1.0}
                                mult = multipliers[outlier_sensitivity]
                                lower_bound = q1 - mult * iqr
                                upper_bound = q3 + mult * iqr
                                
                                outliers_detected = len(original_fps[(original_fps < lower_bound) | (original_fps > upper_bound)])
                                st.write(f"• FPS Outliers Detected: {outliers_detected} ({outliers_detected/len(original_fps)*100:.1f}%)")
                                st.write(f"• Valid FPS Range: {lower_bound:.1f} - {upper_bound:.1f}")
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
                        
                        st.info("💡 **Processing Order**: 1) Remove outliers → 2) Apply Savitzky-Golay smoothing. IQR method menghilangkan extreme drops/spikes, lalu smoothing mengurangi noise kecil.")
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
                        filters_active.append(f"outlier_{outlier_sensitivity}")
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
        
        with st.expander("🧹 About IQR Outlier Removal"):
            st.markdown("""
            **IQR (Interquartile Range) Method** menghilangkan extreme FPS drops:
            - ✅ Mendeteksi dan menghilangkan FPS drops/spikes yang tidak normal
            - ✅ Berdasarkan statistik robust (tidak terpengaruh outlier)
            - ✅ Menggunakan interpolasi untuk mengisi gap yang dihilangkan
            - ✅ Membuat chart lebih clean dan mudah dibaca
            
            **Cara Kerja:**
            1. Hitung Q1 (25th percentile) dan Q3 (75th percentile)
            2. IQR = Q3 - Q1
            3. Lower Bound = Q1 - (multiplier × IQR)
            4. Upper Bound = Q3 + (multiplier × IQR)
            5. Data di luar range dianggap outlier dan di-interpolasi
            
            **Sensitivity Levels:**
            - **Conservative (2.0×)**: Hapus cuma yang sangat ekstrem
            - **Moderate (1.5×)**: Standard statistical outlier detection
            - **Aggressive (1.0×)**: Hapus lebih banyak untuk hasil lebih smooth
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
            """)
        
        # Footer
        st.markdown("---")
        st.markdown("Made with ❤️ for gaming performance analysis | Enhanced with Savitzky-Golay smoothing")

if __name__ == "__main__":
    main()
