import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io
from datetime import datetime
from scipy.signal import savgol_filter

# Page configuration
st.set_page_config(
    page_title="🎮 Gaming Chart Generator - Professional",
    page_icon="🎮",
    layout="wide"
)

class ProfessionalGamingChartGenerator:
    def __init__(self):
        self.original_data = None
        self.processed_data = None
        self.removed_indices = []
    
    def apply_intelligent_smoothing(self, data, window_size='auto', method='savgol'):
        """Apply intelligent smoothing to make data look professional like mobile gaming apps"""
        try:
            if len(data) < 10:
                return data
            
            # Auto-determine optimal window size based on data length and noise level
            if window_size == 'auto':
                data_length = len(data)
                if data_length > 10000:
                    window = 51  # More aggressive smoothing for very long data
                elif data_length > 5000:
                    window = 31  # Moderate smoothing
                elif data_length > 1000:
                    window = 21  # Standard smoothing
                else:
                    window = min(11, data_length // 3)  # Conservative for short data
            else:
                window = window_size
            
            # Ensure window is odd and reasonable
            if window % 2 == 0:
                window += 1
            window = max(5, min(window, len(data) - 1))
            
            if method == 'savgol':
                # Use Savitzky-Golay filter with conservative polynomial order
                poly_order = min(3, window - 1)
                smoothed = savgol_filter(data, window_length=window, polyorder=poly_order)
                
                # Apply additional light moving average for extra smoothness
                if len(smoothed) > 10:
                    smoothed = pd.Series(smoothed).rolling(window=5, center=True).mean().fillna(method='bfill').fillna(method='ffill')
                
                return smoothed
            
            elif method == 'moving_average':
                # Simple moving average
                return pd.Series(data).rolling(window=window, center=True).mean().fillna(method='bfill').fillna(method='ffill')
            
            return data
            
        except Exception as e:
            st.warning(f"Smoothing failed: {e}, using original data")
            return data
    
    def load_csv_data(self, uploaded_file):
        """Load CSV with professional gaming data processing"""
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
            
            # Column detection
            columns = list(self.original_data.columns)
            
            # Check for EXACT required columns
            if 'FPS' not in columns:
                st.error("❌ Required 'FPS' column not found!")
                st.info(f"Available columns: {', '.join(columns)}")
                return False
            
            if 'CPU(%)' not in columns:
                st.error("❌ Required 'CPU(%)' column not found!")
                st.info(f"Available columns: {', '.join(columns)}")
                return False
            
            # Extract and validate data
            fps_data = pd.to_numeric(self.original_data['FPS'], errors='coerce')
            cpu_data = pd.to_numeric(self.original_data['CPU(%)'], errors='coerce')
            
            # Remove NaN values
            fps_data = fps_data.fillna(method='ffill').fillna(method='bfill')
            cpu_data = cpu_data.fillna(method='ffill').fillna(method='bfill')
            
            # Apply intelligent smoothing by default for professional look
            st.info("🎯 Applying professional smoothing for clean visualization...")
            
            # Apply smart smoothing
            fps_smoothed = self.apply_intelligent_smoothing(fps_data, window_size='auto', method='savgol')
            cpu_smoothed = self.apply_intelligent_smoothing(cpu_data, window_size='auto', method='savgol')
            
            # Create processed dataset
            self.original_data['FPS_Raw'] = fps_data
            self.original_data['CPU_Raw'] = cpu_data
            self.original_data['FPS'] = fps_smoothed
            self.original_data['CPU(%)'] = cpu_smoothed
            self.original_data['TimeMinutes'] = [i / 60 for i in range(len(self.original_data))]
            
            # Show data quality info
            st.info(f"📊 Dataset: {len(self.original_data)} rows × {len(self.original_data.columns)} columns")
            
            # Show data preview
            col1, col2 = st.columns(2)
            with col1:
                st.metric("📈 FPS Range", f"{fps_smoothed.min():.1f} - {fps_smoothed.max():.1f}")
                st.metric("📊 FPS Average", f"{fps_smoothed.mean():.1f}")
            with col2:
                st.metric("🖥️ CPU Range", f"{cpu_smoothed.min():.1f}% - {cpu_smoothed.max():.1f}%")
                st.metric("📊 CPU Average", f"{cpu_smoothed.mean():.1f}%")
            
            return True
            
        except Exception as e:
            st.error(f"❌ Error loading CSV: {e}")
            return False
    
    def apply_additional_processing(self, enable_outlier_removal=False, outlier_sensitivity='moderate'):
        """Apply additional processing if needed"""
        
        if not enable_outlier_removal:
            # Just copy the already-smoothed data
            self.processed_data = self.original_data.copy()
            return True
        
        try:
            # Apply outlier removal on already-smoothed data
            fps_data = self.original_data['FPS'].dropna()
            
            # Calculate thresholds
            percentile_1 = fps_data.quantile(0.01)
            percentile_5 = fps_data.quantile(0.05)
            
            thresholds = {
                'conservative': percentile_1,
                'moderate': percentile_1 * 1.1,
                'aggressive': percentile_5
            }
            threshold = thresholds.get(outlier_sensitivity, percentile_1)
            
            # Get indices to keep
            keep_mask = fps_data >= threshold
            keep_indices = fps_data[keep_mask].index.tolist()
            
            # Create processed dataset
            self.processed_data = pd.DataFrame({
                'FPS': self.original_data.loc[keep_indices, 'FPS'].values,
                'CPU(%)': self.original_data.loc[keep_indices, 'CPU(%)'].values,
                'TimeMinutes': [i / 60 for i in range(len(keep_indices))]
            })
            
            # Store removed indices
            removed_indices = fps_data[~keep_mask].index.tolist()
            self.removed_indices = removed_indices
            
            removal_count = len(removed_indices)
            removal_pct = (removal_count / len(fps_data)) * 100
            
            st.success(f"✅ Removed {removal_count} outlier frames ({removal_pct:.1f}%)")
            
            return True
            
        except Exception as e:
            st.error(f"❌ Additional processing failed: {e}")
            self.processed_data = self.original_data.copy()
            return False
    
    def create_professional_chart(self, game_title, game_settings, game_mode, smartphone_name,
                                fps_color, cpu_color, show_raw_data=False):
        """Create professional gaming chart like mobile gaming monitoring apps"""
        
        # Use processed data if available, otherwise original
        data_to_use = self.processed_data if self.processed_data is not None else self.original_data
        
        # Create figure with high resolution
        fig, ax1 = plt.subplots(figsize=(19.2, 10.8))
        fig.patch.set_facecolor('#1e1e1e')  # Dark background like gaming apps
        
        # Setup primary axis (FPS)
        ax1.set_xlabel('Time (minutes)', fontsize=14, fontweight='bold', color='white')
        ax1.set_ylabel('FPS', color=fps_color, fontsize=14, fontweight='bold')
        ax1.tick_params(axis='y', labelcolor=fps_color, labelsize=12, colors='white')
        ax1.tick_params(axis='x', labelcolor='white', labelsize=12, colors='white')
        
        # Setup secondary axis (CPU)
        ax2 = ax1.twinx()
        ax2.set_ylabel('CPU Usage (%)', color=cpu_color, fontsize=14, fontweight='bold')
        ax2.tick_params(axis='y', labelcolor=cpu_color, labelsize=12, colors='white')
        ax2.set_ylim(0, 100)
        
        # Get data
        time_data = data_to_use['TimeMinutes']
        fps_data = data_to_use['FPS']
        cpu_data = data_to_use['CPU(%)']
        
        # Plot raw data (if requested) - very faded
        if show_raw_data and 'FPS_Raw' in self.original_data:
            raw_time = self.original_data['TimeMinutes'][:len(time_data)]
            raw_fps = self.original_data['FPS_Raw'][:len(time_data)]
            raw_cpu = self.original_data['CPU_Raw'][:len(time_data)]
            
            ax1.plot(raw_time, raw_fps, color=fps_color, linewidth=0.5, 
                    alpha=0.15, zorder=1, label='FPS (Raw)')
            ax2.plot(raw_time, raw_cpu, color=cpu_color, linewidth=0.5,
                    alpha=0.15, zorder=1, label='CPU (Raw)')
        
        # Plot main smoothed data - prominent and clean
        ax1.plot(time_data, fps_data, color=fps_color, linewidth=3.0,
                alpha=0.9, zorder=4, label='FPS', solid_capstyle='round')
        ax2.plot(time_data, cpu_data, color=cpu_color, linewidth=3.0,
                alpha=0.9, zorder=3, label='CPU', solid_capstyle='round')
        
        # Add fill areas for better visual appeal (like gaming apps)
        ax1.fill_between(time_data, 0, fps_data, color=fps_color, alpha=0.1, zorder=2)
        ax2.fill_between(time_data, 0, cpu_data, color=cpu_color, alpha=0.1, zorder=1)
        
        # Set axis limits
        fps_max = max(fps_data) * 1.05  # Small padding
        ax1.set_ylim(0, fps_max)
        
        # Professional title styling
        title_text = f"{game_title}\n{game_settings}\n{game_mode}"
        plt.suptitle(title_text, fontsize=20, fontweight='bold', y=0.95, color='white')
        
        # Professional grid styling
        ax1.grid(True, alpha=0.2, linestyle='-', color='white', linewidth=0.5)
        ax1.set_facecolor('#1e1e1e')
        
        # Professional legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        
        if lines1 or lines2:
            # Create custom legend
            legend_elements = []
            
            # Add device name
            legend_elements.append(plt.Line2D([0], [0], color='white', linewidth=0, 
                                            marker='s', markersize=8, markerfacecolor='none', 
                                            markeredgecolor='white', label=smartphone_name))
            
            # Add FPS line
            legend_elements.append(plt.Line2D([0], [0], color=fps_color, linewidth=3, label='FPS'))
            
            # Add CPU line  
            legend_elements.append(plt.Line2D([0], [0], color=cpu_color, linewidth=3, label='CPU'))
            
            legend = ax1.legend(handles=legend_elements, loc='upper right', 
                              framealpha=0.9, fancybox=True, shadow=True,
                              facecolor='#2d2d2d', edgecolor='white', fontsize=11)
            
            # Style legend text
            for text in legend.get_texts():
                text.set_color('white')
                if smartphone_name in text.get_text():
                    text.set_fontweight('bold')
        
        # Remove spines for clean look
        for spine in ax1.spines.values():
            spine.set_color('white')
            spine.set_linewidth(0.5)
        for spine in ax2.spines.values():
            spine.set_color('white')
            spine.set_linewidth(0.5)
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        
        return fig
    
    def get_statistics(self):
        """Get performance statistics"""
        if self.original_data is None:
            return {}
        
        # Use processed data if available
        data_to_use = self.processed_data if self.processed_data is not None else self.original_data
        
        fps_data = data_to_use['FPS'].dropna()
        cpu_data = data_to_use['CPU(%)'].dropna()
        duration = len(data_to_use) / 60
        
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
        """Generate CSV with processed data"""
        data_to_use = self.processed_data if self.processed_data is not None else self.original_data
        
        if data_to_use is None:
            return None
        
        # Create clean DataFrame for export
        export_data = pd.DataFrame({
            'Time_Minutes': data_to_use['TimeMinutes'],
            'FPS': data_to_use['FPS'],
            'CPU_Percent': data_to_use['CPU(%)']
        })
        
        # Round values
        export_data['Time_Minutes'] = export_data['Time_Minutes'].round(3)
        export_data['FPS'] = export_data['FPS'].round(1)
        export_data['CPU_Percent'] = export_data['CPU_Percent'].round(1)
        
        # Convert to CSV
        csv_buffer = io.StringIO()
        export_data.to_csv(csv_buffer, index=False)
        csv_content = csv_buffer.getvalue()
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{game_title.replace(' ', '_')}_professional_data_{timestamp}.csv"
        
        return csv_content, filename

def main():
    # Header
    st.title("🎮 Professional Gaming Chart Generator")
    st.markdown("Create **professional gaming charts** like mobile gaming monitoring apps")
    
    # Initialize
    generator = ProfessionalGamingChartGenerator()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("🎮 Game Configuration")
        game_title = st.text_input("Game Title", value="MOBILE LEGENDS")
        game_settings = st.text_input("Graphics Settings", value="ULTRA - 120 FPS")
        game_mode = st.text_input("Performance Mode", value="BOOST MODE")
        smartphone_name = st.text_input("Smartphone Model", value="iPhone 15 Pro Max")
        
        st.header("🎨 Chart Colors")
        fps_color = st.color_picker("FPS Color", "#4A90E2")  # Blue
        cpu_color = st.color_picker("CPU Color", "#FF6600")   # Orange
        
        st.header("📊 Chart Options")
        show_raw_data = st.checkbox("Show Raw Data Background", value=False,
                                   help="Show original noisy data as faded background")
        
        st.header("🔧 Additional Processing")
        enable_outlier_removal = st.toggle("🚫 Remove Outlier Frames", value=False,
                                          help="Remove extreme outlier frames")
        
        if enable_outlier_removal:
            outlier_sensitivity = st.select_slider(
                "Outlier Sensitivity",
                options=['conservative', 'moderate', 'aggressive'],
                value='moderate'
            )
        else:
            outlier_sensitivity = 'moderate'
    
    # File upload
    st.header("📁 Upload Gaming Log CSV")
    uploaded_file = st.file_uploader("Drop your CSV file here", type=['csv'])
    
    if uploaded_file:
        # Load and process data
        if generator.load_csv_data(uploaded_file):
            
            # Apply additional processing
            with st.spinner('🔧 Applying professional processing...'):
                if generator.apply_additional_processing(enable_outlier_removal, outlier_sensitivity):
                    st.success("✅ Professional processing completed!")
                else:
                    st.warning("⚠️ Using default processing")
            
            # Show quick stats
            data_to_display = generator.processed_data if generator.processed_data is not None else generator.original_data
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📊 Data Points", f"{len(data_to_display):,}")
            with col2:
                st.metric("⏱️ Duration", f"{len(data_to_display)/60:.1f} min")
            with col3:
                st.metric("🎯 Avg FPS", f"{data_to_display['FPS'].mean():.1f}")
            with col4:
                st.metric("🖥️ Avg CPU", f"{data_to_display['CPU(%)'].mean():.1f}%")
            
            # Generate professional chart
            st.header("📊 Professional Gaming Performance Chart")
            
            with st.spinner('🎨 Creating professional chart...'):
                chart_fig = generator.create_professional_chart(
                    game_title, game_settings, game_mode, smartphone_name,
                    fps_color, cpu_color, show_raw_data
                )
                st.pyplot(chart_fig, use_container_width=True)
            
            # Performance statistics
            stats = generator.get_statistics()
            
            st.header("📈 Performance Analysis")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Performance Grade", stats['grade'])
            with col2:
                st.metric("FPS Range", f"{stats['min_fps']}-{stats['max_fps']}")
            with col3:
                st.metric("60+ FPS Time", f"{stats['fps_above_60']}%")
            with col4:
                st.metric("Frame Drops", stats['frame_drops'])
            
            # Export section
            st.header("💾 Export Results")
            
            col1, col2 = st.columns(2)
            
            # PNG Export
            with col1:
                st.subheader("📸 Chart Export")
                if 'chart_fig' in locals():
                    img_buffer = io.BytesIO()
                    chart_fig.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight',
                                     facecolor='#1e1e1e', edgecolor='none')
                    img_buffer.seek(0)
                    
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    png_filename = f"{game_title.replace(' ', '_')}_professional_chart_{timestamp}.png"
                    
                    st.download_button(
                        label="📸 Download Professional Chart",
                        data=img_buffer.getvalue(),
                        file_name=png_filename,
                        mime="image/png",
                        use_container_width=True
                    )
            
            # CSV Export
            with col2:
                st.subheader("📄 Data Export")
                result = generator.generate_processed_csv(game_title)
                
                if result is not None:
                    csv_content, csv_filename = result
                    
                    st.download_button(
                        label="📄 Download Processed Data",
                        data=csv_content,
                        file_name=csv_filename,
                        mime="text/csv",
                        use_container_width=True
                    )
                    
                    # Show preview
                    with st.expander("👀 Preview Export Data"):
                        preview_df = pd.read_csv(io.StringIO(csv_content))
                        st.dataframe(preview_df.head(10))
                        st.info(f"📊 Export contains {len(preview_df)} rows")
    
    else:
        # Help section
        st.info("📤 Upload your gaming log CSV to create professional charts!")
        
        with st.expander("📋 Required CSV Format"):
            st.markdown("""
            **Requirements:**
            - Column labeled exactly: **FPS**
            - Column labeled exactly: **CPU(%)**
            
            **Features:**
            - 🎯 **Auto-smoothing** for professional appearance
            - 🎨 **Gaming app styling** with dark theme
            - 📊 **Fill areas** and professional gradients
            - 🔧 **Intelligent processing** based on data characteristics
            """)
        
        with st.expander("✨ Professional Features"):
            st.markdown("""
            **This tool creates charts that look like professional gaming monitoring apps:**
            - **Intelligent smoothing** removes noise automatically
            - **Professional styling** with dark theme and gradients
            - **Mobile gaming app appearance** similar to gaming monitors
            - **High-quality exports** ready for presentations
            - **Automatic optimization** based on data characteristics
            """)

if __name__ == "__main__":
    main()
