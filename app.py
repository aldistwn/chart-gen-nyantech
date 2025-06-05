import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="🎮 Gaming Chart Generator",
    page_icon="🎮",
    layout="wide"
)

class GamingChartGenerator:
    def __init__(self):
        self.data = None
    
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
    
    def create_chart(self, game_title, game_settings, game_mode, fps_color, cpu_color):
        """Generate professional gaming chart with transparent background - 1920x1080"""
        
        # Create figure with 1920x1080 resolution (Full HD)
        fig, ax1 = plt.subplots(figsize=(19.2, 10.8))  # 1920x1080 pixels at 100 DPI
        fig.patch.set_facecolor('none')  # Transparent figure background
        
        # FPS line (primary - in front)
        ax1.set_xlabel('Time (minutes)', fontsize=12, fontweight='bold', color='black')
        ax1.set_ylabel('FPS', color='black', fontsize=12, fontweight='bold')
        ax1.plot(self.data['TimeMinutes'], self.data['FPS'], 
                color=fps_color, linewidth=1.5, label='FPS', alpha=0.9, zorder=3)
        ax1.tick_params(axis='y', labelcolor='black', labelsize=10)
        ax1.tick_params(axis='x', labelcolor='black', labelsize=10)
        ax1.set_ylim(0, max(self.data['FPS']) * 1.1)
        
        # CPU line (secondary - behind)
        ax2 = ax1.twinx()
        ax2.set_ylabel('CPU Usage (%)', color='black', fontsize=12, fontweight='bold')
        ax2.plot(self.data['TimeMinutes'], self.data['CPU(%)'], 
                color=cpu_color, linewidth=1.5, label='CPU(%)', alpha=0.9, zorder=2)
        ax2.tick_params(axis='y', labelcolor='black', labelsize=10)
        ax2.set_ylim(0, 100)
        
        # Professional 3-line title with white color for overlay
        title_text = f"{game_title}\n{game_settings}\n{game_mode}"
        plt.suptitle(title_text, fontsize=16, fontweight='bold', y=0.92, color='white')
        plt.subplots_adjust(top=0.85)
        
        # Styling with transparent background
        ax1.grid(True, alpha=0.3, linestyle='--', color='white')
        ax1.set_facecolor('none')  # Transparent chart area
        
        # Legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        legend = ax1.legend(lines1 + lines2, labels1 + labels2, 
                          loc='upper right', framealpha=0.8, fancybox=True,
                          facecolor='white', edgecolor='gray')
        
        for text in legend.get_texts():
            text.set_color('black')
        
        # Hide spines for cleaner transparent look
        for spine in ax1.spines.values():
            spine.set_visible(False)
        for spine in ax2.spines.values():
            spine.set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def calculate_stats(self):
        """Calculate gaming performance statistics"""
        if self.data is None:
            return {}
        
        fps_data = self.data['FPS'].dropna()
        cpu_data = self.data['CPU(%)'].dropna()
        
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
    st.markdown("Transform your gaming logs into professional performance charts")
    
    # Initialize
    generator = GamingChartGenerator()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("🎮 Game Configuration")
        game_title = st.text_input("Game Title", value="MOBILE LEGENDS")
        game_settings = st.text_input("Graphics Settings", value="ULTRA - 120 FPS")
        game_mode = st.text_input("Performance Mode", value="BOOST MODE")
        
        st.header("🎨 Chart Colors")
        fps_color = st.color_picker("FPS Color", "#FF6600")
        cpu_color = st.color_picker("CPU Color", "#4A90E2")
    
    # File upload
    st.header("📁 Upload Gaming Log CSV")
    uploaded_file = st.file_uploader("Drop your CSV file here", type=['csv'])
    
    if uploaded_file:
        with st.spinner('🔄 Analyzing gaming data...'):
            if generator.load_csv_data(uploaded_file):
                
                # Success message
                st.success("🎉 Gaming log loaded successfully!")
                
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
                
                with st.spinner('🎨 Creating professional chart...'):
                    chart_fig = generator.create_chart(game_title, game_settings, 
                                                     game_mode, fps_color, cpu_color)
                    st.pyplot(chart_fig)
                
                # Performance analysis
                stats = generator.calculate_stats()
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
                
                # Download section
                st.header("💾 Export Results")
                
                # PNG download with transparent background
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
            
            **Features:**
            - ✅ Professional chart generation
            - ✅ High-resolution PNG export  
            - ✅ Gaming performance analysis
            - ✅ Multiple CSV format support
            """)
        
        # Footer
        st.markdown("---")
        st.markdown("Made with ❤️ for gaming performance analysis")

if __name__ == "__main__":
    main()