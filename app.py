import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io
from datetime import datetime

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
    
    def load_csv_data(self, uploaded_file):
        """Load CSV with professional gaming data processing"""
        try:
            # Read CSV
            uploaded_file.seek(0)
            self.original_data = pd.read_csv(uploaded_file)
            
            # Check for required columns
            if 'FPS' not in self.original_data.columns:
                st.error("❌ Required 'FPS' column not found!")
                return False
            
            if 'CPU(%)' not in self.original_data.columns:
                st.error("❌ Required 'CPU(%)' column not found!")
                return False
            
            # Extract data
            fps_data = pd.to_numeric(self.original_data['FPS'], errors='coerce')
            cpu_data = pd.to_numeric(self.original_data['CPU(%)'], errors='coerce')
            
            # Basic cleaning - fill NaN with interpolation
            fps_data = fps_data.interpolate().fillna(method='bfill').fillna(method='ffill')
            cpu_data = cpu_data.interpolate().fillna(method='bfill').fillna(method='ffill')
            
            # Create time column
            time_minutes = [i / 60 for i in range(len(self.original_data))]
            
            # Store processed data
            self.processed_data = pd.DataFrame({
                'FPS': fps_data,
                'CPU(%)': cpu_data,
                'TimeMinutes': time_minutes
            })
            
            st.success(f"✅ Data loaded: {len(self.processed_data)} rows")
            
            # Show preview
            col1, col2 = st.columns(2)
            with col1:
                st.metric("📈 FPS Range", f"{fps_data.min():.1f} - {fps_data.max():.1f}")
                st.metric("📊 FPS Average", f"{fps_data.mean():.1f}")
            with col2:
                st.metric("🖥️ CPU Range", f"{cpu_data.min():.1f}% - {cpu_data.max():.1f}%")
                st.metric("📊 CPU Average", f"{cpu_data.mean():.1f}%")
            
            return True
            
        except Exception as e:
            st.error(f"❌ Error loading CSV: {e}")
            return False
    
    def get_fps_axis_max(self, max_fps):
        """Determine FPS axis maximum based on data maximum"""
        if max_fps <= 31:
            return 30
        elif max_fps >= 145:
            return 144
        elif max_fps >= 121:
            return 120
        elif max_fps >= 91:
            return 90
        elif max_fps >= 61:
            return 60
        else:
            return 30
    
    def format_time_label(self, seconds):
        """Format time label according to requirements"""
        total_seconds = int(seconds)
        
        if total_seconds < 60:
            return f"{total_seconds} s"
        elif total_seconds < 3600:
            minutes = total_seconds // 60
            remaining_seconds = total_seconds % 60
            return f"{minutes} m {remaining_seconds} s"
        else:
            hours = total_seconds // 3600
            remaining_minutes = (total_seconds % 3600) // 60
            return f"{hours} h {remaining_minutes} m"
    
    def create_professional_chart(self, game_title, game_settings, game_mode, smartphone_name,
                                fps_color, cpu_color):
        """Create professional gaming chart"""
        
        if self.processed_data is None:
            st.error("❌ No data available")
            return None
        
        # Get data
        time_data = self.processed_data['TimeMinutes']
        fps_data = self.processed_data['FPS']
        cpu_data = self.processed_data['CPU(%)']
        time_seconds = time_data * 60
        
        # Create figure
        fig, ax1 = plt.subplots(figsize=(16, 9))
        fig.patch.set_facecolor('#1e1e1e')
        
        # Primary axis (FPS)
        ax1.set_xlabel('Time', fontsize=14, fontweight='bold', color='white')
        ax1.set_ylabel('FPS', color=fps_color, fontsize=14, fontweight='bold')
        ax1.tick_params(axis='y', labelcolor=fps_color, labelsize=12)
        ax1.tick_params(axis='x', labelcolor='white', labelsize=12)
        
        # Secondary axis (CPU)
        ax2 = ax1.twinx()
        ax2.set_ylabel('CPU Usage (%)', color=cpu_color, fontsize=14, fontweight='bold')
        ax2.tick_params(axis='y', labelcolor=cpu_color, labelsize=12)
        ax2.set_ylim(0, 100)
        
        # Plot data
        ax1.plot(time_seconds, fps_data, color=fps_color, linewidth=2.5, alpha=0.9, label='FPS')
        ax2.plot(time_seconds, cpu_data, color=cpu_color, linewidth=2.5, alpha=0.9, label='CPU')
        
        # Fill areas
        ax1.fill_between(time_seconds, 0, fps_data, color=fps_color, alpha=0.15)
        ax2.fill_between(time_seconds, 0, cpu_data, color=cpu_color, alpha=0.15)
        
        # Set FPS axis limits
        max_fps = fps_data.max()
        fps_axis_max = self.get_fps_axis_max(max_fps)
        ax1.set_ylim(0, fps_axis_max)
        
        # Time formatting
        max_time_seconds = time_seconds.max()
        num_ticks = 6
        tick_interval = max_time_seconds / (num_ticks - 1)
        tick_positions = [i * tick_interval for i in range(num_ticks)]
        tick_labels = [self.format_time_label(pos) for pos in tick_positions]
        
        ax1.set_xticks(tick_positions)
        ax1.set_xticklabels(tick_labels)
        ax1.set_xlim(0, max_time_seconds)
        
        # Title and styling
        title_text = f"{game_title}\n{game_settings}\n{game_mode}"
        plt.suptitle(title_text, fontsize=18, fontweight='bold', y=0.95, color='white')
        
        # Grid and background
        ax1.grid(True, alpha=0.3, color='white', linewidth=0.5)
        ax1.set_facecolor('#1e1e1e')
        
        # Legend
        legend_elements = [
            plt.Line2D([0], [0], color='white', linewidth=0, marker='s', 
                      markersize=8, markerfacecolor='none', markeredgecolor='white', 
                      label=smartphone_name),
            plt.Line2D([0], [0], color=fps_color, linewidth=3, label='FPS'),
            plt.Line2D([0], [0], color=cpu_color, linewidth=3, label='CPU')
        ]
        
        legend = ax1.legend(handles=legend_elements, loc='upper right', 
                          framealpha=0.9, facecolor='#2d2d2d', edgecolor='white')
        
        for text in legend.get_texts():
            text.set_color('white')
        
        # Clean spines
        for spine in ax1.spines.values():
            spine.set_color('white')
            spine.set_linewidth(0.5)
        for spine in ax2.spines.values():
            spine.set_color('white')
            spine.set_linewidth(0.5)
        
        # Layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        
        return fig
    
    def get_statistics(self):
        """Get performance statistics"""
        if self.processed_data is None:
            return {}
        
        fps_data = self.processed_data['FPS']
        cpu_data = self.processed_data['CPU(%)']
        duration = len(self.processed_data) / 60
        
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
        
        fps_above_60_pct = (len(fps_data[fps_data >= 60]) / len(fps_data)) * 100
        
        return {
            'grade': grade,
            'duration': round(duration, 1),
            'avg_fps': round(avg_fps, 1),
            'min_fps': round(fps_data.min(), 1),
            'max_fps': round(fps_data.max(), 1),
            'avg_cpu': round(cpu_data.mean(), 1),
            'max_cpu': round(cpu_data.max(), 1),
            'fps_above_60': round(fps_above_60_pct, 1),
            'frame_drops': len(fps_data[fps_data < 30])
        }

def main():
    # Header
    st.title("🎮 Professional Gaming Chart Generator")
    st.markdown("Create **professional gaming charts** like mobile gaming monitoring apps")
    
    # Initialize
    generator = ProfessionalGamingChartGenerator()
    
    # Sidebar
    with st.sidebar:
        st.header("🎮 Game Configuration")
        game_title = st.text_input("Game Title", value="MOBILE LEGENDS BANG BANG")
        game_settings = st.text_input("Graphics Settings", value="ULTRA - 60 FPS")
        game_mode = st.text_input("Performance Mode", value="BOOST MODE")
        smartphone_name = st.text_input("Smartphone Model", value="Gaming Phone")
        
        st.header("🎨 Chart Colors")
        fps_color = st.color_picker("FPS Color", "#4A90E2")
        cpu_color = st.color_picker("CPU Color", "#FF6600")
    
    # File upload
    st.header("📁 Upload Gaming Log CSV")
    uploaded_file = st.file_uploader("Drop your CSV file here", type=['csv'])
    
    if uploaded_file:
        # Load data
        if generator.load_csv_data(uploaded_file):
            
            # Show stats
            data = generator.processed_data
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📊 Data Points", f"{len(data):,}")
            with col2:
                st.metric("⏱️ Duration", f"{len(data)/60:.1f} min")
            with col3:
                st.metric("🎯 Avg FPS", f"{data['FPS'].mean():.1f}")
            with col4:
                st.metric("🖥️ Avg CPU", f"{data['CPU(%)'].mean():.1f}%")
            
            # Create chart
            st.header("📊 Professional Gaming Performance Chart")
            
            chart_fig = generator.create_professional_chart(
                game_title, game_settings, game_mode, smartphone_name,
                fps_color, cpu_color
            )
            
            if chart_fig:
                st.pyplot(chart_fig, use_container_width=True)
                
                # Statistics
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
                
                # Export
                st.header("💾 Export Chart")
                
                img_buffer = io.BytesIO()
                chart_fig.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight',
                                 facecolor='#1e1e1e', edgecolor='none')
                img_buffer.seek(0)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{game_title.replace(' ', '_')}_chart_{timestamp}.png"
                
                st.download_button(
                    label="📸 Download Professional Chart",
                    data=img_buffer.getvalue(),
                    file_name=filename,
                    mime="image/png"
                )
            else:
                st.error("❌ Failed to create chart")
    
    else:
        st.info("📤 Upload your Mobile Legends Bang Bang CSV file to create professional charts!")
        
        with st.expander("📋 CSV Requirements"):
            st.markdown("""
            **Required columns:**
            - **FPS** - Frame rate data
            - **CPU(%)** - CPU usage percentage
            
            **Your Mobile Legends Bang Bang CSV is already compatible!** ✅
            """)

if __name__ == "__main__":
    main()
