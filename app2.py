import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io
from datetime import datetime
import warnings
import chardet
import csv

warnings.filterwarnings('ignore')

# --- Helper Functions ---

def csv_restorer_streamlit():
    st.header("🛠️ CSV Restorer")
    st.write("Restore malformed CSV files to proper format with correct delimiters and alignment.")

    uploaded_file = st.file_uploader("Upload CSV to Restore", type=["csv"])
    if uploaded_file:
        lines = uploaded_file.getvalue().decode("utf-8").splitlines()
        preview = "\n".join(lines[:10])
        st.subheader("Input Preview (first 10 rows):")
        st.text_area("Input Preview", preview, height=200)

        reader = csv.reader(lines)
        all_rows = list(reader)

        if len(all_rows) > 1:
            header = all_rows[0]
            first_row = all_rows[1]
            if len(header) != len(first_row):
                st.warning(f"Header has {len(header)} fields, first row has {len(first_row)} fields.")
            else:
                st.success("CSV structure looks OK.")

        if st.button("Restore CSV"):
            new_header = [
                "FPS", "JANK", "BigJANK", "Max FrameTime(ms)", "CPU(%)", "CPU0(%)", "CPU1(%)", "CPU2(%)", "CPU3(%)", 
                "CPU4(%)", "CPU5(%)", "CPU6(%)", "CPU7(%)", "CPU0(MHz)", "CPU1(MHz)", "CPU2(MHz)", "CPU3(MHz)", 
                "CPU4(MHz)", "CPU5(MHz)", "CPU6(MHz)", "CPU7(MHz)", "CPU(°C)", "", "DDR(MHz)", "GPU(%)", "", "GPU(KHz)",
                "Battery(%)", "Battery(°C)", "Battery(mA)", "Battery(volt)"
            ]
            processed_rows = [new_header]
            for row in all_rows[1:]:
                if not row or len(row) < 29:
                    continue
                new_row = []
                for j in range(3):
                    new_row.append(row[j] if j < len(row) else '')
                if 3 < len(row) and 4 < len(row):
                    new_row.append(f"{row[3]}.{row[4]}")
                else:
                    new_row.append(row[3] if 3 < len(row) else '0')
                for j in range(9):
                    if 5 + j < len(row) and 14 + j < len(row):
                        new_row.append(f"{row[5 + j]}.{row[14 + j]}")
                    else:
                        new_row.append(row[5 + j] if 5 + j < len(row) else '0')
                for j in range(8):
                    if 23 + j < len(row):
                        new_row.append(row[23 + j])
                    else:
                        new_row.append('0')
                new_row.append(row[31] if 31 < len(row) else '0')
                new_row.append('0')
                new_row.append(row[32] if 32 < len(row) else '0')
                new_row.append(row[33] if 33 < len(row) else '-1')
                new_row.append('0')
                new_row.append(row[34] if 34 < len(row) else '-1')
                new_row.append(row[35] if 35 < len(row) else '0')
                if 36 < len(row) and 37 < len(row):
                    new_row.append(f"{row[36]}.{row[37]}")
                else:
                    new_row.append(row[36] if 36 < len(row) else '0')
                new_row.append(row[38] if 38 < len(row) else '0')
                battery_volt = row[39] if 39 < len(row) else '0'
                battery_volt = battery_volt.replace(',', '.')
                new_row.append(battery_volt)
                processed_rows.append(new_row)
            preview_text = "\n".join([",".join(row) for row in processed_rows[:10]])
            st.subheader("Output Preview (first 10 rows):")
            st.text_area("Output Preview", preview_text, height=200)
            output_csv = io.StringIO()
            writer = csv.writer(output_csv)
            writer.writerows(processed_rows)
            output_csv.seek(0)
            st.download_button("Download Restored CSV", data=output_csv.getvalue(), file_name="restored.csv", mime="text/csv")
            st.success(f"Restoration complete! Processed {len(processed_rows)-1} rows.")

def render_main_header():
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
    .column-selector {
        background: #1E1E1E;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #333;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="main-header">
        <h1>🎮 Gaming Performance Analyzer</h1>
        <p>Dynamic multi-metric gaming chart generator</p>
    </div>
    """, unsafe_allow_html=True)

def sidebar_game_config():
    st.header("🎮 Game Configuration")
    game_title = st.text_input("🎯 Game Title", value="Mobile Legends: Bang Bang")
    game_settings = st.text_input("⚙️ Graphics Settings", value="Ultra - 120 FPS")
    game_mode = st.text_input("🚀 Performance Mode", value="Game Boost Mode")
    smartphone_name = st.text_input("📱 Device Model", value="iPhone 15 Pro Max")
    st.header("🕹️ FPS Setting")
    fps_options = [30, 60, 90, 120, 144]
    selected_fps_setting = st.selectbox("🎮 Pilih FPS Game yang Digunakan", fps_options, index=3)
    st.divider()
    st.header("🔧 Quick Data Processing")
    enable_outlier_removal = st.toggle("🚫 Remove Outliers", value=False)
    outlier_method, outlier_threshold = None, None
    if enable_outlier_removal:
        outlier_method = st.selectbox("Method", ['percentile', 'iqr', 'zscore'])
        if outlier_method == 'percentile':
            outlier_threshold = st.slider("Bottom Percentile", 0.1, 5.0, 1.0, 0.1)
        elif outlier_method == 'zscore':
            outlier_threshold = st.slider("Z-Score Threshold", 1.0, 4.0, 2.0, 0.1)
        else:
            outlier_threshold = 1.5
    return game_title, game_settings, game_mode, smartphone_name, selected_fps_setting, enable_outlier_removal, outlier_method, outlier_threshold

def show_metric_card(label, value, delta=None, delta_color="normal"):
    st.markdown(
        f"""
        <div class="metric-card">
            <h4>{label}</h4>
            <h2>{value}</h2>
            {"<p style='color: #aaa;'>Δ " + str(delta) + "</p>" if delta else ""}
        </div>
        """, unsafe_allow_html=True
    )

def color_picker_row(selected_columns, analyzer):
    color_pickers = {}
    hide_flags = {}
    color_cols = st.columns(min(len(selected_columns), 4))
    for i, col in enumerate(selected_columns):
        col_display = analyzer.get_column_display_name(col)
        suggested_color = analyzer.get_column_color_suggestion(col)
        with color_cols[i % 4]:
            color_pickers[col] = st.color_picker(
                f"{col_display[:10]}...", 
                suggested_color,
                key=f"color_{col}_{i}",
                help=f"Color for {col_display}"
            )
            hide_flags[col] = st.checkbox(
                f"Hide {col_display[:15]}...",
                value=False,
                key=f"hide_{col}_{i}",
                help=f"Hide {col_display} from chart"
            )
    return color_pickers, hide_flags

# --- Main Analyzer Class ---
class GamingPerformanceAnalyzer:
    def __init__(self, fps_setting=60):
        self.selected_fps_setting = fps_setting
        self.original_data = None
        self.processed_data = None
        self.removed_indices = []
        self.debug_mode = False
        self.available_columns = []
        self.numeric_columns = []

    def load_csv_data(self, file_data, filename):
        """Load and validate CSV data with smart encoding & flexible delimiter detection"""
        # Detect encoding
        encoding_guess = chardet.detect(file_data)['encoding']
        delimiters = [',', ';', '\t', '|', r'\s+']
        for delimiter in delimiters:
            try:
                if delimiter == r'\s+':
                    df = pd.read_csv(io.BytesIO(file_data), delimiter=delimiter, engine='python', encoding=encoding_guess)
                else:
                    df = pd.read_csv(io.BytesIO(file_data), delimiter=delimiter, encoding=encoding_guess)
                if len(df.columns) > 1 and len(df) > 0:
                    if self._validate_and_process_columns(df, delimiter, encoding_guess):
                        return True
            except Exception:
                continue
        st.error("❌ Gagal membaca CSV. Format atau delimiter tidak dikenali.")
        return False

    def _validate_and_process_columns(self, df, delimiter, encoding):
        self.available_columns = list(df.columns)
        processed_df = pd.DataFrame()
        self.numeric_columns = []
        # Try to convert to numeric for each column
        for col in self.available_columns:
            numeric_data = pd.to_numeric(df[col], errors='coerce')
            numeric_data = numeric_data.where(numeric_data >= 0)
            valid_numeric_ratio = (~numeric_data.isna()).sum() / len(df)
            if valid_numeric_ratio >= 0.5 and not numeric_data.isna().all() and np.isfinite(numeric_data.max()):
                processed_df[col] = numeric_data
                self.numeric_columns.append(col)
            else:
                processed_df[col] = df[col]
        if len(self.numeric_columns) < 1:
            st.error("❌ No numeric columns found for analysis")
            return False
        processed_df['TimeMinutes'] = np.arange(len(processed_df)) / self.selected_fps_setting
        self.original_data = processed_df.reset_index(drop=True)
        st.success(f"✅ Data loaded successfully using delimiter '{delimiter}' and encoding '{encoding}'")
        return True

    def get_column_display_name(self, col_name):
        display_names = {
            'fps': 'FPS', 'cpu': 'CPU Usage (%)', 'gpu': 'GPU Usage (%)', 
            'ram': 'RAM Usage (%)', 'memory': 'Memory Usage (%)',
            'temp': 'Temperature (°C)', 'temperature': 'Temperature (°C)',
            'power': 'Power Consumption (W)', 'battery': 'Battery (%)',
            'jank': 'Jank Count', 'frame_drops': 'Frame Drops', 'latency': 'Latency (ms)',
            'ping': 'Ping (ms)', 'network': 'Network Usage', 'bandwidth': 'Bandwidth',
            'thermal': 'Thermal State'
        }
        col_lower = col_name.lower()
        for key, display in display_names.items():
            if key in col_lower:
                return display
        return col_name.replace('_', ' ').title()

    def get_column_color_suggestion(self, col_name):
        color_map = {
            'fps': '#00D4FF', 'cpu': '#FF6B35', 'gpu': '#4ECDC4', 'ram': '#45B7D1',
            'memory': '#45B7D1', 'temp': '#FF4757', 'temperature': '#FF4757',
            'power': '#FFA726', 'battery': '#66BB6A', 'jank': '#8E24AA', 'frame_drops': '#D32F2F',
            'latency': '#FF9800', 'ping': '#FF9800', 'network': '#9C27B0', 'bandwidth': '#9C27B0',
            'thermal': '#E91E63'
        }
        col_lower = col_name.lower()
        for key, color in color_map.items():
            if key in col_lower:
                return color
        default_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8']
        return default_colors[hash(col_name) % len(default_colors)]

    def remove_outliers(self, method='percentile', threshold=1, target_columns=None):
        if self.original_data is None:
            return False
        if target_columns is None:
            target_columns = self.numeric_columns
        data = self.original_data.copy()
        combined_mask = pd.Series([True] * len(data))
        for col in target_columns:
            if col not in data.columns or col not in self.numeric_columns:
                continue
            col_data = data[col]
            if method == 'percentile':
                percent = np.percentile(col_data.dropna(), threshold)
                keep_mask = col_data >= percent
            elif method == 'iqr':
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                keep_mask = (col_data >= lower) & (col_data <= upper)
            elif method == 'zscore':
                z_scores = np.abs((col_data - col_data.mean()) / col_data.std(ddof=0))
                keep_mask = z_scores <= threshold
            else:
                keep_mask = pd.Series([True] * len(data))
            combined_mask = combined_mask & keep_mask
        self.processed_data = data[combined_mask].reset_index(drop=True)
        self.removed_indices = data[~combined_mask].index.tolist()
        self.processed_data['TimeMinutes'] = np.arange(len(self.processed_data)) / self.selected_fps_setting
        return True

    def create_performance_chart(self, config):
        data = self.processed_data if self.processed_data is not None else self.original_data
        if data is None:
            st.error("❌ No data available for chart generation")
            return None
        selected_columns = [col for col in config['selected_columns'] if not config.get(f'hide_{col}', False)]
        if not selected_columns:
            st.error("❌ No columns selected for plotting")
            return None
        plt.style.use('dark_background')
        fig, ax1 = plt.subplots(figsize=(16, 9))
        fig.patch.set_facecolor('#0E1117')
        time_data = data['TimeMinutes']
        axes = [ax1]
        colors_used = []
        for i, col in enumerate(selected_columns):
            if col not in data.columns or col not in self.numeric_columns:
                continue
            ax = ax1 if i == 0 else ax1.twinx()
            if i > 1:
                ax.spines['right'].set_position(('outward', 60 * (i-1)))
            axes.append(ax)
            col_data = data[col]
            col_color = config.get(f'{col}_color', self.get_column_color_suggestion(col))
            col_display = self.get_column_display_name(col)
            ax.plot(time_data, col_data, color=col_color, linewidth=2.5, label=col_display, alpha=0.8, zorder=len(selected_columns)-i)
            ax.set_ylabel(col_display, fontsize=12, color=col_color, fontweight='bold')
            ax.tick_params(axis='y', colors='black', labelsize=10)
            if np.isfinite(col_data.max()) and not col_data.dropna().empty:
                data_max = col_data.max()
                padding = data_max * 0.1
                ax.set_ylim(0, data_max + padding)
            colors_used.append((col_display, col_color))
        ax1.set_xlabel('Time (minutes)', fontsize=12, color='white', fontweight='bold')
        ax1.tick_params(axis='x', colors='black', labelsize=10)
        ax1.grid(True, alpha=0.3, linestyle='--', color='gray')
        ax1.set_facecolor('#0E1117')
        title_lines = [config['game_title']]
        if config['game_settings']:
            title_lines.append(config['game_settings'])
        if config['game_mode']:
            title_lines.append(config['game_mode'])
        plt.suptitle('\n'.join(title_lines), fontsize=20, fontweight='bold', color='white', y=0.95)
        legend_elements = []
        if config['smartphone_name']:
            legend_elements.append(plt.Line2D([0], [0], color='white', label=config['smartphone_name']))
        for col_display, col_color in colors_used:
            legend_elements.append(plt.Line2D([0], [0], color=col_color, linewidth=2.5, label=col_display))
        legend = ax1.legend(handles=legend_elements, loc='upper right', framealpha=0.9, fancybox=True, bbox_to_anchor=(1.0, 1.0))
        legend.get_frame().set_facecolor('#262730')
        for text in legend.get_texts():
            text.set_color('white')
            if config['smartphone_name'] and text.get_text() == config['smartphone_name']:
                text.set_fontweight('bold')
        for ax in axes:
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            if ax != ax1:
                ax.spines['right'].set_color('white')
                ax.spines['right'].set_visible(True)
            else:
                ax.spines['right'].set_visible(False)
        plt.tight_layout()
        return fig

    def get_performance_stats(self, selected_columns=None):
        if self.original_data is None:
            return {}
        data = self.processed_data if self.processed_data is not None else self.original_data
        if selected_columns is None:
            selected_columns = self.numeric_columns
        stats = {
            'duration': len(data) / self.selected_fps_setting,
            'total_frames': len(data),
            'removed_frames': len(self.removed_indices)
        }
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
            if 'fps' in col.lower():
                stats['fps_60_plus'] = (col_data >= 60).sum() / len(col_data) * 100
                stats['frame_drops'] = (col_data < 30).sum()
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

    def create_shadow_table(self, selected_columns=None):
        data = self.processed_data if self.processed_data is not None else self.original_data
        if data is None:
            return None, None
        if selected_columns is None:
            selected_columns = self.numeric_columns
        # Use the user-selected FPS setting for frame calculation
        fps = self.selected_fps_setting
        shadow_table = pd.DataFrame({
            'Frame': range(1, len(data) + 1),
            'Time': (data.index / 60).round(4)
        })
        for col in selected_columns:
            if col in data.columns and col in self.numeric_columns:
                shadow_table[col] = data[col].round(1)
        return shadow_table, fps

    def export_processed_data(self, game_title, selected_columns=None):
        data = self.processed_data if self.processed_data is not None else self.original_data
        if data is None:
            return None, None
        if selected_columns is None:
            selected_columns = self.numeric_columns
        export_data = pd.DataFrame({'Time_Minutes': data['TimeMinutes'].round(3)})
        for col in selected_columns:
            if col in data.columns and col in self.numeric_columns:
                export_data[col.replace(' ', '_')] = data[col].round(1)
        csv_buffer = io.StringIO()
        export_data.to_csv(csv_buffer, index=False)
        csv_content = csv_buffer.getvalue()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{game_title.replace(' ', '_')}_processed_{timestamp}.csv"
        return csv_content, filename

# --- Main App Logic ---
def main():
    st.set_page_config(page_title="Gaming Performance Analyzer", layout="wide")
    render_main_header()
    menu = st.sidebar.radio("Select Menu", ["Analyzer", "CSV Restorer"])
    if menu == "Analyzer":
        # Your existing analyzer logic (leave unchanged)
        ...
    elif menu == "CSV Restorer":
        csv_restorer_streamlit()
    # Footer and docs as before
    (
        game_title,
        game_settings,
        game_mode,
        smartphone_name,
        selected_fps_setting,
        enable_outlier_removal,
        outlier_method,
        outlier_threshold,
    ) = sidebar_game_config()
    analyzer = GamingPerformanceAnalyzer(fps_setting=selected_fps_setting)
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("📁 Upload Performance Data")
        uploaded_file = st.file_uploader(
            "Upload your gaming log CSV file",
            type=['csv'],
            help="CSV should contain numeric performance columns"
        )
        if uploaded_file:
            file_data = uploaded_file.read()
            with st.spinner('📊 Loading and analyzing data structure...'):
                if analyzer.load_csv_data(file_data, uploaded_file.name):
                    if analyzer.original_data is not None and analyzer.numeric_columns:
                        st.subheader("📊 Select Metrics to Display")
                        st.markdown('<div class="column-selector">', unsafe_allow_html=True)
                        default_columns = analyzer.numeric_columns[:4]
                        selected_columns = st.multiselect(
                            "Choose metrics to analyze:",
                            analyzer.numeric_columns,
                            default=default_columns,
                            help="Select one or more numeric columns to display on the chart"
                        )
                        if selected_columns:
                            color_pickers, hide_flags = color_picker_row(selected_columns, analyzer)
                            chart_config = {
                                'game_title': game_title,
                                'game_settings': game_settings,
                                'game_mode': game_mode,
                                'smartphone_name': smartphone_name,
                                'selected_columns': selected_columns
                            }
                            for col in selected_columns:
                                chart_config[f'{col}_color'] = color_pickers[col]
                                chart_config[f'hide_{col}'] = hide_flags[col]
                            st.markdown('</div>', unsafe_allow_html=True)
                            # Outlier removal
                            if enable_outlier_removal and outlier_method:
                                with st.spinner('🔧 Removing outliers...'):
                                    analyzer.remove_outliers(outlier_method, outlier_threshold, selected_columns)
                            else:
                                analyzer.processed_data = analyzer.original_data.copy()
                                st.info("📊 **Raw Mode**: Using original data without processing")
                            # Chart
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
        if uploaded_file and analyzer.original_data is not None:
            st.subheader("📈 Performance Statistics")
            if 'selected_columns' in locals() and selected_columns:
                stats = analyzer.get_performance_stats(selected_columns)
                if any('fps' in col.lower() for col in selected_columns) and 'grade' in stats:
                    st.markdown(
                        f"""
                        <div style="text-align: center; padding: 1rem; background: linear-gradient(45deg, #667eea, #764ba2);
                                    border-radius: 10px; margin-bottom: 1rem;">
                            <h3 style="color: white; margin: 0;">{stats['grade']}</h3>
                            <p style="color: white; margin: 0;">Overall Performance</p>
                        </div>
                        """, unsafe_allow_html=True)
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
                if 'fps_60_plus' in stats:
                    st.metric("🎮 60+ FPS Time", f"{stats['fps_60_plus']:.1f}%")
                if 'frame_drops' in stats:
                    st.metric("⚠️ Frame Drops", f"{stats['frame_drops']}")
                st.metric("⏱️ Duration", f"{stats['duration']:.1f} min")
                if stats['removed_frames'] > 0:
                    st.metric("🚫 Removed Frames", f"{stats['removed_frames']}")
                st.subheader("💾 Export Options")
                # Chart export (transparent background)
                if 'chart_fig' in locals() and chart_fig:
                    img_buffer = io.BytesIO()
                    chart_fig.savefig(
                        img_buffer, format='png', dpi=300, bbox_inches='tight', transparent=True
                    )
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
                # Shadow table preview and export (using selected FPS)
                shadow_table, selected_fps = analyzer.create_shadow_table(selected_columns)
                if shadow_table is not None:
                    st.subheader("🎬 Video Chart Table")
                    st.info(f"📹 **Exported Frame Rate**: {selected_fps} FPS (matches your selected setting)")
                    with st.expander("👀 Preview Video Chart Data (First 10 rows)"):
                        st.dataframe(shadow_table.head(10), use_container_width=True)
                    shadow_csv = shadow_table.to_csv(index=False)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    shadow_filename = f"{game_title.replace(' ', '_')}_video_chart_{timestamp}.csv"
                    st.download_button(
                        label="🎬 Download Video Chart CSV",
                        data=shadow_csv,
                        file_name=shadow_filename,
                        mime="text/csv",
                        use_container_width=True,
                        help="Perfect for creating animated video charts with all selected metrics"
                    )
                # Regular data export
                csv_content, csv_filename = analyzer.export_processed_data(game_title, selected_columns)
                if csv_content:
                    st.download_button(
                        label="📄 Download Processed Data",
                        data=csv_content,
                        file_name=csv_filename,
                        mime="text/csv",
                        use_container_width=True
                    )
            else:
                st.info("📊 Select metrics from the main panel to see statistics")
        else:
            st.info("📤 Upload CSV file to see performance statistics")
    # Documentation & Footer
    with st.expander("📋 CSV Format & Dynamic Features"):
        st.markdown("""
        **🚀 NEW: Dynamic Column Detection**
        - **Auto-detects** all numeric columns in your CSV
        - **Multi-metric support**: FPS, CPU, GPU, RAM, Temperature, etc.
        - **Custom axis creation**: Each metric gets its own Y-axis
        - **Smart color suggestions**: Based on metric type

        **📊 Supported Metric Types:**
        - **Performance**: FPS, Frame Drops, Jank Count
        - **Hardware**: CPU(%), GPU(%), RAM(%), Memory(%)
        - **Thermal**: Temperature, Thermal State
        - **Power**: Battery(%), Power Consumption
        - **Network**: Latency, Ping, Bandwidth
        - **Custom**: Any numeric column

        **📁 Example CSV Formats:**
        ```
        FPS,CPU(%),GPU(%),Temp,RAM(%)
        60,45.2,67.8,42.1,58.3
        58,48.1,70.2,43.5,59.1
        ```

        **✨ Features:**
        - **Dynamic axis scaling**: Automatic Y-axis adjustment
        - **Color customization**: Per-metric color control
        - **Show/Hide toggles**: Control visibility per metric
        - **Multi-column outlier removal**: Apply to selected metrics only
        """)
    with st.expander("🎨 Chart Customization Guide"):
        st.markdown("""
        **🎯 Metric Selection:**
        - Choose **1-6 metrics** for optimal readability
        - **Primary metrics** (FPS, CPU) get prominent positioning
        - **Secondary metrics** get additional Y-axes

        **🌈 Color Strategy:**
        - **FPS**: Cyan/Blue (performance focus)
        - **CPU/GPU**: Orange/Red (thermal colors)
        - **RAM/Memory**: Blue shades (data colors)
        - **Temperature**: Red/Pink (heat colors)
        - **Battery**: Green (energy colors)

        **📊 Multi-Axis Layout:**
        - **Left axis**: Primary metric (usually FPS)
        - **Right axis**: Secondary metric (usually CPU)
        - **Additional axes**: Offset to the right
        - **Legend**: Shows all active metrics
        """)
    with st.expander("🎬 Video Chart Export"):
        st.markdown("""
        **📹 Enhanced Video Chart Features:**
        - **Multi-metric support**: All selected metrics in one CSV
        - **Frame-perfect timing**: Synced to detected FPS
        - **Ready for animation**: Frame-by-frame progression

        **🎥 Recommended Workflow:**
        1. **Select metrics** you want in your video
        2. **Customize colors** for video consistency
        3. **Download Video Chart CSV**
        4. **Import to video editor** (After Effects, Premiere Pro)
        5. **Create animated charts** with multiple metrics

        **📊 Video Chart Columns:**
        - **Frame**: Sequential numbering (1, 2, 3...)
        - **Time**: Frame converted to minutes
        - **[Metric columns]**: All selected metrics with values

        **🎨 Animation Ideas:**
        - **Racing bar charts**: Show metric competition over time
        - **Multi-line graphs**: Animated line progression
        - **Gauge animations**: Circular metric displays
        - **Heatmaps**: Color-coded performance zones
        """)
    with st.expander("🔧 Advanced Processing"):
        st.markdown("""
        **🚫 Outlier Removal Options:**
        - **Per-metric control**: Choose which metrics to clean
        - **Multiple methods**: Percentile, IQR, Z-Score
        - **Selective processing**: Keep some metrics raw

        **📊 Processing Strategies:**
        - **FPS cleaning**: Remove frame drops and stutters
        - **CPU/GPU smoothing**: Remove usage spikes
        - **Temperature filtering**: Remove sensor glitches
        - **Mixed approach**: Different methods per metric

        **⚡ Performance Tips:**
        - **Start simple**: 2-3 metrics for first analysis
        - **Color contrast**: Ensure visibility on dark background
        - **Axis scaling**: Check auto-scaling results
        - **Export timing**: Process before video creation
        """)
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #888; padding: 1rem;">
        🎮 Gaming Performance Analyzer v4.0 - Dynamic Multi-Metric Edition<br>
        <small>📊 Dynamic Columns • 🎨 Multi-Axis Charts • 🎬 Video Export • 📈 Advanced Analytics</small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
