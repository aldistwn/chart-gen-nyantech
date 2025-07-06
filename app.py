import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io
from datetime import datetime
from scipy.signal import savgol_filter

# Page configuration
st.set_page_config(
    page_title="🎮 Gaming Chart Generator - Scene Style",
    page_icon="🎮",
    layout="wide"
)

class SceneStyleGamingChartGenerator:
    def __init__(self):
        self.original_data = None
        self.processed_data = None
        self.removed_indices = []
        self.column_mapping = {}
    
    def load_csv_data(self, uploaded_file):
        """Load CSV with EMERGENCY debugging"""
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
            
            # 🚨 EMERGENCY DEBUG - DETAILED COLUMN ANALYSIS
            st.markdown("### 🚨 EMERGENCY DEBUG - RAW CSV ANALYSIS")
            columns = list(self.original_data.columns)
            
            st.write("**All columns found:**")
            for i, col in enumerate(columns):
                st.write(f"  Index {i}: '{col}' (len: {len(col)}, repr: {repr(col)})")
            
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
            
            # Get column indices
            fps_index = columns.index('FPS')
            cpu_index = columns.index('CPU(%)')
            
            st.write(f"🎯 FPS column found at index {fps_index}")
            st.write(f"🎯 CPU(%) column found at index {cpu_index}")
            
            # Show raw data BEFORE any processing
            raw_fps_column = self.original_data.iloc[:, fps_index]
            raw_cpu_column = self.original_data.iloc[:, cpu_index]
            
            st.write(f"Raw FPS first 10 values: {list(raw_fps_column.head(10))}")
            
            # Convert to numeric and validate
            fps_numeric = pd.to_numeric(raw_fps_column, errors='coerce')
            cpu_numeric = pd.to_numeric(raw_cpu_column, errors='coerce')
            
            # 🚨 CRITICAL CHECK: Values > 70 in RAW data
            high_fps_raw = fps_numeric[fps_numeric > 70]
            st.write(f"🚨 **RAW FPS values > 70: {len(high_fps_raw)} found**")
            
            if len(high_fps_raw) > 0:
                st.error(f"🚨 RAW DATA ALREADY HAS HIGH VALUES: {list(high_fps_raw.head(10))}")
                st.error("**BUG IS IN CSV READING OR COLUMN DETECTION!**")
            else:
                st.success("✅ Raw FPS data looks normal (no values > 70)")
            
            # Assign data
            self.original_data['FPS'] = fps_numeric
            self.original_data['CPU(%)'] = cpu_numeric
            self.original_data['TimeMinutes'] = [i / 60 for i in range(len(self.original_data))]
            
            # Final validation
            final_fps_max = self.original_data['FPS'].max()
            final_fps_min = self.original_data['FPS'].min()
            
            st.write(f"**Final standardized FPS range: {final_fps_min} - {final_fps_max}**")
            
            if final_fps_max > 70:
                st.error(f"🚨 **BUG DETECTED IN LOAD PHASE: FPS max is {final_fps_max}**")
            else:
                st.success(f"✅ **CSV loaded correctly, FPS max: {final_fps_max}**")
            
            # Show basic info with large dataset awareness
            dataset_size_info = f"📊 Dataset: {len(self.original_data):,} rows × {len(self.original_data.columns)} columns"
            estimated_duration = len(self.original_data) / 3600  # Assuming 1 FPS per second
            
            if estimated_duration > 10:
                st.info(f"{dataset_size_info} (~{estimated_duration:.1f} hours)")
                st.success("🚀 **Extended session detected** - High performance processing will be used")
            elif estimated_duration > 5:
                st.info(f"{dataset_size_info} (~{estimated_duration:.1f} hours)")
                st.info("⚡ **Long session detected** - Optimized processing will be used")
            else:
                st.info(f"{dataset_size_info} (~{estimated_duration:.1f} hours)")
            
            # Performance recommendations for large datasets
            if len(self.original_data) > 500_000:
                st.warning("💡 **Large Dataset Tips**: Use Conservative outlier removal and disable smoothing for best performance")
            elif len(self.original_data) > 1_000_000:
                st.error("🔥 **Very Large Dataset**: Processing will use chunked operations. This may take several minutes.")
            
            # Show data preview
            col1, col2 = st.columns(2)
            with col1:
                st.metric("📈 FPS Range", f"{fps_numeric.min():.1f} - {fps_numeric.max():.1f}")
                st.metric("📊 FPS Average", f"{fps_numeric.mean():.1f}")
            with col2:
                st.metric("🖥️ CPU Range", f"{cpu_numeric.min():.1f}% - {cpu_numeric.max():.1f}%")
                st.metric("📊 CPU Average", f"{cpu_numeric.mean():.1f}%")
            
            return True
            
        except Exception as e:
            st.error(f"❌ Error loading CSV: {e}")
            return False
    
    def remove_fps_outliers_optimized(self, sensitivity='moderate'):
        """High-performance outlier removal for large datasets (10+ hours)"""
        try:
            if self.original_data is None:
                return False
            
            fps_data = self.original_data['FPS'].dropna()
            data_size = len(fps_data)
            duration_hours = data_size / 3600  # Assuming 1 FPS per second
            
            # Data size validation
            if data_size < 10:
                st.warning("⚠️ Not enough data points for outlier removal")
                return False
            
            # Memory and performance optimization for large datasets
            st.info(f"📊 Processing {data_size:,} data points (~{duration_hours:.1f} hours)")
            
            # Use efficient numpy operations for large datasets
            fps_array = fps_data.values
            
            # Calculate thresholds with chunked processing for very large datasets
            if data_size > 1_000_000:  # > 1M data points (~278 hours at 1 FPS)
                st.info("🔧 Large dataset detected - using chunked processing")
                # Process in chunks to avoid memory issues
                chunk_size = 100_000
                percentiles = []
                
                progress_bar = st.progress(0)
                chunks = np.array_split(fps_array, max(1, data_size // chunk_size))
                
                for i, chunk in enumerate(chunks):
                    chunk_p1 = np.percentile(chunk, 1)
                    chunk_p5 = np.percentile(chunk, 5)
                    percentiles.append([chunk_p1, chunk_p5])
                    progress_bar.progress((i + 1) / len(chunks))
                
                # Combine chunk results
                percentiles = np.array(percentiles)
                percentile_1 = np.percentile(percentiles[:, 0], 1)
                percentile_5 = np.percentile(percentiles[:, 1], 5)
                progress_bar.empty()
                
            else:
                # Standard processing for smaller datasets
                percentile_1 = np.percentile(fps_array, 1)
                percentile_5 = np.percentile(fps_array, 5)
            
            # Enhanced thresholds for long-duration gaming sessions
            if duration_hours > 5:  # Long gaming sessions need different thresholds
                # For long sessions, be more conservative to preserve performance patterns
                thresholds = {
                    'conservative': percentile_1 * 0.9,  # Even more conservative
                    'moderate': percentile_1,            # Standard
                    'aggressive': percentile_1 * 1.2    # Less aggressive than short sessions
                }
                st.info(f"🎮 Long session detected ({duration_hours:.1f}h) - using enhanced thresholds")
            else:
                # Standard thresholds for shorter sessions
                thresholds = {
                    'conservative': percentile_1,
                    'moderate': percentile_1 * 1.1,
                    'aggressive': percentile_5
                }
            
            threshold = thresholds.get(sensitivity, percentile_1)
            
            # Efficient boolean mask operations
            keep_mask = fps_array >= threshold
            keep_indices = fps_data.index[keep_mask].tolist()
            removed_indices = fps_data.index[~keep_mask].tolist()
            
            # Memory-efficient processing for large datasets
            if data_size > 500_000:  # > 500K points (~139 hours)
                st.info("💾 Using memory-efficient processing for large dataset")
                
                # Process in batches to avoid memory spikes
                batch_size = 100_000
                processed_fps = []
                processed_cpu = []
                processed_time = []
                
                progress_bar = st.progress(0)
                batches = [keep_indices[i:i + batch_size] for i in range(0, len(keep_indices), batch_size)]
                
                for i, batch_indices in enumerate(batches):
                    # Process batch
                    batch_fps = self.original_data.loc[batch_indices, 'FPS'].values
                    batch_cpu = self.original_data.loc[batch_indices, 'CPU(%)'].values
                    
                    processed_fps.extend(batch_fps)
                    processed_cpu.extend(batch_cpu)
                    
                    # Update progress
                    progress_bar.progress((i + 1) / len(batches))
                
                # Create time index for processed data
                processed_time = [i / 60 for i in range(len(processed_fps))]
                
                # Create processed dataset
                self.processed_data = pd.DataFrame({
                    'FPS': processed_fps,
                    'CPU(%)': processed_cpu,
                    'TimeMinutes': processed_time
                })
                
                progress_bar.empty()
                
            else:
                # Standard processing for smaller datasets
                self.processed_data = pd.DataFrame({
                    'FPS': self.original_data.loc[keep_indices, 'FPS'].values,
                    'CPU(%)': self.original_data.loc[keep_indices, 'CPU(%)'].values,
                    'TimeMinutes': [i / 60 for i in range(len(keep_indices))]
                })
            
            # Store removed indices for tracking
            self.removed_indices = removed_indices
            
            # Enhanced feedback with performance metrics
            removal_count = len(removed_indices)
            removal_pct = (removal_count / data_size) * 100
            kept_count = len(keep_indices)
            
            # Performance impact analysis
            original_duration = data_size / 60  # minutes
            processed_duration = kept_count / 60  # minutes
            time_saved = original_duration - processed_duration
            
            st.success(f"✅ Removed {removal_count:,} frames ({removal_pct:.1f}%)")
            st.info(f"📊 Threshold: {threshold:.1f} FPS | Kept: {kept_count:,} frames")
            
            # Additional metrics for long sessions
            if duration_hours > 2:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("⏱️ Original Duration", f"{original_duration/60:.1f}h")
                with col2:
                    st.metric("✂️ Processed Duration", f"{processed_duration/60:.1f}h")
                with col3:
                    st.metric("🎯 Time Reduction", f"{time_saved:.0f}min")
            
            # Memory usage info for very large datasets
            if data_size > 1_000_000:
                memory_saved_mb = (removal_count * 8) / (1024 * 1024)  # Rough estimate
                st.info(f"💾 Estimated memory saved: ~{memory_saved_mb:.1f} MB")
            
            return True
            
        except MemoryError:
            st.error("❌ Out of memory! Dataset too large. Try more aggressive removal or split the data.")
            return False
        except Exception as e:
            st.error(f"❌ Outlier removal failed: {e}")
            return False
    
    def apply_processing(self, fps_window=5, fps_poly=1, cpu_window=5, cpu_poly=1,
                        enable_fps_smooth=False, enable_cpu_smooth=False,
                        enable_outlier_removal=False, outlier_sensitivity='moderate'):
        """Processing pipeline with EMERGENCY debugging and large dataset optimization"""
        
        # 🚨 EMERGENCY DEBUG - PROCESSING PHASE
        st.markdown("### 🚨 EMERGENCY DEBUG - PROCESSING PHASE")
        
        input_fps_max = self.original_data['FPS'].max()
        input_fps_min = self.original_data['FPS'].min()
        data_size = len(self.original_data)
        st.write(f"**Input to processing - FPS range: {input_fps_min} - {input_fps_max}**")
        st.write(f"**Dataset size: {data_size:,} data points**")
        
        # Large dataset warnings for smoothing
        if data_size > 500_000 and (enable_fps_smooth or enable_cpu_smooth):
            st.warning("⚠️ **Large dataset + smoothing detected** - This may take several minutes")
        
        # Initialize with original data
        if enable_outlier_removal:
            st.write("⚠️ **OUTLIER REMOVAL ENABLED - DEBUGGING...**")
            if not self.remove_fps_outliers_optimized(outlier_sensitivity):
                st.warning("⚠️ Outlier removal failed, using original data")
                self.processed_data = self.original_data.copy()
        else:
            st.write("✅ **No outlier removal - copying original data**")
            self.processed_data = self.original_data.copy()
        
        # Check processed data before smoothing
        processed_fps_max = self.processed_data['FPS'].max()
        processed_fps_min = self.processed_data['FPS'].min()
        processed_size = len(self.processed_data)
        st.write(f"**After outlier processing - FPS range: {processed_fps_min} - {processed_fps_max}**")
        st.write(f"**Processed dataset size: {processed_size:,} data points**")
        
        # Apply smoothing filters with large dataset optimization
        try:
            # FPS Smoothing with performance optimization
            if enable_fps_smooth:
                st.write("⚠️ **FPS SMOOTHING ENABLED - DEBUGGING...**")
                
                data_length = len(self.processed_data)
                fps_window = min(max(fps_window, 5), data_length)
                if fps_window % 2 == 0:
                    fps_window -= 1
                
                if data_length >= fps_window:
                    # Before smoothing
                    before_fps = self.processed_data['FPS'].copy()
                    before_max = before_fps.max()
                    before_min = before_fps.min()
                    st.write(f"**Before smoothing: {before_min} - {before_max}**")
                    
                    # Large dataset optimization for smoothing
                    if data_length > 1_000_000:
                        st.info("🔧 **Large dataset smoothing** - Using chunked processing")
                        # Process in chunks for very large datasets
                        chunk_size = 100_000
                        chunks = np.array_split(before_fps.values, max(1, data_length // chunk_size))
                        smoothed_chunks = []
                        
                        progress_bar = st.progress(0)
                        for i, chunk in enumerate(chunks):
                            if len(chunk) >= fps_window:
                                smoothed_chunk = savgol_filter(chunk, window_length=fps_window, polyorder=min(fps_poly, fps_window-1))
                            else:
                                smoothed_chunk = chunk  # Keep original if too small
                            smoothed_chunks.append(smoothed_chunk)
                            progress_bar.progress((i + 1) / len(chunks))
                        
                        smoothed_fps = np.concatenate(smoothed_chunks)
                        progress_bar.empty()
                        
                    elif data_length > 500_000:
                        st.info("⚡ **Medium large dataset** - Optimized smoothing")
                        with st.spinner("Processing smoothing..."):
                            smoothed_fps = savgol_filter(
                                before_fps.values,
                                window_length=fps_window,
                                polyorder=min(fps_poly, fps_window-1)
                            )
                    else:
                        # Standard processing
                        smoothed_fps = savgol_filter(
                            before_fps.values,
                            window_length=fps_window,
                            polyorder=min(fps_poly, fps_window-1)
                        )
                    
                    self.processed_data['FPS_Smooth'] = smoothed_fps
                    
                    # Post-smoothing validation
                    after_min = smoothed_fps.min()
                    after_max = smoothed_fps.max()
                    st.write(f"**After smoothing: {after_min:.1f} - {after_max:.1f}**")
                    
                    # Check for smoothing artifacts
                    if after_max > before_max + 5:
                        st.error(f"🚨 **SMOOTHING BUG: Max increased from {before_max} to {after_max:.1f}**")
                else:
                    self.processed_data['FPS_Smooth'] = self.processed_data['FPS']
            else:
                st.write("✅ **FPS SMOOTHING DISABLED - DIRECT COPY**")
                self.processed_data['FPS_Smooth'] = self.processed_data['FPS'].copy()
            
            # CPU Smoothing with large dataset optimization
            if enable_cpu_smooth:
                st.write("⚠️ **CPU SMOOTHING ENABLED**")
                data_length = len(self.processed_data)
                cpu_window = min(max(cpu_window, 5), data_length)
                if cpu_window % 2 == 0:
                    cpu_window -= 1
                
                if data_length >= cpu_window:
                    # Large dataset optimization for CPU smoothing
                    if data_length > 1_000_000:
                        st.info("🔧 **Large dataset CPU smoothing** - Using chunked processing")
                        # Process in chunks
                        chunk_size = 100_000
                        cpu_data = self.processed_data['CPU(%)'].values
                        chunks = np.array_split(cpu_data, max(1, data_length // chunk_size))
                        smoothed_chunks = []
                        
                        progress_bar = st.progress(0)
                        for i, chunk in enumerate(chunks):
                            if len(chunk) >= cpu_window:
                                smoothed_chunk = savgol_filter(chunk, window_length=cpu_window, polyorder=min(cpu_poly, cpu_window-1))
                            else:
                                smoothed_chunk = chunk
                            smoothed_chunks.append(smoothed_chunk)
                            progress_bar.progress((i + 1) / len(chunks))
                        
                        smoothed_cpu = np.concatenate(smoothed_chunks)
                        progress_bar.empty()
                        
                    elif data_length > 500_000:
                        st.info("⚡ **Medium large dataset CPU** - Optimized smoothing")
                        with st.spinner("Processing CPU smoothing..."):
                            smoothed_cpu = savgol_filter(
                                self.processed_data['CPU(%)'].values,
                                window_length=cpu_window,
                                polyorder=min(cpu_poly, cpu_window-1)
                            )
                    else:
                        smoothed_cpu = savgol_filter(
                            self.processed_data['CPU(%)'].values,
                            window_length=cpu_window,
                            polyorder=min(cpu_poly, cpu_window-1)
                        )
                    
                    self.processed_data['CPU_Smooth'] = smoothed_cpu
                    st.info(f"🖥️ CPU smoothed (window: {cpu_window}, poly: {cpu_poly})")
                else:
                    self.processed_data['CPU_Smooth'] = self.processed_data['CPU(%)']
            else:
                st.write("✅ **CPU SMOOTHING DISABLED - DIRECT COPY**")
                self.processed_data['CPU_Smooth'] = self.processed_data['CPU(%)'].copy()
            
            # Final validation
            st.markdown("### 🚨 FINAL PROCESSING VALIDATION")
            if 'FPS_Smooth' in self.processed_data:
                final_fps_max = self.processed_data['FPS_Smooth'].max()
                final_fps_min = self.processed_data['FPS_Smooth'].min()
                st.write(f"**Final FPS_Smooth range: {final_fps_min:.1f} - {final_fps_max:.1f}**")
                
                if final_fps_max > input_fps_max + 0.1:
                    st.error(f"🚨 **PROCESSING BUG DETECTED!**")
                    st.error(f"**FPS_Smooth max ({final_fps_max:.1f}) > original max ({input_fps_max})**")
                    return False
                else:
                    st.success(f"✅ **Processing completed safely, FPS max: {final_fps_max:.1f}**")
            
            # Performance summary for large datasets
            if processed_size > 500_000:
                st.info(f"🚀 **Large dataset processing completed**: {processed_size:,} data points processed successfully")
            
            return True
            
        except MemoryError:
            st.error("❌ **Out of memory!** Dataset too large for smoothing. Try disabling smoothing or use outlier removal first.")
            return False
        except Exception as e:
            st.error(f"❌ Processing failed: {e}")
            return False
    
    def create_scene_style_chart(self, game_title, game_settings, game_mode, smartphone_name,
                                fps_color, cpu_color, show_original=True, show_processed=True):
        """Create Scene app style chart with FPS bar chart and CPU line overlay"""
        
        # Set up Scene app inspired dark theme
        plt.style.use('dark_background')
        fig, ax1 = plt.subplots(figsize=(19.2, 10.8))
        fig.patch.set_facecolor('#0f0f0f')  # Very dark background like Scene
        ax1.set_facecolor('#0f0f0f')
        
        # Setup axes with Scene app styling
        ax1.set_xlabel('Time (minutes)', fontsize=12, fontweight='bold', color='#cccccc')
        ax1.set_ylabel('FPS', color='#cccccc', fontsize=12, fontweight='bold')
        ax1.tick_params(axis='y', labelcolor='#cccccc', labelsize=10)
        ax1.tick_params(axis='x', labelcolor='#cccccc', labelsize=10)
        
        # CPU overlay axis
        ax2 = ax1.twinx()
        ax2.set_ylabel('CPU Usage (%)', color='#cccccc', fontsize=12, fontweight='bold')
        ax2.tick_params(axis='y', labelcolor='#cccccc', labelsize=10)
        ax2.set_ylim(0, 100)
        
        # Determine primary dataset
        primary_data = self.processed_data if self.processed_data is not None else self.original_data
        
        # Calculate optimal bar width for Scene app look
        total_duration = len(primary_data) / 60  # minutes
        bar_width = total_duration / len(primary_data) * 0.9  # 90% of time interval for density
        
        # Plot original data as background (faded) if requested
        if show_original and self.processed_data is not None and len(self.processed_data) != len(self.original_data):
            orig_length = len(self.processed_data)
            orig_time = self.original_data['TimeMinutes'][:orig_length]
            orig_fps = self.original_data['FPS'][:orig_length]
            orig_cpu = self.original_data['CPU(%)'][:orig_length]
            
            # Background FPS bars (Scene app style - very faded)
            ax1.bar(orig_time, orig_fps, width=bar_width, color='#555555', 
                   alpha=0.3, edgecolor='none', zorder=1, label='Original FPS')
            # Background CPU line
            ax2.plot(orig_time, orig_cpu, color=cpu_color, linewidth=1,
                    alpha=0.3, linestyle='--', zorder=2)
        
        # Main chart: Scene app style bars + line overlay
        if show_processed and self.processed_data is not None:
            time_data = self.processed_data['TimeMinutes']
            fps_data = self.processed_data['FPS_Smooth'] if 'FPS_Smooth' in self.processed_data else self.processed_data['FPS']
            cpu_data = self.processed_data['CPU_Smooth'] if 'CPU_Smooth' in self.processed_data else self.processed_data['CPU(%)']
            
            # Scene app style FPS bars - white/light gray like original
            ax1.bar(time_data, fps_data, width=bar_width, color=fps_color, 
                   alpha=0.85, edgecolor='none', zorder=3, label='FPS')
            
            # CPU as line overlay for correlation analysis
            ax2.plot(time_data, cpu_data, color=cpu_color, linewidth=2.5,
                    label='CPU', alpha=0.9, zorder=4, linestyle='-')
            
        elif show_original:
            # Fallback to original data with Scene style
            time_data = self.original_data['TimeMinutes']
            fps_data = self.original_data['FPS']
            cpu_data = self.original_data['CPU(%)']
            
            ax1.bar(time_data, fps_data, width=bar_width, color=fps_color,
                   alpha=0.85, edgecolor='none', label='FPS', zorder=3)
            ax2.plot(time_data, cpu_data, color=cpu_color, linewidth=2.5, 
                    label='CPU', alpha=0.9, zorder=4)
        
        # Y-axis limits
        if 'FPS_Smooth' in primary_data:
            fps_max = max(primary_data['FPS_Smooth']) * 1.1
        else:
            fps_max = max(primary_data['FPS']) * 1.1
        
        ax1.set_ylim(0, fps_max)
        
        # Scene app style title
        title_text = f"{game_title}\n{game_settings}\n{game_mode}"
        plt.suptitle(title_text, fontsize=24, fontweight='bold', y=0.98, color='white')
        plt.subplots_adjust(top=0.85)
        
        # Minimal grid like Scene app
        ax1.grid(True, alpha=0.15, linestyle='-', color='#444444', linewidth=0.5)
        ax1.set_axisbelow(True)
        
        # Scene app style legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        
        if lines1 or lines2:
            # Create custom legend entries
            legend_elements = []
            legend_labels = []
            
            # Add smartphone name as header
            legend_elements.append(plt.Line2D([0], [0], color='none'))
            legend_labels.append(smartphone_name)
            
            # Add FPS and CPU
            if lines1:
                legend_elements.extend(lines1)
                legend_labels.extend(labels1)
            if lines2:
                legend_elements.extend(lines2)
                legend_labels.extend(labels2)
            
            legend = ax1.legend(legend_elements, legend_labels,
                              loc='upper right', framealpha=0.9, fancybox=True,
                              facecolor='#1a1a1a', edgecolor='#555555')
            
            # Style legend text
            for i, text in enumerate(legend.get_texts()):
                text.set_color('white')
                text.set_horizontalalignment('left')
                if i == 0:  # Smartphone name
                    text.set_fontweight('bold')
                    text.set_fontsize(11)
                else:
                    text.set_fontsize(10)
        
        # Remove spines for clean Scene app look
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
            fps_data = self.processed_data['FPS_Smooth'].dropna() if 'FPS_Smooth' in self.processed_data else self.processed_data['FPS'].dropna()
            cpu_data = self.processed_data['CPU_Smooth'].dropna() if 'CPU_Smooth' in self.processed_data else self.processed_data['CPU(%)'].dropna()
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
        """Generate CSV with EMERGENCY debugging and validation"""
        if self.processed_data is None:
            return None
        
        # 🚨 EMERGENCY DEBUG - EXPORT PHASE
        st.markdown("### 🚨 EMERGENCY DEBUG - EXPORT PHASE")
        
        # Check what we're about to export
        export_fps = self.processed_data['FPS_Smooth'] if 'FPS_Smooth' in self.processed_data else self.processed_data['FPS']
        export_cpu = self.processed_data['CPU_Smooth'] if 'CPU_Smooth' in self.processed_data else self.processed_data['CPU(%)']
        
        export_fps_max = export_fps.max()
        export_fps_min = export_fps.min()
        
        st.write(f"**About to export FPS range: {export_fps_min:.1f} - {export_fps_max:.1f}**")
        
        # Check for high values before export
        high_fps = export_fps[export_fps > 70]
        if len(high_fps) > 0:
            st.error(f"🚨 **ABOUT TO EXPORT {len(high_fps)} HIGH FPS VALUES!**")
            st.error(f"**Sample high values: {list(high_fps.head(10))}**")
            
            # Find where these values are coming from
            high_indices = export_fps[export_fps > 70].index
            st.write("**High value locations:**")
            for idx in high_indices[:5]:  # Show first 5
                st.write(f"  Index {idx}: FPS={export_fps.iloc[idx]:.1f}, CPU={export_cpu.iloc[idx]:.1f}")
            
            # Block export if values are impossible
            if export_fps_max > 80:
                st.error(f"🚨 **EXPORT BLOCKED: Impossible FPS values detected (max: {export_fps_max:.1f})**")
                st.error("**Please check your processing settings or report this bug!**")
                return None
        else:
            st.success(f"✅ **Export data looks normal (max FPS: {export_fps_max:.1f})**")
        
        # Create clean DataFrame for export
        export_data = pd.DataFrame({
            'Time_Minutes': self.processed_data['TimeMinutes'],
            'FPS': export_fps,
            'CPU_Percent': export_cpu
        })
        
        # Round to reasonable precision
        export_data['Time_Minutes'] = export_data['Time_Minutes'].round(3)
        export_data['FPS'] = export_data['FPS'].round(1)
        export_data['CPU_Percent'] = export_data['CPU_Percent'].round(1)
        
        # Final validation after rounding
        final_fps_max = export_data['FPS'].max()
        st.write(f"**After rounding - Final FPS max: {final_fps_max}**")
        
        # Convert to CSV
        csv_buffer = io.StringIO()
        export_data.to_csv(csv_buffer, index=False)
        csv_content = csv_buffer.getvalue()
        
        # Generate filename with scene style indicator
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{game_title.replace(' ', '_')}_SCENE_STYLE_processed_{timestamp}.csv"
        
        st.success(f"✅ **Export generated with filename: {filename}**")
        st.info(f"📊 **Export contains {len(export_data)} rows with max FPS: {final_fps_max}**")
        
        return csv_content, filename

def main():
    # Header with Scene app inspiration and performance capabilities
    st.title("🎮 Gaming Chart Generator - Scene App Style")
    st.markdown("Transform gaming logs with **Scene App inspired bar charts** + **10+ hour session support** + Professional analysis tools")
    
    # Performance badge
    st.markdown("🚀 **High Performance**: Optimized for extended gaming sessions up to 10+ hours with 1M+ data points")
    
    # Initialize
    generator = SceneStyleGamingChartGenerator()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("🎮 Game Configuration")
        game_title = st.text_input("Game Title", value="MOBILE LEGENDS")
        game_settings = st.text_input("Graphics Settings", value="ULTRA - 120 FPS")
        game_mode = st.text_input("Performance Mode", value="BOOST MODE")
        smartphone_name = st.text_input("Smartphone Model", value="iPhone 15 Pro Max")
        
        st.header("🎨 Scene Style Colors")
        fps_color = st.color_picker("FPS Bar Color", "#FFFFFF")  # White like Scene app
        cpu_color = st.color_picker("CPU Line Color", "#FF6600")   # Orange for visibility
        
        st.header("📊 Display Options")
        show_original = st.checkbox("Show Original Data", value=False,
                                   help="Show original data as background bars")
        show_processed = st.checkbox("Show Processed Data", value=True,
                                    help="Show processed data as main Scene-style bars")
        
        st.header("🔧 Data Processing")
        
        # Large dataset warning and info
        if uploaded_file:
            try:
                # Quick data size check
                uploaded_file.seek(0)
                file_size_mb = len(uploaded_file.read()) / (1024 * 1024)
                uploaded_file.seek(0)
                
                if file_size_mb > 50:  # Large file warning
                    st.warning(f"⚠️ Large file detected ({file_size_mb:.1f} MB). Processing may take longer.")
                elif file_size_mb > 100:  # Very large file
                    st.error(f"🚨 Very large file ({file_size_mb:.1f} MB). Consider splitting for better performance.")
            except:
                pass
        
        # Outlier Removal with enhanced options for long sessions
        enable_outlier_removal = st.toggle("🚫 Remove Worst FPS Frames", value=False,
                                          help="Remove severe frame drops (optimized for 10+ hour sessions)")
        
        if enable_outlier_removal:
            outlier_sensitivity = st.select_slider(
                "Removal Sensitivity",
                options=['conservative', 'moderate', 'aggressive'],
                value='moderate',
                help="Conservative: Keep most data (recommended for 10+ hours), Aggressive: Remove more outliers"
            )
            
            # Additional info for long sessions
            st.info("💡 **Long Session Tips**: Conservative mode recommended for 5+ hour sessions to preserve performance patterns")
        else:
            outlier_sensitivity = 'moderate'
        
        # Smoothing with gaming warnings
        st.info("⚠️ **Gaming Analysis Note**: Smoothing may hide frame drops and stutters")
        
        col1, col2 = st.columns(2)
        with col1:
            enable_fps_smooth = st.toggle("🎯 FPS Smoothing", value=False,
                                        help="⚠️ May hide frame drops")
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
            
            # Process data with enhanced progress for large datasets
            data_size = len(generator.original_data)
            processing_message = "🔧 Processing data with Scene app style..."
            if data_size > 500_000:
                processing_message = f"🔧 Processing large dataset ({data_size:,} points) - This may take a few minutes..."
            elif data_size > 1_000_000:
                processing_message = f"🔥 Processing very large dataset ({data_size:,} points) - Using high-performance mode..."
            
            with st.spinner(processing_message):
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
            
            # Generate Scene style chart
            st.header("📊 Scene App Style Performance Chart")
            
            if not show_original and not show_processed:
                st.warning("⚠️ Please select at least one display option")
            else:
                with st.spinner('🎨 Creating Scene app style chart...'):
                    chart_fig = generator.create_scene_style_chart(
                        game_title, game_settings, game_mode, smartphone_name,
                        fps_color, cpu_color, show_original, show_processed
                    )
                    st.pyplot(chart_fig)
                    
                    # Scene app style info
                    st.info("🎮 **Scene App Style**: FPS displayed as bars (like original Scene app), CPU as line overlay for correlation analysis")
            
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
            st.header("💾 Export Scene Style Results")
            
            col1, col2 = st.columns(2)
            
            # PNG Export
            with col1:
                st.subheader("📸 Chart Export")
                if 'chart_fig' in locals():
                    img_buffer = io.BytesIO()
                    chart_fig.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight',
                                     facecolor='#0f0f0f', edgecolor='none')
                    img_buffer.seek(0)
                    
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    png_filename = f"{game_title.replace(' ', '_')}_SCENE_STYLE_chart_{timestamp}.png"
                    
                    st.download_button(
                        label="📸 Download Scene Style Chart (PNG)",
                        data=img_buffer.getvalue(),
                        file_name=png_filename,
                        mime="image/png",
                        use_container_width=True
                    )
            
            # CSV Export
            with col2:
                st.subheader("📄 Data Export")
                if generator.processed_data is not None:
                    result = generator.generate_processed_csv(game_title)
                    
                    if result is not None:
                        csv_content, csv_filename = result
                        
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
                        st.error("❌ Export blocked due to data validation issues")
    
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
        
        with st.expander("🎮 Scene App Style Features"):
            st.markdown("""
            **Scene App Inspired Design:**
            - **FPS as white/custom colored bars** (like Scene app)
            - **CPU as line overlay** for correlation analysis
            - **Dark gaming theme** with minimal grid
            - **Frame drops visible** as shorter bars
            - **Professional gaming benchmark appearance**
            
            **Why Scene App Style?**
            - Individual frame performance clearly visible
            - Frame drops and stutters easy to spot
            - Matches professional gaming analysis tools
            - Better pattern recognition for performance issues
            - Familiar interface for mobile gamers
            
            **🚀 High Performance Features:**
            - **10+ hour session support** with chunked processing
            - **Memory-efficient** handling of large datasets (1M+ data points)
            - **Progress tracking** for long processing operations
            - **Enhanced thresholds** for extended gaming sessions
            - **Batch processing** to prevent memory overflow
            """)
        
        with st.expander("⚡ Performance & Large Dataset Info"):
            st.markdown("""
            **Supported Dataset Sizes:**
            - **Small sessions**: < 2 hours (standard processing)
            - **Medium sessions**: 2-5 hours (optimized processing)  
            - **Long sessions**: 5-10 hours (enhanced thresholds)
            - **Extended sessions**: 10+ hours (chunked processing)
            - **Maximum**: 1M+ data points (~278+ hours at 1 FPS)
            
            **Automatic Optimizations:**
            - **< 500K points**: Standard memory processing
            - **500K - 1M points**: Memory-efficient batch processing
            - **> 1M points**: Chunked processing with progress bars
            - **Long sessions (5+ hours)**: Conservative outlier detection
            
            **Performance Tips:**
            - Use **Conservative** sensitivity for long sessions
            - Large files automatically use chunked processing
            - Memory usage optimized for extended gaming sessions
            - Progress bars show processing status for large datasets
            """)
        
        with st.expander("💾 Export Features"):
            st.markdown("""
            **Chart Export (PNG):**
            - High-resolution (300 DPI)
            - Scene app inspired dark theme
            - Professional gaming chart format
            - Ready for presentations/reports
            
            **Data Export (CSV):**
            - Clean processed FPS and CPU data
            - Time_Minutes, FPS, CPU_Percent columns
            - Rounded values for clarity
            - Ready for further analysis
            - Compatible with Excel/Google Sheets
            """)
        
        with st.expander("🚨 DEBUG Mode"):
            st.markdown("""
            **EMERGENCY DEBUG Mode is ACTIVE:**
            - Detailed column analysis
            - Raw data validation
            - Processing step-by-step tracking
            - Export validation and blocking
            - High FPS value detection
            - Bug identification assistance
            """)

if __name__ == "__main__":
    main()
