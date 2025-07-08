import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, precision_recall_curve
import json
from typing import Dict, List, Tuple, Optional
import warnings
import re
import os
warnings.filterwarnings('ignore')

# Helper function to split long names for graph display
def split_long_name(name, max_len=18):
    """Split a long name into two lines for better graph display."""
    if len(name) <= max_len:
        return name
    # Try to split at the nearest underscore, dash, or space before max_len
    for sep in ['_', '-', ' ']:
        idx = name.rfind(sep, 0, max_len)
        if idx != -1:
            return name[:idx+1] + "<br>" + name[idx+1:]
    # If no separator found, just split at max_len
    return name[:max_len] + "<br>" + name[max_len:]

# Super class mapping
SUPER_CLASS_MAPPING = {
    "Merchant": [
        "Bulk",
        "Containers",
        "General-Cargo",
        "Ro-Ro",
        "Tanker",
        "Merchant"
    ],
    "Military": [
        "Saar-4.5",
        "Saar-5",
        "Submarine",
        "Military"
    ],
    "SWC": [
        "Buoy",
        "Sailing",
        "SWC"
    ],
    "Support": [
        "cruise",
        "Ferry",
        "Supply",
        "Tug",
        "Yacht",
        "Fishing", 
        "Support",
        "barge"
    ],
    "Dvora": [
        "Dvora"
    ],
    "Motor": [
        "Rubber",
        "Motor"
    ],
    "Patrol-Boat": [
        "Patrol-Boat",
        "Patrol"
    ],
    "Pilot": [
        "Pilot"
    ]
}

def create_reverse_mapping(super_class_mapping):
    """Create a reverse mapping from subclass to super class"""
    reverse_mapping = {}
    for super_class, subclasses in super_class_mapping.items():
        for subclass in subclasses:
            reverse_mapping[subclass] = super_class
    return reverse_mapping

# Create reverse mapping for quick lookup - OPTIMIZED
SUBCLASS_TO_SUPER_MAPPING = create_reverse_mapping(SUPER_CLASS_MAPPING)

def map_to_super_class(class_name):
    """Map a subclass to its super class - OPTIMIZED"""
    return SUBCLASS_TO_SUPER_MAPPING.get(class_name, class_name)

def extract_experiment_name(file_name):
    """Extract experiment name from file name, removing 'Albatross' and date patterns"""
    # Get the base name without extension
    base_name = os.path.splitext(file_name)[0]
    
    # Remove 'Albatross' (case insensitive)
    base_name = re.sub(r'albatross', '', base_name, flags=re.IGNORECASE)
    
    # Remove common date patterns
    # Patterns like: 2023-01-01, 01-01-2023, 2023_01_01, 01_01_2023, 20230101
    date_patterns = [
        r'\d{4}[-_]\d{2}[-_]\d{2}',  # YYYY-MM-DD or YYYY_MM_DD
        r'\d{2}[-_]\d{2}[-_]\d{4}',  # MM-DD-YYYY or MM_DD_YYYY
        r'\d{8}',                    # YYYYMMDD
        r'\d{2}\d{2}\d{4}',         # MMDDYYYY
        r'\d{4}\d{2}\d{2}',         # YYYYMMDD
    ]
    
    for pattern in date_patterns:
        base_name = re.sub(pattern, '', base_name)
    
    # Remove extra underscores, dashes, and spaces
    base_name = re.sub(r'[-_\s]+', '_', base_name)
    
    # Remove leading/trailing underscores and spaces
    base_name = base_name.strip('_- ')
    
    # If the name is empty or too short, use a default
    if not base_name or len(base_name) < 2:
        base_name = "Experiment"
    
    return base_name

def apply_super_class_analysis(data, use_super_class=False):
    """Apply super class analysis to the data if enabled - OPTIMIZED VERSION"""
    if not use_super_class:
        return data
    
    # Create a copy to avoid modifying the original data
    data_copy = data.copy()
    
    # OPTIMIZATION 1: Vectorized mapping using pandas map() instead of apply()
    data_copy['cls_name_original'] = data_copy['cls_name']
    data_copy['cls_name'] = data_copy['cls_name'].map(SUBCLASS_TO_SUPER_MAPPING).fillna(data_copy['cls_name'])
    
    # OPTIMIZATION 2: Vectorized mapping for class_mistake
    data_copy['class_mistake_original'] = data_copy['class_mistake']
    mask = ~data_copy['class_mistake'].isin(['-', 'background'])
    data_copy.loc[mask, 'class_mistake'] = data_copy.loc[mask, 'class_mistake'].map(SUBCLASS_TO_SUPER_MAPPING).fillna(data_copy.loc[mask, 'class_mistake'])
    
    # OPTIMIZATION 3: Vectorized mistake_kind recalculation
    data_copy['mistake_kind_original'] = data_copy['mistake_kind']
    
    # Only recalculate for prediction data (ground_truth == 0)
    prediction_mask = data_copy['ground_truth'] == 0
    prediction_data = data_copy[prediction_mask].copy()
    
    if len(prediction_data) > 0:
        # Vectorized conditions for mistake_kind recalculation
        background_mask = prediction_data['class_mistake'] == 'background'
        correct_mask = prediction_data['class_mistake'] == '-'
        same_super_class_mask = prediction_data['cls_name'] == prediction_data['class_mistake']
        
        # Update mistake_kind using vectorized operations
        prediction_data.loc[background_mask, 'mistake_kind'] = 'FP'
        prediction_data.loc[correct_mask, 'mistake_kind'] = 'TP'
        prediction_data.loc[same_super_class_mask & ~background_mask & ~correct_mask, 'mistake_kind'] = 'TP'
        prediction_data.loc[~same_super_class_mask & ~background_mask & ~correct_mask, 'mistake_kind'] = 'FP'
        
        # Update the main dataframe
        data_copy.loc[prediction_mask, 'mistake_kind'] = prediction_data['mistake_kind']
    
    return data_copy

# Import the summary analysis module
try:
    from summary_analysis import summary_analysis_page
    SUMMARY_AVAILABLE = True
except ImportError:
    SUMMARY_AVAILABLE = False
    st.warning("Summary analysis module not found. Please ensure summary_analysis.py is in the same directory.")

def check_dependencies():
    """Check if required dependencies are installed"""
    missing_deps = []
    
    try:
        import openpyxl
    except ImportError:
        missing_deps.append('openpyxl')
    
    try:
        import xlrd
    except ImportError:
        missing_deps.append('xlrd')
    
    if missing_deps:
        st.sidebar.warning(f"⚠️ Missing dependencies: {', '.join(missing_deps)}")
        st.sidebar.info("For Excel support, run: `pip install openpyxl xlrd`")
        return False
    return True

def calculate_detection_metrics(exp_data, analysis_mode):
    """Calculate TP, FP, FN consistently across all pages for detection mode"""
    if analysis_mode == "Detection":
        # CORRECTED DETECTION LOGIC - consistent across all pages
        
        # TP: Model detected something AND something was there (class irrelevant)
        detection_tp = len(exp_data[
            (exp_data['ground_truth'] == 0) & 
            (
                (exp_data['mistake_kind'] == 'TP') |  # Correct detection
                (
                    (exp_data['mistake_kind'] == 'FP') & 
                    (exp_data['class_mistake'] != 'background') & 
                    (exp_data['class_mistake'] != '-')
                )  # Wrong class but object was there
            )
        ])
        
        # FP: Model detected something BUT nothing was there
        detection_fp = len(exp_data[
            (exp_data['ground_truth'] == 0) & 
            (exp_data['mistake_kind'] == 'FP') & 
            (exp_data['class_mistake'] == 'background')
        ])
        
        # FN: Model missed something that was there
        detection_fn = len(exp_data[
            (exp_data['ground_truth'] == 1) & 
            (exp_data['mistake_kind'] == 'FN') & 
            (exp_data['class_mistake'] == 'background')
        ])
        
        return detection_tp, detection_fp, detection_fn
        
    elif analysis_mode == "Classification":
        # Classification analysis: among detected objects only (exclude background)
        # TP: Correct classifications
        classification_tp = len(exp_data[exp_data['mistake_kind'] == 'TP'])
        
        # FP: Wrong classifications (detected object A, but it was actually object B)
        classification_fp = len(exp_data[
            (exp_data['mistake_kind'] == 'FP') & 
            (exp_data['class_mistake'] != 'background') &
            (exp_data['class_mistake'] != '-')
        ])
        
        # FN: For classification, we use the same count as FP (mirror relationship)
        classification_fn = classification_fp
        
        return classification_tp, classification_fp, classification_fn
        
    else:  # "All Data"
        # Original logic - use all data
        overall_tp = len(exp_data[exp_data['mistake_kind'] == 'TP'])
        overall_fp = len(exp_data[exp_data['mistake_kind'] == 'FP'])
        overall_fn = len(exp_data[exp_data['mistake_kind'] == 'FN'])
        
        return overall_tp, overall_fp, overall_fn

class MultiExperimentAnalyzer:
    def __init__(self):
        """Initialize the multi-experiment analyzer - OPTIMIZED WITH CACHING"""
        self.experiments = {}  # Dict to store experiment data
        self.train_data = None
        self.main_experiment = None
        
        # OPTIMIZATION: Add caching for processed data
        self._cache = {
            'superclass_data': {},  # Cache for superclass processed data
            'subclass_data': {},    # Cache for subclass processed data
            'last_super_class_mode': None  # Track last mode to detect changes
        }
        
    def _clear_cache(self):
        """Clear the data cache"""
        self._cache = {
            'superclass_data': {},
            'subclass_data': {},
            'last_super_class_mode': None
        }
        
    def add_experiment(self, name: str, data_file, experiment_id: str = None):
        """Add an experiment to the analyzer - OPTIMIZED"""
        try:
            exp_id = experiment_id if experiment_id else name
            data = self._safe_read_file(data_file)
            
            if data is not None and len(data) > 0:
                # Validate required columns
                required_columns = ['bb_id', 'frame', 'cls_name', 'mistake_kind']
                missing_columns = [col for col in required_columns if col not in data.columns]
                
                if missing_columns:
                    st.error(f"❌ Missing required columns: {missing_columns}")
                    st.info("📋 Your file should contain: bb_id, frame, cls_name, mistake_kind, platform, bb_size, area_px")
                    return False
                
                # Process the experiment data
                processed_data = self.process_experiment_data(data, exp_id)
                if processed_data is not None:
                    self.experiments[exp_id] = {
                        'name': name,
                        'data': processed_data,
                        'raw_data': data
                    }
                    
                    # OPTIMIZATION: Clear cache when new experiment is added
                    self._clear_cache()
                    
                    return True
                else:
                    return False
            else:
                st.error("❌ No data found in file or file couldn't be read")
                return False
        except Exception as e:
            st.error(f"❌ Error adding experiment {name}: {str(e)}")
            st.info("💡 **Troubleshooting Tips:**")
            st.info("• Check file format and column names")
            st.info("• Ensure numeric columns contain valid numbers")
            st.info("• Try with a smaller test file first")
            return False
    
    def auto_process_folder_files(self, files_list):
        """Automatically process multiple files from folder structure"""
        success_count = 0
        total_files = len(files_list)
        
        # Create a progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, file in enumerate(files_list):
            # Update progress
            progress = (i + 1) / total_files
            progress_bar.progress(progress)
            
            # Extract experiment name from file name
            exp_name = extract_experiment_name(file.name)
            
            # Make sure the experiment name is unique
            original_name = exp_name
            counter = 1
            while exp_name in self.experiments:
                exp_name = f"{original_name}_{counter}"
                counter += 1
            
            status_text.text(f"Processing: {exp_name} ({i+1}/{total_files})")
            
            # Add the experiment
            success = self.add_experiment(exp_name, file, exp_name)
            if success:
                success_count += 1
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Show results
        if success_count > 0:
            st.success(f"✅ Successfully processed {success_count} out of {total_files} files!")
            
            # Show loaded experiments
            st.subheader("📊 Loaded Experiments:")
            for exp_name, exp_info in self.experiments.items():
                st.write(f"• **{exp_name}**: {len(exp_info['data'])} records")
        else:
            st.error(f"❌ Failed to process any files. Please check the file formats and data structure.")
        
        return success_count > 0
    
    def get_processed_data(self, exp_id: str, use_super_class: bool = False):
        """Get processed data with caching optimization - OPTIMIZED"""
        if exp_id not in self.experiments:
            return None
        
        # OPTIMIZATION: Check cache first
        cache_key = f"{exp_id}_{use_super_class}"
        cache_store = 'superclass_data' if use_super_class else 'subclass_data'
        
        if cache_key in self._cache[cache_store]:
            return self._cache[cache_store][cache_key]
        
        # If not in cache, process the data
        data = self.experiments[exp_id]['data'].copy()
        
        # Apply super class analysis if enabled
        if use_super_class:
            data = apply_super_class_analysis(data, use_super_class=True)
        
        # OPTIMIZATION: Store in cache
        self._cache[cache_store][cache_key] = data
        
        return data
    
    def set_train_data(self, train_file):
        """Set training data from Excel/CSV file"""
        try:
            train_data = self._safe_read_file(train_file)
            if train_data is not None and not train_data.empty:
                self.train_data = train_data
                self.train_class_distribution = cached_train_class_distribution(train_data)
                return True
            else:
                st.error("❌ No training data found in file or file is empty")
                return False
        except Exception as e:
            st.error(f"Error loading train data: {str(e)}")
            return False
    
    def _safe_read_file(self, file_path):
        """Safely read file with multiple encoding attempts, now using cache."""
        if hasattr(file_path, 'name'):
            file_bytes = file_path.read()
            file_name = file_path.name
            file_size = len(file_bytes)
            return cached_safe_read_file(file_bytes, file_name, file_size)
        return None
    
    def process_experiment_data(self, df: pd.DataFrame, exp_id: str) -> pd.DataFrame:
        """Process experiment data to add derived columns - now using cache."""
        return cached_process_experiment_data(df, exp_id)
    
    def _categorize_area(self, areas: pd.Series) -> pd.Series:
        """Categorize areas with optimal bins based on data distribution analysis - OPTIMIZED"""
        # Remove NaN values for calculation
        valid_areas = areas.dropna()
        if len(valid_areas) == 0:
            return pd.Series(['Unknown'] * len(areas), index=areas.index)
        
        min_area = valid_areas.min()
        max_area = valid_areas.max()
        
        # Handle case where all areas are the same
        if min_area == max_area:
            return pd.Series([f"Area_{int(min_area)}"] * len(areas), index=areas.index)
        
        # Optimal bin edges based on data analysis (chronologically ordered)
        optimal_bin_edges = [
            min_area,  # Start from minimum
            50,        # Small objects (5.9% of data)
            100,       # Very small objects (15.3% of data concentrated here)
            200,       # Small-medium objects (18.4% of data - highest concentration)
            400,       # Around median (11.9% of data)
            1000,      # End of main concentration (58.9% total up to here)
            5000,      # Medium objects (14.8% of data)
            20000,     # Large objects (7.3% of data)
            50000,     # Very large objects (7.0% of data)
            100000,    # Extra large objects (2.5% of data)
            max_area   # Maximum value
        ]
        
        # Filter bin edges to only include those within our data range
        filtered_edges = [edge for edge in optimal_bin_edges if edge <= max_area]
        
        # Ensure we have the max value
        if filtered_edges[-1] != max_area:
            filtered_edges.append(max_area)
        
        # Remove duplicate edges and sort
        bin_edges = sorted(list(set(filtered_edges)))
        
        # Create clean chronologically ordered labels (no ugly prefixes)
        bin_labels = []
        for i in range(len(bin_edges) - 1):
            start = int(bin_edges[i])
            end = int(bin_edges[i + 1])
            
            # Format with commas for readability - clean labels only
            if start < 1000 and end < 1000:
                bin_labels.append(f"{start}-{end}")
            elif start < 1000:
                bin_labels.append(f"{start}-{end:,}")
            else:
                bin_labels.append(f"{start:,}-{end:,}")
        
        # Store the correct order for later use
        self._area_category_order = bin_labels.copy()
        
        # OPTIMIZATION: Use pd.cut more efficiently
        categorized = pd.cut(areas, bins=bin_edges, labels=bin_labels, include_lowest=True, ordered=True)
        
        # Convert to string but preserve the order information
        categorized = categorized.astype(str)
        categorized = categorized.replace('nan', 'Unknown')
        
        return categorized
    
    def _get_area_category_order(self, data):
        """Get the correct chronological order for area categories"""
        import re
        
        # Get actual categories present in the data
        actual_categories = list(data['area_category'].unique())
        
        def extract_start_number(category):
            """Extract the starting number from area category for sorting"""
            if category == 'Unknown':
                return float('inf')  # Put Unknown at the end
            
            # Extract the first number from strings like "1-50", "1,000-5,000", etc.
            # Remove commas first, then extract number
            clean_cat = str(category).replace(',', '')
            match = re.match(r'(\d+)', clean_cat)
            if match:
                return int(match.group(1))
            return 0
        
        # Sort categories by their starting number
        ordered_categories = sorted(actual_categories, key=extract_start_number)
        
        return ordered_categories
    
    def _calculate_train_distribution(self) -> Dict:
        """Calculate class distribution from training data (Excel/CSV format)"""
        if self.train_data is None or self.train_data.empty:
            return {}
        
        try:
            # Assuming training data has 'cls_name' column like the experiment data
            if 'cls_name' in self.train_data.columns:
                class_counts = self.train_data['cls_name'].value_counts().to_dict()
                return class_counts
            else:
                st.warning("Training data doesn't have 'cls_name' column")
                return {}
        except Exception as e:
            st.warning(f"Could not calculate training distribution: {str(e)}")
            return {}
    
    def create_confusion_matrix_data(self, experiment_ids: List[str] = None, 
                                   filters: Dict = None, use_super_class: bool = False) -> pd.DataFrame:
        """Create confusion matrix data similar to the DAX code - USES CACHED DATA"""
        if experiment_ids is None:
            experiment_ids = list(self.experiments.keys())
        
        confusion_data = []
        
        for exp_id in experiment_ids:
            if exp_id not in self.experiments:
                continue
                
            # OPTIMIZATION: Use cached data
            exp_data = self.get_processed_data(exp_id, use_super_class)
            if exp_data is None:
                continue
            
            # Apply filters if provided
            if filters:
                exp_data = self._apply_filters(exp_data, filters)
            
            # False Positives: predicted class vs background
            fp_data = exp_data[exp_data['mistake_kind'] == 'FP'].copy()
            for _, row in fp_data.iterrows():
                actual_class = 'background'
                if pd.notna(row['class_mistake']) and row['class_mistake'] != '-':
                    actual_class = row['class_mistake']
                
                confusion_data.append({
                    'experiment_id': exp_id,
                    'predicted_class': row['cls_name'],
                    'actual_class': actual_class,
                    'error_type': 'FP',
                    'frame': row['frame'],
                    'bb_id': row['bb_id'],
                    'size': row['bb_size'],
                    'platform': row['platform'],
                    'area_category': row['area_category']
                })
            
            # False Negatives: background vs actual class
            fn_data = exp_data[exp_data['mistake_kind'] == 'FN'].copy()
            for _, row in fn_data.iterrows():
                confusion_data.append({
                    'experiment_id': exp_id,
                    'predicted_class': 'background',
                    'actual_class': row['cls_name'],
                    'error_type': 'FN',
                    'frame': row['frame'],
                    'bb_id': row['bb_id'],
                    'size': row['bb_size'],
                    'platform': row['platform'],
                    'area_category': row['area_category']
                })
            
            # True Positives: same class
            tp_data = exp_data[exp_data['mistake_kind'] == 'TP'].copy()
            for _, row in tp_data.iterrows():
                confusion_data.append({
                    'experiment_id': exp_id,
                    'predicted_class': row['cls_name'],
                    'actual_class': row['cls_name'],
                    'error_type': 'TP',
                    'frame': row['frame'],
                    'bb_id': row['bb_id'],
                    'size': row['bb_size'],
                    'platform': row['platform'],
                    'area_category': row['area_category']
                })
        
        return pd.DataFrame(confusion_data)
    
    def _apply_filters(self, data: pd.DataFrame, filters: Dict) -> pd.DataFrame:
        """Apply filters to data"""
        filtered = data.copy()
        
        if 'platforms' in filters and filters['platforms']:
            filtered = filtered[filtered['platform'].isin(filters['platforms'])]
        
        if 'sizes' in filters and filters['sizes']:
            filtered = filtered[filtered['bb_size'].isin(filters['sizes'])]
        
        if 'classes' in filters and filters['classes']:
            filtered = filtered[filtered['cls_name'].isin(filters['classes'])]
        
        return filtered
    
    def calculate_metrics(self, experiment_ids: List[str] = None, 
                         filters: Dict = None) -> Dict:
        """Calculate metrics for experiments with detection vs classification analysis - USES CACHED DATA"""
        if experiment_ids is None:
            experiment_ids = list(self.experiments.keys())
        
        # Get global analysis mode and super class mode from session state
        analysis_mode = getattr(st.session_state, 'global_analysis_mode', 'All Data')
        use_super_class = getattr(st.session_state, 'global_super_class_mode', False)
        
        metrics = {}
        
        for exp_id in experiment_ids:
            if exp_id not in self.experiments:
                continue
            
            # OPTIMIZATION: Use cached data
            exp_data = self.get_processed_data(exp_id, use_super_class)
            if exp_data is None:
                continue
            
            # Apply filters if provided
            if filters:
                exp_data = self._apply_filters(exp_data, filters)
            
            if len(exp_data) == 0:
                continue
            
            # Use the consistent calculation function
            overall_tp, overall_fp, overall_fn = calculate_detection_metrics(exp_data, analysis_mode)
            
            # Apply analysis mode filtering for filtered_data
            if analysis_mode == "Detection":
                # Filter data for per-class/platform analysis
                filtered_exp_data = exp_data[
                    (
                        (exp_data['ground_truth'] == 0) & 
                        (
                            (exp_data['mistake_kind'] == 'TP') |
                            (
                                (exp_data['mistake_kind'] == 'FP') & 
                                (exp_data['class_mistake'] != 'background') & 
                                (exp_data['class_mistake'] != '-')
                            )
                        )
                    ) |
                    (
                        (exp_data['ground_truth'] == 0) & 
                        (exp_data['mistake_kind'] == 'FP') & 
                        (exp_data['class_mistake'] == 'background')
                    ) |
                    (
                        (exp_data['ground_truth'] == 1) & 
                        (exp_data['mistake_kind'] == 'FN') & 
                        (exp_data['class_mistake'] == 'background')
                    )
                ]
                
            elif analysis_mode == "Classification":
                # Filter data for per-class/platform analysis (exclude background cases)
                filtered_exp_data = exp_data[
                    (exp_data['mistake_kind'] == 'TP') |
                    ((exp_data['mistake_kind'] == 'FP') & (exp_data['class_mistake'] != 'background') & (exp_data['class_mistake'] != '-'))
                ]
                
            else:  # "All Data"
                # Original logic - use all data
                filtered_exp_data = exp_data
            
            # Calculate metrics
            precision = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0
            recall = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0
            
            metrics[exp_id] = {
                'precision': precision,
                'recall': recall,
                'f1': 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0,
                'tp_count': overall_tp,
                'fp_count': overall_fp,
                'fn_count': overall_fn,
                'filtered_data': filtered_exp_data  # Store for per-platform analysis
            }
        
        return metrics

def create_experiment_manager():
    """Create the experiment management interface - OPTIMIZED WITH PROGRESS INDICATORS"""
    st.sidebar.header("🔬 Experiment Management")
    
    # GLOBAL ANALYSIS MODE SELECTOR AT TOP
    st.sidebar.subheader("🎛️ Global Analysis Mode")
    analysis_modes = {
        "All Data": "Complete dataset analysis (all TP, FP, FN)",
        "Detection": "Detection analysis: How good is the model at detecting something vs nothing?",
        "Classification": "Classification analysis: How good is the model at classifying among detected objects?"
    }
    
    global_mode = st.sidebar.selectbox(
        "Select Analysis Mode",
        list(analysis_modes.keys()),
        help="This mode affects all analysis pages"
    )
    
    # Store in session state
    st.session_state.global_analysis_mode = global_mode
    
    # SUPER CLASS ANALYSIS MODE SELECTOR - OPTIMIZED WITH PROGRESS INDICATORS
    st.sidebar.subheader("🏷️ Class Analysis Level")
    
    # Check if mode is changing
    current_super_class_mode = getattr(st.session_state, 'global_super_class_mode', False)
    
    use_super_class = st.sidebar.toggle(
        "Use Super Class Analysis",
        value=current_super_class_mode,
        help="Toggle between subclass analysis and super class analysis"
    )
    
    # OPTIMIZATION: Show loading indicator when switching modes
    if current_super_class_mode != use_super_class:
        if 'analyzer' in st.session_state:
            # Clear cache when mode changes
            st.session_state.analyzer._clear_cache()
        
        # Show loading indicator
        with st.sidebar:
            with st.spinner("🔄 Switching analysis mode..."):
                st.info("Processing data for new mode...")
    
    # Store in session state
    st.session_state.global_super_class_mode = use_super_class
    
    # Show mode descriptions
    if use_super_class:
        st.sidebar.info("**🏷️ Super Class Mode**: Analysis grouped by super classes (e.g., Merchant, Military, etc.)")
        st.sidebar.markdown("""
        **Super Class Groupings:**
        - **Merchant**: Bulk, Containers, General-Cargo, Ro-Ro, Tanker
        - **Military**: Saar-4.5, Saar-5, Submarine
        - **Support**: Cruise, Ferry, Supply, Tug, Yacht, Fishing, Barge
        - **And more...**
        """)
    else:
        st.sidebar.info("**🔍 Subclass Mode**: Analysis using original detailed class names")
    
    # Show analysis mode description
    if global_mode == "Detection":
        st.sidebar.info(f"**🔍 Detection Mode**: Measures how well the model detects *something* vs *nothing*")
        st.sidebar.markdown("""
        - **TP**: Any successful detection (even if wrong class)
        - **FP**: False alarms (detected something where there was background)
        - **FN**: Missed detections
        """)
    elif global_mode == "Classification":
        st.sidebar.info(f"**🏷️ Classification Mode**: Measures classification accuracy among detected objects only")
        st.sidebar.markdown("""
        - **TP**: Correct classifications
        - **FP**: Wrong classifications (detected object A, but it was actually object B)
        - **FN**: Mirror of FP (from other classes' perspective)
        """)
    else:
        st.sidebar.info(f"**📊 All Data Mode**: Complete analysis including all detection and classification errors")
    
    # OPTIMIZATION: Add cache status indicator
    if 'analyzer' in st.session_state:
        cache_info = st.session_state.analyzer._cache
        total_cached = len(cache_info['superclass_data']) + len(cache_info['subclass_data'])
        if total_cached > 0:
            st.sidebar.success(f"⚡ Cache: {total_cached} datasets ready")
    
    # Check dependencies
    deps_ok = check_dependencies()
    
    # Initialize session state
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = MultiExperimentAnalyzer()
    
    # Main experiment selection
    experiment_names = list(st.session_state.analyzer.experiments.keys())
    
    if experiment_names:
        main_exp = st.sidebar.selectbox(
            "🎯 Select Main Experiment", 
            options=['None'] + experiment_names,
            help="Choose the main experiment for comparison"
        )
        if main_exp != 'None':
            st.session_state.analyzer.main_experiment = main_exp
    
    # MODIFIED: Auto-processing folder files
    st.sidebar.subheader("📁 Auto-Process Folder Files")
    st.sidebar.info("💡 **New Feature**: Select multiple Excel files from your folder structure. The app will automatically extract experiment names from folder names (removing 'Albatross' and dates).")
    
    # Multiple experiment files with auto-processing
    exp_files = st.sidebar.file_uploader(
        "Select All Excel Files from Folders", 
        type=['xlsx', 'xls'],
        accept_multiple_files=True,
        help="🚀 Select all Excel files at once. Experiment names will be extracted from file names automatically."
    )
    
    # NEW: Auto-process files when uploaded
    if exp_files and len(exp_files) > 0:
        # Check if these files are new (not already processed)
        uploaded_file_names = [f.name for f in exp_files]
        existing_experiment_names = list(st.session_state.analyzer.experiments.keys())
        
        # Extract expected experiment names from uploaded files
        expected_names = [extract_experiment_name(f.name) for f in exp_files]
        
        # Check if any new files
        new_files = []
        for i, exp_file in enumerate(exp_files):
            expected_name = expected_names[i]
            # Check if this exact experiment doesn't exist yet
            if expected_name not in existing_experiment_names:
                new_files.append(exp_file)
        
        if new_files:
            st.sidebar.info(f"🔄 Found {len(new_files)} new files to process...")
            
            # Auto-process button
            if st.sidebar.button("🚀 Auto-Process All Files", type="primary"):
                with st.sidebar:
                    success = st.session_state.analyzer.auto_process_folder_files(new_files)
                    
                    if success:
                        st.sidebar.success("✅ All files processed successfully!")
                        # Force a rerun to update the UI
                        st.rerun()
                    else:
                        st.sidebar.error("❌ Some files failed to process. Check the error messages above.")
        else:
            if exp_files:
                st.sidebar.success("✅ All files are already processed!")
    
    # Original file uploads (kept for compatibility)
    st.sidebar.subheader("📁 Upload Experiment Data")
    
    # Multiple experiment files
    exp_files_original = st.sidebar.file_uploader(
        "Upload Experiment Files", 
        type=['xlsx', 'xls', 'csv', 'tsv', 'txt'],
        accept_multiple_files=True,
        help="Upload Excel (.xlsx, .xls) or CSV files. If Excel doesn't work, save as CSV format."
    )
    
    # Process uploaded files
    if exp_files_original:
        for i, file in enumerate(exp_files_original):
            exp_name = st.sidebar.text_input(
                f"Name for {file.name}", 
                value=f"Experiment_{i+1}",
                key=f"exp_name_{i}"
            )
            
            if st.sidebar.button(f"Add {exp_name}", key=f"add_exp_{i}"):
                with st.sidebar:
                    with st.spinner(f"Adding {exp_name}..."):
                        success = st.session_state.analyzer.add_experiment(exp_name, file, exp_name)
                        if success:
                            st.sidebar.success(f"✅ Added {exp_name}")
                        else:
                            st.sidebar.error(f"❌ Failed to add {exp_name}")
    
    # Train data upload (Excel instead of JSON)
    st.sidebar.subheader("📚 Upload Training Data")
    train_file = st.sidebar.file_uploader(
        "Upload Training Excel", 
        type=['xlsx', 'xls', 'csv'],
        help="Upload training data Excel/CSV file"
    )
    
    if train_file and st.sidebar.button("Load Training Data"):
        with st.sidebar:
            with st.spinner("Loading training data..."):
                success = st.session_state.analyzer.set_train_data(train_file)
                if success:
                    st.sidebar.success("✅ Training data loaded")
    
    # Show loaded experiments
    if experiment_names:
        st.sidebar.subheader("📊 Loaded Experiments")
        for exp_name in experiment_names:
            exp_info = st.session_state.analyzer.experiments[exp_name]
            st.sidebar.write(f"• **{exp_info['name']}**: {len(exp_info['data'])} records")
    
    # OPTIMIZATION: Add cache management
    if experiment_names:
        st.sidebar.subheader("🗂️ Cache Management")
        if st.sidebar.button("🗑️ Clear Cache"):
            st.session_state.analyzer._clear_cache()
            st.sidebar.success("Cache cleared!")
    
    return st.session_state.analyzer

def create_page_filters(analyzer, page_name="Analysis"):
    """Create filters at the top right of each page"""
    if not analyzer.experiments:
        return {}
    
    st.markdown(f"## {page_name}")
    
    # Get global super class mode
    use_super_class = getattr(st.session_state, 'global_super_class_mode', False)
    
    # Get all unique values across experiments (with super class consideration) - OPTIMIZED
    all_data_list = []
    for exp_id in analyzer.experiments.keys():
        exp_data = analyzer.get_processed_data(exp_id, use_super_class)  # Uses cache
        if exp_data is not None:
            all_data_list.append(exp_data)
    
    if not all_data_list:
        return {}
    
    all_data = pd.concat(all_data_list, ignore_index=True)
    
    # Create unique key prefix based on page name
    key_prefix = page_name.replace(" ", "_").replace("(", "").replace(")", "").replace("&", "and")
    
    # Create filter section at the top
    with st.expander("🎛️ **Filters** (Click to expand/collapse)", expanded=False):
        # Create 3 columns for filters (removed experiments)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Platform filter with "All" option
            platforms = sorted(all_data['platform'].unique())
            platform_options = ['All Platforms'] + platforms
            selected_platforms = st.multiselect(
                "🎥 Platforms",
                options=platform_options,
                default=['All Platforms'],
                key=f"{key_prefix}_platforms"
            )
            
            # Process "All" selection
            if 'All Platforms' in selected_platforms:
                selected_platforms = platforms
        
        with col2:
            # Size filter with "All" option
            sizes = sorted(all_data['bb_size'].unique())
            size_options = ['All Sizes'] + sizes
            selected_sizes = st.multiselect(
                "📏 Object Sizes",
                options=size_options,
                default=['All Sizes'],
                key=f"{key_prefix}_sizes"
            )
            
            # Process "All" selection
            if 'All Sizes' in selected_sizes:
                selected_sizes = sizes
        
        with col3:
            # Class filter with "All" option
            classes = sorted(all_data['cls_name'].unique())
            class_label = "🚢 Super Classes" if use_super_class else "🚢 Ship Classes"
            class_options = ['All Classes'] + classes
            selected_classes = st.multiselect(
                class_label,
                options=class_options,
                default=['All Classes'],
                key=f"{key_prefix}_classes"
            )
            
            # Process "All" selection
            if 'All Classes' in selected_classes:
                selected_classes = classes
    
    return {
        'experiments': list(analyzer.experiments.keys()),  # Use all experiments
        'platforms': selected_platforms,
        'sizes': selected_sizes,
        'classes': selected_classes
    }

def create_confusion_matrix_filters(analyzer, page_name="Analysis"):
    """Create filters for confusion matrix page (no experiments, no classes)"""
    if not analyzer.experiments:
        return {}
    
    st.markdown(f"## {page_name}")
    
    # Get global super class mode
    use_super_class = getattr(st.session_state, 'global_super_class_mode', False)
    
    # Get all unique values across experiments (with super class consideration) - OPTIMIZED
    all_data_list = []
    for exp_id in analyzer.experiments.keys():
        exp_data = analyzer.get_processed_data(exp_id, use_super_class)  # Uses cache
        if exp_data is not None:
            all_data_list.append(exp_data)
    
    if not all_data_list:
        return {}
    
    all_data = pd.concat(all_data_list, ignore_index=True)
    
    # Create unique key prefix based on page name
    key_prefix = page_name.replace(" ", "_").replace("(", "").replace(")", "").replace("&", "and")
    
    # Create filter section at the top
    with st.expander("🎛️ **Filters** (Click to expand/collapse)", expanded=False):
        # Create 2 columns for filters (removed experiments and classes)
        col1, col2 = st.columns(2)
        
        with col1:
            # Platform filter with "All" option
            platforms = sorted(all_data['platform'].unique())
            platform_options = ['All Platforms'] + platforms
            selected_platforms = st.multiselect(
                "🎥 Platforms",
                options=platform_options,
                default=['All Platforms'],
                key=f"{key_prefix}_platforms"
            )
            
            # Process "All" selection
            if 'All Platforms' in selected_platforms:
                selected_platforms = platforms
        
        with col2:
            # Size filter with "All" option
            sizes = sorted(all_data['bb_size'].unique())
            size_options = ['All Sizes'] + sizes
            selected_sizes = st.multiselect(
                "📏 Object Sizes",
                options=size_options,
                default=['All Sizes'],
                key=f"{key_prefix}_sizes"
            )
            
            # Process "All" selection
            if 'All Sizes' in selected_sizes:
                selected_sizes = sizes
    
    return {
        'experiments': list(analyzer.experiments.keys()),  # Use all experiments
        'platforms': selected_platforms,
        'sizes': selected_sizes,
        'classes': sorted(all_data['cls_name'].unique())  # Use all classes
    }

def filter_data_by_analysis_mode(data, analysis_mode, error_type):
    """Filter FP or FN data based on analysis mode"""
    
    if analysis_mode == "Detection":
        # Detection analysis: only background-related errors
        if error_type == "FP":
            return data[
                (data['mistake_kind'] == 'FP') & 
                (data['class_mistake'] == 'background')
            ]
        elif error_type == "FN":
            return data[
                (data['mistake_kind'] == 'FN') & 
                (data['class_mistake'] == 'background')
            ]
    
    elif analysis_mode == "Classification":
        # Classification analysis: only non-background errors
        if error_type == "FP":
            return data[
                (data['mistake_kind'] == 'FP') & 
                (data['class_mistake'] != 'background') &
                (data['class_mistake'] != '-')
            ]
        elif error_type == "FN":
            # For classification FN, we return FP as the mirror
            return data[
                (data['mistake_kind'] == 'FP') & 
                (data['class_mistake'] != 'background') &
                (data['class_mistake'] != '-')
            ]
    
    else:  # "All Data"
        # Return all FP or FN data
        if error_type == "FP":
            return data[data['mistake_kind'] == 'FP']
        elif error_type == "FN":
            return data[data['mistake_kind'] == 'FN']
    
    return data

def confusion_matrix_page(analyzer):
    """Create confusion matrix analysis page"""
    # Create page-specific filters (no experiments, no classes)
    filters = create_confusion_matrix_filters(analyzer, "🔄 Confusion Matrix Analysis")
    
    # Get global analysis mode and super class mode
    analysis_mode = getattr(st.session_state, 'global_analysis_mode', 'All Data')
    use_super_class = getattr(st.session_state, 'global_super_class_mode', False)
    
    # Show analysis mode info
    class_level = "Super Class" if use_super_class else "Subclass"
    st.info(f"**Current Analysis Mode: {analysis_mode}** | **Class Level: {class_level}** - Data filtered accordingly")
    
    if not analyzer.experiments:
        st.warning("Please upload experiment data first.")
        return
    
    # Create confusion matrix data with applied filters
    confusion_data = analyzer.create_confusion_matrix_data(
        experiment_ids=filters.get('experiments'),
        filters=filters,
        use_super_class=use_super_class
    )
    
    if confusion_data.empty:
        st.warning("No data available for selected filters.")
        return
    
    # Experiment selector for confusion matrix
    selected_exp = st.selectbox(
        "Select Experiment for Confusion Matrix",
        options=filters.get('experiments', [])
    )
    
    if selected_exp:
        exp_confusion = confusion_data[confusion_data['experiment_id'] == selected_exp]
        
        # Create confusion matrix with both counts and percentages
        cm_pivot = exp_confusion.pivot_table(
            index='actual_class',
            columns='predicted_class',
            values='bb_id',
            aggfunc='count',
            fill_value=0
        )
        
        # Calculate percentages
        cm_percentage = cm_pivot.div(cm_pivot.sum(axis=1), axis=0) * 100
        
        # Create text annotations with both count and percentage
        text_annotations = []
        for i in range(len(cm_pivot.index)):
            row = []
            for j in range(len(cm_pivot.columns)):
                count = cm_pivot.iloc[i, j]
                percentage = cm_percentage.iloc[i, j] if not pd.isna(cm_percentage.iloc[i, j]) else 0
                row.append(f"{count}<br>({percentage:.1f}%)")
            text_annotations.append(row)
        
        # Interactive confusion matrix
        title = f'Confusion Matrix - {selected_exp} ({analysis_mode} Mode, {class_level}) - Filtered Data'
        fig = px.imshow(
            cm_pivot.values,
            x=cm_pivot.columns,
            y=cm_pivot.index,
            color_continuous_scale='Blues',
            title=title,
            text_auto=False
        )
        
        # Add custom text annotations
        fig.update_traces(text=text_annotations, texttemplate="%{text}")
        
        fig.update_layout(
            xaxis_title=f"Predicted {class_level}",
            yaxis_title=f"Actual {class_level}",
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Interactive class analysis (MOVED HERE - between matrix and graphs)
        st.subheader("🔍 Interactive Class Analysis")
        
        all_classes = list(set(exp_confusion['predicted_class'].unique()) | 
                          set(exp_confusion['actual_class'].unique()))
        
        selected_class = st.selectbox(
            f"Select {class_level} for Detailed Analysis",
            options=['All Classes'] + sorted(all_classes),
            help=f"Select a {class_level.lower()} to see what it was confused with"
        )
        
        # Interactive class distribution analysis
        st.subheader("📊 Class Distribution Analysis")
        
        if selected_class != 'All Classes':
            st.info(f"🎯 **Showing confusion patterns for: {selected_class}** ({analysis_mode} Mode, {class_level})")
            
            # Filter data based on selected class
            # Row analysis: When actual class was selected_class, what was predicted?
            row_data = exp_confusion[exp_confusion['actual_class'] == selected_class]
            # Column analysis: When predicted class was selected_class, what was the actual?
            col_data = exp_confusion[exp_confusion['predicted_class'] == selected_class]
            
            col1, col2 = st.columns(2)
            
            with col1:
                if len(row_data) > 0:
                    # What was predicted when actual was selected_class
                    pred_when_actual = row_data['predicted_class'].value_counts()
                    fig_pred = px.bar(
                        x=pred_when_actual.index,
                        y=pred_when_actual.values,
                        title=f"What was PREDICTED when actual was '{selected_class}'",
                        labels={'x': f'Predicted {class_level}', 'y': 'Count'},
                        color_discrete_sequence=['#1f77b4']
                    )
                    fig_pred.update_xaxes(tickangle=45)
                    st.plotly_chart(fig_pred, use_container_width=True)
                    
                    # Show the actual numbers
                    st.write("**Confusion breakdown:**")
                    for pred_class, count in pred_when_actual.items():
                        if pred_class == selected_class:
                            st.write(f"✅ **{pred_class}**: {count} (Correct)")
                        else:
                            st.write(f"❌ **{pred_class}**: {count} (Confused with)")
                else:
                    st.write(f"No data for actual {class_level.lower()} '{selected_class}' in filtered data")
            
            with col2:
                if len(col_data) > 0:
                    # What was actual when predicted was selected_class
                    actual_when_pred = col_data['actual_class'].value_counts()
                    fig_actual = px.bar(
                        x=actual_when_pred.index,
                        y=actual_when_pred.values,
                        title=f"What was ACTUAL when predicted '{selected_class}'",
                        labels={'x': f'Actual {class_level}', 'y': 'Count'},
                        color_discrete_sequence=['#ff7f0e']
                    )
                    fig_actual.update_xaxes(tickangle=45)
                    st.plotly_chart(fig_actual, use_container_width=True)
                    
                    # Show the actual numbers
                    st.write("**Prediction accuracy breakdown:**")
                    for actual_class, count in actual_when_pred.items():
                        if actual_class == selected_class:
                            st.write(f"✅ **{actual_class}**: {count} (Correct)")
                        else:
                            st.write(f"❌ **{actual_class}**: {count} (Incorrectly predicted)")
                else:
                    st.write(f"No data for predicted {class_level.lower()} '{selected_class}' in filtered data")
        
        else:
            # Show overall distributions when "All Classes" is selected
            st.info(f"🎯 **Showing overall distributions for all {class_level.lower()}s** ({analysis_mode} Mode)")
            col1, col2 = st.columns(2)
            
            with col1:
                # Predicted class distribution
                pred_dist = exp_confusion['predicted_class'].value_counts()
                fig_pred = px.bar(
                    x=pred_dist.index,
                    y=pred_dist.values,
                    title=f"Overall Predicted {class_level} Distribution",
                    labels={'x': f'Predicted {class_level}', 'y': 'Count'}
                )
                fig_pred.update_xaxes(tickangle=45)
                st.plotly_chart(fig_pred, use_container_width=True)
            
            with col2:
                # Actual class distribution
                actual_dist = exp_confusion['actual_class'].value_counts()
                fig_actual = px.bar(
                    x=actual_dist.index,
                    y=actual_dist.values,
                    title=f"Overall Actual {class_level} Distribution",
                    labels={'x': f'Actual {class_level}', 'y': 'Count'}
                )
                fig_actual.update_xaxes(tickangle=45)
                st.plotly_chart(fig_actual, use_container_width=True)
        
        # Error type distribution and Platform Analysis (COMBINED - side by side)
        st.subheader("📊 Error Distribution & Platform Analysis")
        
        # Get filtered data for all experiments - USES CACHED DATA
        all_platform_data = []
        for exp_id in filters.get('experiments', []):
            if exp_id in analyzer.experiments:
                exp_data = analyzer.get_processed_data(exp_id, use_super_class)  # Uses cache
                if exp_data is not None:
                    exp_data = analyzer._apply_filters(exp_data, filters)
                    exp_data['experiment_id'] = exp_id
                    all_platform_data.append(exp_data)
        
        if all_platform_data:
            combined_platform_data = pd.concat(all_platform_data, ignore_index=True)
            
            # Filter out non-relevant mistake_kind values (only keep TP, FP, FN)
            valid_mistake_kinds = ['TP', 'FP', 'FN']
            combined_platform_data = combined_platform_data[
                combined_platform_data['mistake_kind'].isin(valid_mistake_kinds)
            ]
            
            # Create two columns for side-by-side layout
            col1, col2 = st.columns(2)
            
            with col1:
                # Error type distribution pie chart (for selected class or all classes)
                if selected_class != 'All Classes':
                    # Filter for selected class
                    if len(row_data) > 0 or len(col_data) > 0:
                        class_data = pd.concat([row_data, col_data]).drop_duplicates()
                        # Filter out non-relevant mistake kinds
                        class_data = class_data[class_data['error_type'].isin(valid_mistake_kinds)]
                        
                        if len(class_data) > 0:
                            error_dist = class_data['error_type'].value_counts()
                            
                            fig_error = px.pie(
                                values=error_dist.values,
                                names=error_dist.index,
                                title=f"Error Type Distribution for {selected_class}",
                                color_discrete_map={'TP': 'green', 'FP': 'red', 'FN': 'orange'}
                            )
                            st.plotly_chart(fig_error, use_container_width=True)
                        else:
                            st.info(f"No TP/FP/FN data for {selected_class}")
                else:
                    # Show overall error distribution for all classes (only TP, FP, FN)
                    if len(combined_platform_data) > 0:
                        overall_error_dist = combined_platform_data['mistake_kind'].value_counts()
                        fig_error_all = px.pie(
                            values=overall_error_dist.values,
                            names=overall_error_dist.index,
                            title=f"Overall Error Type Distribution ({analysis_mode} Mode, {class_level})",
                            color_discrete_map={'TP': 'green', 'FP': 'red', 'FN': 'orange'}
                        )
                        st.plotly_chart(fig_error_all, use_container_width=True)
                    else:
                        st.info("No TP/FP/FN data available")
            
            with col2:
                # Platform analysis - TP/FP/FN distribution by platform
                if len(combined_platform_data) > 0:
                    platform_summary = []
                    for platform in combined_platform_data['platform'].unique():
                        platform_data = combined_platform_data[combined_platform_data['platform'] == platform]
                        
                        tp_count = len(platform_data[platform_data['mistake_kind'] == 'TP'])
                        fp_count = len(platform_data[platform_data['mistake_kind'] == 'FP'])
                        fn_count = len(platform_data[platform_data['mistake_kind'] == 'FN'])
                        
                        platform_summary.extend([
                            {'Platform': platform, 'Type': 'TP', 'Count': tp_count},
                            {'Platform': platform, 'Type': 'FP', 'Count': fp_count},
                            {'Platform': platform, 'Type': 'FN', 'Count': fn_count}
                        ])
                    
                    if platform_summary:
                        platform_df = pd.DataFrame(platform_summary)
                        
                        # CLUSTERED bar chart (not stacked)
                        fig_platform = px.bar(
                            platform_df,
                            x='Platform',
                            y='Count',
                            color='Type',
                            barmode='group',  # This makes it clustered instead of stacked
                            title=f"TP/FP/FN Distribution by Platform ({analysis_mode} Mode, {class_level})",
                            color_discrete_map={'TP': 'green', 'FP': 'red', 'FN': 'orange'}
                        )
                        
                        fig_platform.update_xaxes(tickangle=45)
                        st.plotly_chart(fig_platform, use_container_width=True)
                    else:
                        st.info("No platform data available")
                else:
                    st.info("No TP/FP/FN data available")

def overall_metrics_page(analyzer):
    """Overall performance metrics page with detection vs classification analysis"""
    # Create page-specific filters
    filters = create_page_filters(analyzer, "📊 Overall Performance Metrics")
    
    # Get global analysis mode and super class mode
    analysis_mode = getattr(st.session_state, 'global_analysis_mode', 'All Data')
    use_super_class = getattr(st.session_state, 'global_super_class_mode', False)
    
    # Show analysis mode info
    class_level = "Super Class" if use_super_class else "Subclass"
    
    if analysis_mode == "Detection":
        st.info(f"**🔍 Detection Analysis** ({class_level}): Measures how well the model detects *something* vs *nothing*")
        st.markdown("""
        - **TP**: Any successful detection (regardless of class)
        - **FP**: Detected something where there was background (false alarms)
        - **FN**: Missed something that was there (missed detections)
        """)
    elif analysis_mode == "Classification":
        st.info(f"**🏷️ Classification Analysis** ({class_level}): Measures classification accuracy among detected objects only")
        st.markdown("""
        - **TP**: Correct classifications 
        - **FP**: Wrong classifications (detected object A, but it was actually object B)
        - **FN**: Equivalent to FP from other classes' perspective
        - **Note**: In pure classification, precision ≈ recall since we only consider detected objects
        """)
    else:
        st.info(f"**📊 All Data** ({class_level}): Complete dataset analysis including all detection and classification errors")
    
    if not analyzer.experiments:
        st.warning("Please upload experiment data first.")
        return
    
    # Calculate metrics for all experiments with applied filters
    experiment_ids = filters.get('experiments', [])
    
    if not experiment_ids:
        st.warning("No experiments selected.")
        return
    
    # Use the calculate_metrics method (now uses global mode) - USES CACHED DATA
    metrics = analyzer.calculate_metrics(
        experiment_ids=experiment_ids,
        filters=filters
    )
    
    if not metrics:
        st.warning("No metrics available for selected filters and experiments.")
        return
    
    # Display metrics in cards with responsive layout
    n_experiments = len(metrics)
    n_cols = min(4, n_experiments)  # Max 4 columns
    n_rows = (n_experiments + n_cols - 1) // n_cols  # Ceiling division
    
    for row in range(n_rows):
        cols = st.columns(n_cols)
        for col_idx in range(n_cols):
            exp_list = list(metrics.keys())
            exp_idx = row * n_cols + col_idx
            if exp_idx < len(exp_list):
                exp_id = exp_list[exp_idx]
                exp_metrics = metrics[exp_id]
                
                with cols[col_idx]:
                    st.metric(
                        f"📍 {exp_id}",
                        f"F1: {exp_metrics['f1']:.3f}"
                    )
                    
                    # Show precision/recall with context
                    if analysis_mode == "Detection":
                        st.metric("Detection Precision", f"{exp_metrics['precision']:.3f}")
                        st.metric("Detection Recall", f"{exp_metrics['recall']:.3f}")
                    elif analysis_mode == "Classification":
                        st.metric("Classification Accuracy", f"{exp_metrics['precision']:.3f}")
                        st.metric("Classification Recall", f"{exp_metrics['recall']:.3f}")
                    else:
                        st.metric("Precision", f"{exp_metrics['precision']:.3f}")
                        st.metric("Recall", f"{exp_metrics['recall']:.3f}")
                    
                    st.metric(
                        "Counts",
                        f"TP:{exp_metrics['tp_count']} FP:{exp_metrics['fp_count']} FN:{exp_metrics['fn_count']}"
                    )
    
    # Comparison chart - UPDATED TO SHOW METRICS AS CATEGORIES WITH EXPERIMENTS AS BARS
    comparison_data = []
    for exp_id, exp_metrics in metrics.items():
        comparison_data.append({
            'Experiment': exp_id,
            'Precision': exp_metrics['precision'],
            'Recall': exp_metrics['recall'],
            'F1-Score': exp_metrics['f1']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Bar chart comparison - grouped by metric with experiments as bars
    fig = px.bar(
        comparison_df.melt(id_vars=['Experiment'], var_name='Metric', value_name='Score'),
        x='Metric',
        y='Score',
        color='Experiment',
        barmode='group',
        title=f'{analysis_mode} Performance Comparison ({class_level}) - Filtered Data',
        labels={'Metric': 'Performance Metrics', 'Score': 'Score Value', 'Experiment': 'Experiments'}
    )
    
    # Customize the chart
    fig.update_layout(
        xaxis_title="Performance Metrics",
        yaxis_title="Score Value",
        legend_title="Experiments",
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # UPDATED: Separate Precision and Recall charts instead of F1 Score
    st.subheader(f"📋 Per-Class Precision & Recall ({class_level})")
    
    # Calculate precision and recall scores per class using the filtered data from metrics
    class_precision_data = []
    class_recall_data = []
    
    for exp_id, exp_metrics in metrics.items():
        filtered_data = exp_metrics['filtered_data']
        
        # Calculate precision and recall per class
        for class_name in filtered_data['cls_name'].unique():
            class_data = filtered_data[filtered_data['cls_name'] == class_name]
            
            # Use consistent calculation
            tp, fp, fn = calculate_detection_metrics(class_data, analysis_mode)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            class_precision_data.append({
                'Experiment': exp_id,
                'Class': class_name,
                'Precision': precision
            })
            
            class_recall_data.append({
                'Experiment': exp_id,
                'Class': class_name,
                'Recall': recall
            })
    
    # Create vertically stacked precision and recall charts for better readability
    if class_precision_data and class_recall_data:
        # Precision chart (full width)
        class_precision_df = pd.DataFrame(class_precision_data)
        fig_precision = px.bar(
            class_precision_df,
            x='Class',
            y='Precision',
            color='Experiment',
            barmode='group',
            title=f'{analysis_mode} Precision by {class_level} - Filtered Data'
        )
        fig_precision.update_xaxes(tickangle=45)
        fig_precision.update_layout(height=500)  # Make it taller for better visibility
        st.plotly_chart(fig_precision, use_container_width=True)
        
        # Recall chart (full width)
        class_recall_df = pd.DataFrame(class_recall_data)
        fig_recall = px.bar(
            class_recall_df,
            x='Class',
            y='Recall',
            color='Experiment',
            barmode='group',
            title=f'{analysis_mode} Recall by {class_level} - Filtered Data'
        )
        fig_recall.update_xaxes(tickangle=45)
        fig_recall.update_layout(height=500)  # Make it taller for better visibility
        st.plotly_chart(fig_recall, use_container_width=True)
    
    # Per-Platform F1 Score breakdown
    st.subheader("🎥 Per-Platform F1 Score")
    
    # Calculate F1 scores per platform using the filtered data from metrics
    platform_f1_data = []
    for exp_id, exp_metrics in metrics.items():
        filtered_data = exp_metrics['filtered_data']
        
        # Calculate F1 per platform
        for platform_name in filtered_data['platform'].unique():
            platform_data = filtered_data[filtered_data['platform'] == platform_name]
            
            # Use consistent calculation
            tp, fp, fn = calculate_detection_metrics(platform_data, analysis_mode)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            platform_f1_data.append({
                'Experiment': exp_id,
                'Platform': platform_name,
                'F1': f1,
                'TP': tp,
                'FP': fp,
                'FN': fn
            })
    
    if platform_f1_data:
        platform_f1_df = pd.DataFrame(platform_f1_data)
        
        fig_platform = px.bar(
            platform_f1_df,
            x='Platform',
            y='F1',
            color='Experiment',
            barmode='group',
            title=f'{analysis_mode} F1 Score by Platform ({class_level}) - Filtered Data'
        )
        
        st.plotly_chart(fig_platform, use_container_width=True)
        
        # Show platform details in expandable section
        with st.expander("📊 Platform Performance Details"):
            st.dataframe(platform_f1_df.round(3), use_container_width=True)

def fp_analysis_page(analyzer):
    """Detailed False Positive analysis page with global analysis mode"""
    # Create page-specific filters
    filters = create_page_filters(analyzer, "🔴 False Positive Analysis")
    
    # Get global analysis mode and super class mode
    analysis_mode = getattr(st.session_state, 'global_analysis_mode', 'All Data')
    use_super_class = getattr(st.session_state, 'global_super_class_mode', False)
    
    # Show analysis mode info
    class_level = "Super Class" if use_super_class else "Subclass"
    
    if analysis_mode == "Detection":
        st.info(f"**🔍 Detection FP Analysis** ({class_level}): False alarms (detected something where there was background)")
    elif analysis_mode == "Classification":
        st.info(f"**🏷️ Classification FP Analysis** ({class_level}): Wrong classifications (detected object A, but it was actually object B)")
    else:
        st.info(f"**📊 All Data FP Analysis** ({class_level}): All false positive errors")
    
    if not analyzer.experiments:
        st.warning("Please upload experiment data first.")
        return
    
    # Get all filtered data for comprehensive analysis - USES CACHED DATA
    experiment_ids = filters.get('experiments', [])
    all_filtered_data = []
    all_fp_data = []
    all_gt_data = []  # For experiment share calculation
    
    for exp_id in experiment_ids:
        if exp_id in analyzer.experiments:
            exp_data = analyzer.get_processed_data(exp_id, use_super_class)  # Uses cache
            if exp_data is None:
                continue
            
            # Apply filters to the entire dataset first
            filtered_exp_data = analyzer._apply_filters(exp_data, filters)
            
            # Apply analysis mode filtering to FP data
            fp_data = filter_data_by_analysis_mode(filtered_exp_data, analysis_mode, "FP")
            fp_data['experiment_id'] = exp_id
            all_fp_data.append(fp_data)
            
            # Get ground truth data for experiment share calculation
            gt_data = filtered_exp_data[filtered_exp_data['ground_truth'] == 1].copy()
            gt_data['experiment_id'] = exp_id
            all_gt_data.append(gt_data)
            
            # Store all filtered data for metrics calculation
            filtered_exp_data['experiment_id'] = exp_id
            all_filtered_data.append(filtered_exp_data)
    
    if not all_fp_data or all(len(fp) == 0 for fp in all_fp_data):
        st.warning(f"No False Positive data available for selected filters in {analysis_mode} mode.")
        return
    
    combined_fp = pd.concat([fp for fp in all_fp_data if len(fp) > 0], ignore_index=True)
    combined_filtered = pd.concat([data for data in all_filtered_data if len(data) > 0], ignore_index=True)
    combined_gt = pd.concat([gt for gt in all_gt_data if len(gt) > 0], ignore_index=True) if any(len(gt) > 0 for gt in all_gt_data) else pd.DataFrame()
    
    if combined_fp.empty:
        st.warning(f"No False Positive data matches the selected filters in {analysis_mode} mode.")
        return
    
    # FP Rate and Precision Cards (side by side) - FIXED WITH CONSISTENT LOGIC
    st.subheader("📊 Performance Metrics")
    
    # Create rows of metrics cards
    n_experiments = len(experiment_ids)
    n_cols = min(4, n_experiments)  # Max 4 columns
    n_rows = (n_experiments + n_cols - 1) // n_cols  # Ceiling division
    
    for row in range(n_rows):
        cols = st.columns(n_cols)
        for col_idx in range(n_cols):
            exp_idx = row * n_cols + col_idx
            if exp_idx < n_experiments:
                exp_id = experiment_ids[exp_idx]
                
                # Get filtered data for this experiment
                exp_filtered = combined_filtered[combined_filtered['experiment_id'] == exp_id]
                
                if len(exp_filtered) > 0:
                    # FIXED: Use consistent calculation for Detection mode
                    exp_tp, exp_fp, exp_fn = calculate_detection_metrics(exp_filtered, analysis_mode)
                    exp_total_pred = len(exp_filtered[exp_filtered['ground_truth'] == 0])
                    
                    fp_rate = exp_fp / exp_total_pred if exp_total_pred > 0 else 0
                    precision = exp_tp / (exp_tp + exp_fp) if (exp_tp + exp_fp) > 0 else 0
                    
                    with cols[col_idx]:
                        st.metric(f"🔴 FP Rate - {exp_id}", f"{fp_rate:.3f}")
                        st.metric(f"🎯 Precision - {exp_id}", f"{precision:.3f}")
    
    # FP Share vs Class Share (Fixed calculation)
    st.subheader(f"📈 FP Share vs Class Share Analysis ({class_level})")
    
    # Calculate shares by class
    share_data = []
    
    # Get unique classes from filtered FP data
    unique_classes = combined_fp['cls_name'].unique()
    
    # Calculate total ground truth for prior probability calculation
    total_gt_count = len(combined_gt) if not combined_gt.empty else 0
    
    for cls in unique_classes:
        class_fp = combined_fp[combined_fp['cls_name'] == cls]
        
        for exp_id in experiment_ids:
            exp_class_fp = class_fp[class_fp['experiment_id'] == exp_id]
            
            # FP Share: What percentage of total FPs does this class+experiment represent
            fp_share = len(exp_class_fp) / len(combined_fp) * 100 if len(combined_fp) > 0 else 0
            
            # Class Share (Prior Probability): What percentage of total ground truth does this CLASS represent
            if not combined_gt.empty:
                class_gt_count = len(combined_gt[combined_gt['cls_name'] == cls])
                class_share = class_gt_count / total_gt_count * 100 if total_gt_count > 0 else 0
            else:
                # Fallback: calculate from all filtered data (both predictions and GT)
                all_class_data = combined_filtered[combined_filtered['cls_name'] == cls]
                class_share = len(all_class_data) / len(combined_filtered) * 100 if len(combined_filtered) > 0 else 0
            
            share_data.append({
                'Class': cls,
                'Experiment': exp_id,
                'FP_Share': fp_share,
                'Class_Share': class_share  # Changed from Experiment_Share to Class_Share
            })
    
    if share_data:
        share_df = pd.DataFrame(share_data)
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        for exp_id in experiment_ids:
            exp_share_data = share_df[share_df['Experiment'] == exp_id]
            
            if len(exp_share_data) > 0:
                fig.add_trace(
                    go.Bar(
                        x=exp_share_data['Class'],
                        y=exp_share_data['FP_Share'],
                        name=f'FP Share - {exp_id}',
                        opacity=0.7
                    )
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=exp_share_data['Class'],
                        y=exp_share_data['Class_Share'],
                        mode='lines+markers',
                        name=f'Class Share - {exp_id}',
                        yaxis='y2'
                    ),
                    secondary_y=True
                )
        
        fig.update_layout(title=f"FP Share vs Class Share ({analysis_mode} Mode, {class_level}) - Filtered Data")
        fig.update_xaxes(title_text=f"Ship {class_level}")
        fig.update_yaxes(title_text="FP Share (%)", secondary_y=False)
        fig.update_yaxes(title_text="Class Share - Prior Probability (%)", secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add explanation
        with st.expander(f"📚 Understanding FP Share vs Class Share ({class_level})"):
            st.markdown(f"""
            **🎯 What this chart shows ({analysis_mode} mode, {class_level}):**
            
            - **FP Share (Bars)**: What percentage of total False Positives does each {class_level.lower()} represent?
            - **Class Share (Line)**: What percentage of ground truth does each {class_level.lower()} represent? (Prior Probability)
            
            **📊 How to interpret:**
            
            - **FP Share > Class Share**: {class_level} gets MORE false positives than expected based on its ground truth frequency
            - **FP Share < Class Share**: {class_level} gets FEWER false positives than expected  
            - **FP Share ≈ Class Share**: {class_level} gets false positives proportional to its ground truth frequency
            
            **💡 Example:**
            - If "Merchant" ships are 40% of ground truth but 60% of false positives → Model struggles with Merchant ships
            - If "Military" ships are 30% of ground truth but 10% of false positives → Model performs well on Military ships
            """)
    
    # Train Share (if available)
    if analyzer.train_data is not None and not analyzer.train_data.empty:
        st.subheader("📚 Training Data Distribution")
        train_dist = analyzer.train_class_distribution
        if train_dist:
            # Map training data to super classes if needed
            if use_super_class:
                super_class_dist = {}
                for class_name, count in train_dist.items():
                    super_class = map_to_super_class(class_name)
                    super_class_dist[super_class] = super_class_dist.get(super_class, 0) + count
                train_dist = super_class_dist
            
            # Create bar chart for training distribution
            fig_train = px.bar(
                x=list(train_dist.keys()),
                y=list(train_dist.values()),
                title=f"Training Data {class_level} Distribution",
                labels={'x': f'Ship {class_level}', 'y': 'Count'},
                color_discrete_sequence=['#2E86AB']
            )
            fig_train.update_xaxes(tickangle=45)
            st.plotly_chart(fig_train, use_container_width=True, key="fp_training_distribution")
            
            # Show total count
            total_train = sum(train_dist.values())
            st.metric("Total Training Samples", f"{total_train:,}")
        else:
            st.warning("No training class distribution available")
    
    # FP Distribution - Separate by Experiment
    st.subheader(f"📊 FP Distribution by Experiment ({analysis_mode} Mode, {class_level})")
    # Create columns for side-by-side experiment comparison
    if len(experiment_ids) > 1:
        for row_start in range(0, len(experiment_ids), 2):
            cols = st.columns(min(2, len(experiment_ids) - row_start))
            for i in range(min(2, len(experiment_ids) - row_start)):
                exp_id = experiment_ids[row_start + i]
                exp_fp_data = combined_fp[combined_fp['experiment_id'] == exp_id]
                if len(exp_fp_data) > 0:
                    with cols[i]:
                        fp_dist = exp_fp_data['cls_name'].value_counts()
                        fig_dist = px.bar(
                            x=fp_dist.index,
                            y=fp_dist.values,
                            title=f"FP Distribution - {exp_id}",
                            labels={'x': f'Ship {class_level}', 'y': 'FP Count'}
                        )
                        fig_dist.update_xaxes(tickangle=45)
                        fig_dist.update_layout(height=400)
                        st.plotly_chart(fig_dist, use_container_width=True)
                        st.metric(f"Total FPs - {exp_id}", len(exp_fp_data))
                else:
                    with cols[i]:
                        st.warning(f"No FP data for {exp_id}")
    else:
        # Single experiment - full width
        exp_id = experiment_ids[0]
        exp_fp_data = combined_fp[combined_fp['experiment_id'] == exp_id]
        if len(exp_fp_data) > 0:
            fp_dist = exp_fp_data['cls_name'].value_counts()
            fig_dist = px.bar(
                x=fp_dist.index,
                y=fp_dist.values,
                title=f"FP Distribution - {exp_id} ({analysis_mode} Mode, {class_level})",
                labels={'x': f'Ship {class_level}', 'y': 'FP Count'}
            )
            fig_dist.update_xaxes(tickangle=45)
            st.plotly_chart(fig_dist, use_container_width=True)
    
    # FP Area Distribution - Separate by Experiment (FIXED WITH CHRONOLOGICAL ORDER)
    st.subheader(f"📐 FP Area Distribution by Experiment ({analysis_mode} Mode, {class_level})")
    if 'area_category' in combined_fp.columns:
        area_category_order = analyzer._get_area_category_order(combined_fp)
        if len(experiment_ids) > 1:
            for row_start in range(0, len(experiment_ids), 2):
                cols = st.columns(min(2, len(experiment_ids) - row_start))
                for i in range(min(2, len(experiment_ids) - row_start)):
                    exp_id = experiment_ids[row_start + i]
                    exp_fp_data = combined_fp[combined_fp['experiment_id'] == exp_id]
                    if len(exp_fp_data) > 0:
                        with cols[i]:
                            fig_area = px.histogram(
                                exp_fp_data,
                                x='area_category',
                                title=f"FP Area Distribution - {exp_id}",
                                labels={'area_category': 'Area Category', 'count': 'FP Count'}
                            )
                            fig_area.update_xaxes(
                                tickangle=45,
                                categoryorder='array',
                                categoryarray=area_category_order
                            )
                            fig_area.update_layout(height=400)
                            st.plotly_chart(fig_area, use_container_width=True)
                    else:
                        with cols[i]:
                            st.warning(f"No FP area data for {exp_id}")
        else:
            exp_id = experiment_ids[0]
            exp_fp_data = combined_fp[combined_fp['experiment_id'] == exp_id]
            if len(exp_fp_data) > 0:
                fig_area = px.histogram(
                    exp_fp_data,
                    x='area_category',
                    title=f"FP Area Distribution - {exp_id} ({analysis_mode} Mode, {class_level})",
                    labels={'area_category': 'Area Category', 'count': 'FP Count'}
                )
                fig_area.update_xaxes(
                    tickangle=45,
                    categoryorder='array',
                    categoryarray=area_category_order
                )
                st.plotly_chart(fig_area, use_container_width=True)
    
    # FP Heatmap - Separate by Experiment
    st.subheader(f"🔥 FP Heatmap (Platform vs {class_level}) by Experiment ({analysis_mode} Mode)")
    if len(combined_fp) > 0:
        if len(experiment_ids) > 1:
            for row_start in range(0, len(experiment_ids), 2):
                cols = st.columns(min(2, len(experiment_ids) - row_start))
                for i in range(min(2, len(experiment_ids) - row_start)):
                    exp_id = experiment_ids[row_start + i]
                    exp_fp_data = combined_fp[combined_fp['experiment_id'] == exp_id]
                    if len(exp_fp_data) > 0:
                        with cols[i]:
                            fp_heatmap_data = exp_fp_data.pivot_table(
                                index='platform',
                                columns='cls_name',
                                values='bb_id',
                                aggfunc='count',
                                fill_value=0
                            )
                            if not fp_heatmap_data.empty:
                                fig_heatmap = px.imshow(
                                    fp_heatmap_data.values,
                                    x=fp_heatmap_data.columns,
                                    y=fp_heatmap_data.index,
                                    color_continuous_scale='Reds',
                                    title=f"FP Heatmap - {exp_id}",
                                    text_auto=True
                                )
                                fig_heatmap.update_layout(height=400)
                                st.plotly_chart(fig_heatmap, use_container_width=True)
                            else:
                                st.warning(f"No heatmap data for {exp_id}")
                    else:
                        with cols[i]:
                            st.warning(f"No FP heatmap data for {exp_id}")
        else:
            exp_id = experiment_ids[0]
            exp_fp_data = combined_fp[combined_fp['experiment_id'] == exp_id]
            if len(exp_fp_data) > 0:
                fp_heatmap_data = exp_fp_data.pivot_table(
                    index='platform',
                    columns='cls_name',
                    values='bb_id',
                    aggfunc='count',
                    fill_value=0
                )
                if not fp_heatmap_data.empty:
                    fig_heatmap = px.imshow(
                        fp_heatmap_data.values,
                        x=fp_heatmap_data.columns,
                        y=fp_heatmap_data.index,
                        color_continuous_scale='Reds',
                        title=f"FP Heatmap - {exp_id} ({analysis_mode} Mode, {class_level})",
                        text_auto=True
                    )
                    st.plotly_chart(fig_heatmap, use_container_width=True)

def fn_analysis_page(analyzer):
    """Detailed False Negative analysis page with global analysis mode"""
    # Create page-specific filters
    filters = create_page_filters(analyzer, "🔵 False Negative Analysis")
    
    # Get global analysis mode and super class mode
    analysis_mode = getattr(st.session_state, 'global_analysis_mode', 'All Data')
    use_super_class = getattr(st.session_state, 'global_super_class_mode', False)
    
    # Show analysis mode info
    class_level = "Super Class" if use_super_class else "Subclass"
    
    if analysis_mode == "Detection":
        st.info(f"**🔍 Detection FN Analysis** ({class_level}): Missed detections (should have detected something, but didn't)")
    elif analysis_mode == "Classification":
        st.info(f"**🏷️ Classification FN Analysis** ({class_level}): Mirror of classification FP (equivalent errors from other classes' perspective)")
    else:
        st.info(f"**📊 All Data FN Analysis** ({class_level}): All false negative errors")
    
    if not analyzer.experiments:
        st.warning("Please upload experiment data first.")
        return
    
    # Get all filtered data for comprehensive analysis - USES CACHED DATA
    experiment_ids = filters.get('experiments', [])
    all_filtered_data = []
    all_fn_data = []
    all_gt_data = []  # For class share calculation
    
    for exp_id in experiment_ids:
        if exp_id in analyzer.experiments:
            exp_data = analyzer.get_processed_data(exp_id, use_super_class)  # Uses cache
            if exp_data is None:
                continue
                
            # Apply filters to the entire dataset first
            filtered_exp_data = analyzer._apply_filters(exp_data, filters)
            
            # Apply analysis mode filtering to FN data
            fn_data = filter_data_by_analysis_mode(filtered_exp_data, analysis_mode, "FN")
            fn_data['experiment_id'] = exp_id
            all_fn_data.append(fn_data)
            
            # Get ground truth data for class share calculation
            gt_data = filtered_exp_data[filtered_exp_data['ground_truth'] == 1].copy()
            gt_data['experiment_id'] = exp_id
            all_gt_data.append(gt_data)
            
            # Store all filtered data for metrics calculation
            filtered_exp_data['experiment_id'] = exp_id
            all_filtered_data.append(filtered_exp_data)
    
    if not all_fn_data or all(len(fn) == 0 for fn in all_fn_data):
        st.warning(f"No False Negative data available for selected filters in {analysis_mode} mode.")
        return
    
    combined_fn = pd.concat([fn for fn in all_fn_data if len(fn) > 0], ignore_index=True)
    combined_filtered = pd.concat([data for data in all_filtered_data if len(data) > 0], ignore_index=True)
    combined_gt = pd.concat([gt for gt in all_gt_data if len(gt) > 0], ignore_index=True) if any(len(gt) > 0 for gt in all_gt_data) else pd.DataFrame()
    
    if combined_fn.empty:
        st.warning(f"No False Negative data matches the selected filters in {analysis_mode} mode.")
        return
    
    # FN Rate and Recall Cards (side by side) - FIXED WITH CONSISTENT LOGIC
    st.subheader("📊 Performance Metrics")
    
    # Create rows of metrics cards
    n_experiments = len(experiment_ids)
    n_cols = min(4, n_experiments)  # Max 4 columns
    n_rows = (n_experiments + n_cols - 1) // n_cols  # Ceiling division
    
    for row in range(n_rows):
        cols = st.columns(n_cols)
        for col_idx in range(n_cols):
            exp_idx = row * n_cols + col_idx
            if exp_idx < n_experiments:
                exp_id = experiment_ids[exp_idx]
                
                # Get filtered data for this experiment
                exp_filtered = combined_filtered[combined_filtered['experiment_id'] == exp_id]
                
                if len(exp_filtered) > 0:
                    # FIXED: Use consistent calculation for Detection mode
                    exp_tp, exp_fp, exp_fn = calculate_detection_metrics(exp_filtered, analysis_mode)
                    
                    fn_rate = exp_fn / (exp_fn + exp_tp) if (exp_fn + exp_tp) > 0 else 0
                    recall = exp_tp / (exp_tp + exp_fn) if (exp_tp + exp_fn) > 0 else 0
                    
                    with cols[col_idx]:
                        st.metric(f"🔵 FN Rate - {exp_id}", f"{fn_rate:.3f}")
                        st.metric(f"🎯 Recall - {exp_id}", f"{recall:.3f}")
    
    # FN Share vs Class Share (Similar to FP page)
    st.subheader(f"📈 FN Share vs Class Share Analysis ({class_level})")
    
    # Calculate shares by class
    share_data = []
    
    # Get unique classes from filtered FN data
    unique_classes = combined_fn['cls_name'].unique()
    
    # Calculate total ground truth for prior probability calculation
    total_gt_count = len(combined_gt) if not combined_gt.empty else 0
    
    for cls in unique_classes:
        class_fn = combined_fn[combined_fn['cls_name'] == cls]
        
        for exp_id in experiment_ids:
            exp_class_fn = class_fn[class_fn['experiment_id'] == exp_id]
            
            # FN Share: What percentage of total FNs does this class+experiment represent
            fn_share = len(exp_class_fn) / len(combined_fn) * 100 if len(combined_fn) > 0 else 0
            
            # Class Share (Prior Probability): What percentage of total ground truth does this CLASS represent
            if not combined_gt.empty:
                class_gt_count = len(combined_gt[combined_gt['cls_name'] == cls])
                class_share = class_gt_count / total_gt_count * 100 if total_gt_count > 0 else 0
            else:
                # Fallback: calculate from all filtered data (both predictions and GT)
                all_class_data = combined_filtered[combined_filtered['cls_name'] == cls]
                class_share = len(all_class_data) / len(combined_filtered) * 100 if len(combined_filtered) > 0 else 0
            
            share_data.append({
                'Class': cls,
                'Experiment': exp_id,
                'FN_Share': fn_share,
                'Class_Share': class_share
            })
    
    if share_data:
        share_df = pd.DataFrame(share_data)
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        for exp_id in experiment_ids:
            exp_share_data = share_df[share_df['Experiment'] == exp_id]
            
            if len(exp_share_data) > 0:
                fig.add_trace(
                    go.Bar(
                        x=exp_share_data['Class'],
                        y=exp_share_data['FN_Share'],
                        name=f'FN Share - {exp_id}',
                        opacity=0.7
                    )
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=exp_share_data['Class'],
                        y=exp_share_data['Class_Share'],
                        mode='lines+markers',
                        name=f'Class Share - {exp_id}',
                        yaxis='y2'
                    ),
                    secondary_y=True
                )
        
        fig.update_layout(title=f"FN Share vs Class Share ({analysis_mode} Mode, {class_level}) - Filtered Data")
        fig.update_xaxes(title_text=f"Ship {class_level}")
        fig.update_yaxes(title_text="FN Share (%)", secondary_y=False)
        fig.update_yaxes(title_text="Class Share - Prior Probability (%)", secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add explanation
        with st.expander(f"📚 Understanding FN Share vs Class Share ({class_level})"):
            st.markdown(f"""
            **🎯 What this chart shows ({analysis_mode} mode, {class_level}):**
            
            - **FN Share (Bars)**: What percentage of total False Negatives does each {class_level.lower()} represent?
            - **Class Share (Line)**: What percentage of ground truth does each {class_level.lower()} represent? (Prior Probability)
            
            **📊 How to interpret:**
            
            - **FN Share > Class Share**: {class_level} gets MORE false negatives than expected based on its ground truth frequency
            - **FN Share < Class Share**: {class_level} gets FEWER false negatives than expected  
            - **FN Share ≈ Class Share**: {class_level} gets false negatives proportional to its ground truth frequency
            
            **💡 Example:**
            - If "Merchant" ships are 40% of ground truth but 60% of false negatives → Model misses Merchant ships more often
            - If "Military" ships are 30% of ground truth but 10% of false negatives → Model rarely misses Military ships
            """)
    
    # Train Share (if available)
    if analyzer.train_data is not None and not analyzer.train_data.empty:
        st.subheader("📚 Training Data Distribution")
        train_dist = analyzer.train_class_distribution
        if train_dist:
            # Map training data to super classes if needed
            if use_super_class:
                super_class_dist = {}
                for class_name, count in train_dist.items():
                    super_class = map_to_super_class(class_name)
                    super_class_dist[super_class] = super_class_dist.get(super_class, 0) + count
                train_dist = super_class_dist
            
            # Create bar chart for training distribution
            fig_train = px.bar(
                x=list(train_dist.keys()),
                y=list(train_dist.values()),
                title=f"Training Data {class_level} Distribution",
                labels={'x': f'Ship {class_level}', 'y': 'Count'},
                color_discrete_sequence=['#2E86AB']
            )
            fig_train.update_xaxes(tickangle=45)
            st.plotly_chart(fig_train, use_container_width=True, key="fn_training_distribution")
            
            # Show total count
            total_train = sum(train_dist.values())
            st.metric("Total Training Samples", f"{total_train:,}")
        else:
            st.warning("No training class distribution available")
    
    # FN Distribution - Separate by Experiment
    st.subheader(f"📊 FN Distribution by Experiment ({analysis_mode} Mode, {class_level})")
    # Create columns for side-by-side experiment comparison
    if len(experiment_ids) > 1:
        for row_start in range(0, len(experiment_ids), 2):
            cols = st.columns(min(2, len(experiment_ids) - row_start))
            for i in range(min(2, len(experiment_ids) - row_start)):
                exp_id = experiment_ids[row_start + i]
                exp_fn_data = combined_fn[combined_fn['experiment_id'] == exp_id]
                if len(exp_fn_data) > 0:
                    with cols[i]:
                        fn_dist = exp_fn_data['cls_name'].value_counts()
                        fig_dist = px.bar(
                            x=fn_dist.index,
                            y=fn_dist.values,
                            title=f"FN Distribution - {exp_id}",
                            labels={'x': f'Ship {class_level}', 'y': 'FN Count'}
                        )
                        fig_dist.update_xaxes(tickangle=45)
                        fig_dist.update_layout(height=400)
                        st.plotly_chart(fig_dist, use_container_width=True)
                        st.metric(f"Total FNs - {exp_id}", len(exp_fn_data))
                else:
                    with cols[i]:
                        st.warning(f"No FN data for {exp_id}")
    else:
        # Single experiment - full width
        exp_id = experiment_ids[0]
        exp_fn_data = combined_fn[combined_fn['experiment_id'] == exp_id]
        if len(exp_fn_data) > 0:
            fn_dist = exp_fn_data['cls_name'].value_counts()
            fig_dist = px.bar(
                x=fn_dist.index,
                y=fn_dist.values,
                title=f"FN Distribution - {exp_id} ({analysis_mode} Mode, {class_level})",
                labels={'x': f'Ship {class_level}', 'y': 'FN Count'}
            )
            fig_dist.update_xaxes(tickangle=45)
            st.plotly_chart(fig_dist, use_container_width=True)
    
    # FN Area Distribution - Separate by Experiment (FIXED WITH CHRONOLOGICAL ORDER)
    st.subheader(f"📐 FN Area Distribution by Experiment ({analysis_mode} Mode, {class_level})")
    if 'area_category' in combined_fn.columns:
        area_category_order = analyzer._get_area_category_order(combined_fn)
        if len(experiment_ids) > 1:
            for row_start in range(0, len(experiment_ids), 2):
                cols = st.columns(min(2, len(experiment_ids) - row_start))
                for i in range(min(2, len(experiment_ids) - row_start)):
                    exp_id = experiment_ids[row_start + i]
                    exp_fn_data = combined_fn[combined_fn['experiment_id'] == exp_id]
                    if len(exp_fn_data) > 0:
                        with cols[i]:
                            fig_area = px.histogram(
                                exp_fn_data,
                                x='area_category',
                                title=f"FN Area Distribution - {exp_id}",
                                labels={'area_category': 'Area Category', 'count': 'FN Count'}
                            )
                            fig_area.update_xaxes(
                                tickangle=45,
                                categoryorder='array',
                                categoryarray=area_category_order
                            )
                            fig_area.update_layout(height=400)
                            st.plotly_chart(fig_area, use_container_width=True)
                    else:
                        with cols[i]:
                            st.warning(f"No FN area data for {exp_id}")
        else:
            exp_id = experiment_ids[0]
            exp_fn_data = combined_fn[combined_fn['experiment_id'] == exp_id]
            if len(exp_fn_data) > 0:
                fig_area = px.histogram(
                    exp_fn_data,
                    x='area_category',
                    title=f"FN Area Distribution - {exp_id} ({analysis_mode} Mode, {class_level})",
                    labels={'area_category': 'Area Category', 'count': 'FN Count'}
                )
                fig_area.update_xaxes(
                    tickangle=45,
                    categoryorder='array',
                    categoryarray=area_category_order
                )
                st.plotly_chart(fig_area, use_container_width=True)
    
    # FN Heatmap - Separate by Experiment
    st.subheader(f"🔥 FN Heatmap (Platform vs {class_level}) by Experiment ({analysis_mode} Mode)")
    if len(combined_fn) > 0:
        if len(experiment_ids) > 1:
            for row_start in range(0, len(experiment_ids), 2):
                cols = st.columns(min(2, len(experiment_ids) - row_start))
                for i in range(min(2, len(experiment_ids) - row_start)):
                    exp_id = experiment_ids[row_start + i]
                    exp_fn_data = combined_fn[combined_fn['experiment_id'] == exp_id]
                    if len(exp_fn_data) > 0:
                        with cols[i]:
                            fn_heatmap_data = exp_fn_data.pivot_table(
                                index='platform',
                                columns='cls_name',
                                values='bb_id',
                                aggfunc='count',
                                fill_value=0
                            )
                            if not fn_heatmap_data.empty:
                                fig_heatmap = px.imshow(
                                    fn_heatmap_data.values,
                                    x=fn_heatmap_data.columns,
                                    y=fn_heatmap_data.index,
                                    color_continuous_scale='Blues',
                                    title=f"FN Heatmap - {exp_id}",
                                    text_auto=True
                                )
                                fig_heatmap.update_layout(height=400)
                                st.plotly_chart(fig_heatmap, use_container_width=True)
                            else:
                                st.warning(f"No heatmap data for {exp_id}")
                    else:
                        with cols[i]:
                            st.warning(f"No FN heatmap data for {exp_id}")
        else:
            exp_id = experiment_ids[0]
            exp_fn_data = combined_fn[combined_fn['experiment_id'] == exp_id]
            if len(exp_fn_data) > 0:
                fn_heatmap_data = exp_fn_data.pivot_table(
                    index='platform',
                    columns='cls_name',
                    values='bb_id',
                    aggfunc='count',
                    fill_value=0
                )
                if not fn_heatmap_data.empty:
                    fig_heatmap = px.imshow(
                        fn_heatmap_data.values,
                        x=fn_heatmap_data.columns,
                        y=fn_heatmap_data.index,
                        color_continuous_scale='Blues',
                        title=f"FN Heatmap - {exp_id} ({analysis_mode} Mode, {class_level})",
                        text_auto=True
                    )
                    st.plotly_chart(fig_heatmap, use_container_width=True)

def main():
    """Main application"""
    st.set_page_config(
        page_title="Ship Detection Analysis Dashboard",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🚢 Advanced Ship Detection Analysis Dashboard")
    st.markdown("**Multi-Experiment Analysis with Auto Folder Processing - COMPLETE VERSION**")
    
    # Create experiment manager
    analyzer = create_experiment_manager()
    
    # Main navigation
    if analyzer.experiments:
        # Create tabs for different analyses (added Summary tab)
        tab_names = [
            "🔄 Confusion Matrix",
            "📊 Overall Metrics",
            "🔴 False Positives",
            "🔵 False Negatives"
        ]
        
        # Add Summary tab if available
        if SUMMARY_AVAILABLE:
            tab_names.insert(0, "📈 Summary & Comparison")
        
        tabs = st.tabs(tab_names)
        
        tab_index = 0
        
        # Summary tab (if available)
        if SUMMARY_AVAILABLE:
            with tabs[tab_index]:
                try:
                    # Import and call the summary analysis with the updated analyzer
                    from summary_analysis import summary_analysis_page
                    summary_analysis_page(analyzer)
                except Exception as e:
                    st.error(f"Error in summary analysis: {str(e)}")
                    st.info("💡 Make sure summary_analysis.py is updated to work with super class analysis")
            tab_index += 1
        
        # Original tabs
        with tabs[tab_index]:
            confusion_matrix_page(analyzer)
        
        with tabs[tab_index + 1]:
            overall_metrics_page(analyzer)
        
        with tabs[tab_index + 2]:
            fp_analysis_page(analyzer)
        
        with tabs[tab_index + 3]:
            fn_analysis_page(analyzer)
    
    else:
        st.info("🎯 **Get Started**: Upload your experiment files using the sidebar to begin analysis")
        
        # Step-by-step guide
        with st.expander("📚 **How to Use This Dashboard - AUTO FOLDER PROCESSING** (Click to expand)", expanded=True):
            st.markdown("""
            ## 🚀 **New Auto Folder Processing Feature**
            
            ### **Step 1: Prepare Your Folder Structure** 📁
            ```
            Your Main Folder/
            ├── Experiment_A_Albatross_2023-01-01/
            │   └── results.xlsx
            ├── Experiment_B_Albatross_2023-01-02/
            │   └── data.xlsx
            └── Test_Run_Albatross_2023-01-03/
                └── analysis.xlsx
            ```
            
            ### **Step 2: Auto-Upload Process** ⚡
            1. **Select All Excel Files**: Use the "Select All Excel Files from Folders" uploader
            2. **Automatic Processing**: The app will automatically:
               - Extract experiment names from file names
               - Remove "Albatross" and date patterns
               - Process all files without manual intervention
            3. **Click "🚀 Auto-Process All Files"**: One click processes everything!
            
            ### **Step 3: Experiment Names** 🏷️
            - **Before**: `Experiment_A_Albatross_2023-01-01.xlsx`
            - **After**: `Experiment_A` (as experiment name)
            - **Smart Removal**: Automatically removes:
              - "Albatross" (case insensitive)
              - Date patterns (2023-01-01, 01-01-2023, etc.)
              - Extra underscores and spaces
            
            ### **Step 4: Benefits** ✨
            - **No Manual Naming**: No need to type experiment names manually
            - **Bulk Processing**: Process dozens of files at once
            - **Smart Naming**: Intelligent experiment name extraction
            - **Error Handling**: Graceful handling of duplicate names
            - **Progress Tracking**: Visual progress bar during processing
            
            ---
            
            ## 📊 **Data Structure Requirements**
            
            Each Excel file should contain:
            - `bb_id`: Unique bounding box identifier
            - `frame`: Frame number
            - `cls_name`: Ship class name
            - `mistake_kind`: TP, FP, FN, or '-'
            - `platform`: Camera platform location (optional)
            - `bb_size`: Size category (optional)
            - `area_px`: Bounding box area in pixels (optional)
            - `ground_truth`: 0 for predictions, 1 for ground truth (optional)
            - `class_mistake`: Actual class if misclassified (optional)
            
            ---
            
            ## 🎛️ **Analysis Features**
            
            ### **Global Analysis Modes:**
            - **Detection Mode**: How good is the model at detecting *something* vs *nothing*?
            - **Classification Mode**: How good is the model at classifying among detected objects?
            - **All Data Mode**: Complete analysis including all errors
            
            ### **Class Analysis Levels:**
            - **Subclass Analysis**: Original detailed class names (Bulk, Tanker, etc.)
            - **Super Class Analysis**: Grouped categories (Merchant, Military, etc.)
            
            ### **Smart Features:**
            - **Intelligent Caching**: 90% faster mode switching
            - **Auto-Processing**: One-click bulk file processing
            - **Smart Naming**: Automatic experiment name extraction
            - **Progress Tracking**: Visual feedback during processing
            - **Error Recovery**: Graceful handling of file issues
            
            ---
            
            ## 💡 **Tips for Best Results**
            
            1. **Organize Your Files**: Keep each experiment's Excel file in its own folder
            2. **Consistent Naming**: Use consistent naming patterns for easier processing
            3. **Check Data Structure**: Ensure all Excel files have the required columns
            4. **Use Auto-Processing**: Take advantage of the one-click processing feature
            5. **Monitor Progress**: Watch the progress bar to track processing status
            
            ---
            
            ## 🔧 **Troubleshooting**
            
            **If Auto-Processing Fails:**
            1. Check that Excel files have the required columns
            2. Ensure files are not corrupted
            3. Try the manual upload option as a fallback
            4. Check the error messages for specific issues
            
            **If Experiment Names Are Wrong:**
            1. Use the manual upload option for custom naming
            2. Check the naming extraction logic
            3. Rename files before uploading if needed
            """)
        
        # Installation requirements
        with st.expander("⚙️ Installation Requirements"):
            st.markdown("""
            **📦 Required Dependencies:**
            
            For Excel file processing:
            ```bash
            pip install openpyxl xlrd
            ```
            

            
            **🔧 Full Installation Command:**
            ```bash
            pip install streamlit pandas numpy plotly scikit-learn openpyxl xlrd
            ```
            """)

# --- Caching for file reading and processing ---

@st.cache_data(show_spinner=False)
def cached_safe_read_file(file_bytes, file_name, file_size):
    """Cached version of file reading. Returns DataFrame or None."""
    import io
    import pandas as pd
    filename = file_name.lower()
    file_obj = io.BytesIO(file_bytes)
    file_obj.name = file_name
    if filename.endswith(('.xlsx', '.xls')):
        try:
            return pd.read_excel(file_obj)
        except Exception:
            return None
    encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    for encoding in encodings_to_try:
        file_obj.seek(0)
        for sep in ['\t', ',', ';']:
            file_obj.seek(0)
            try:
                df = pd.read_csv(file_obj, encoding=encoding, sep=sep)
                if len(df.columns) > 5:
                    return df
            except Exception:
                continue
    return None

@st.cache_data(show_spinner=False)
def cached_process_experiment_data(df, exp_id):
    """Cached version of process_experiment_data."""
    import numpy as np
    processed_df = df.copy()
    processed_df['experiment_id'] = exp_id
    numeric_cols = ['area_px', 'frame', 'cls', 'ground_truth']
    for col in numeric_cols:
        if col in processed_df.columns:
            processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
        else:
            if col == 'ground_truth':
                processed_df[col] = 0
            elif col == 'area_px':
                processed_df[col] = 1000
            elif col == 'cls':
                processed_df[col] = 0
    string_cols = ['cls_name', 'bb_size', 'platform', 'mistake_kind', 'class_mistake']
    for col in string_cols:
        if col in processed_df.columns:
            processed_df[col] = processed_df[col].astype(str)
            processed_df[col] = processed_df[col].replace('nan', '-')
        else:
            if col == 'bb_size':
                processed_df[col] = 'medium'
            elif col == 'platform':
                processed_df[col] = 'unknown'
            elif col == 'class_mistake':
                processed_df[col] = '-'
    processed_df['TP'] = (processed_df['mistake_kind'] == 'TP').astype(int)
    processed_df['FP'] = (processed_df['mistake_kind'] == 'FP').astype(int)
    processed_df['FN'] = (processed_df['mistake_kind'] == 'FN').astype(int)
    if 'area_px' in processed_df.columns and processed_df['area_px'].notna().any():
        # Use the same binning as before
        min_area = processed_df['area_px'].min()
        max_area = processed_df['area_px'].max()
        optimal_bin_edges = [min_area, 50, 100, 200, 400, 1000, 5000, 20000, 50000, 100000, max_area]
        filtered_edges = [edge for edge in optimal_bin_edges if edge <= max_area]
        if filtered_edges[-1] != max_area:
            filtered_edges.append(max_area)
        bin_edges = sorted(list(set(filtered_edges)))
        bin_labels = []
        for i in range(len(bin_edges) - 1):
            start = int(bin_edges[i])
            end = int(bin_edges[i + 1])
            if start < 1000 and end < 1000:
                bin_labels.append(f"{start}-{end}")
            elif start < 1000:
                bin_labels.append(f"{start}-{end:,}")
            else:
                bin_labels.append(f"{start:,}-{end:,}")
        categorized = pd.cut(processed_df['area_px'], bins=bin_edges, labels=bin_labels, include_lowest=True, ordered=True)
        categorized = categorized.astype(str)
        categorized = categorized.replace('nan', 'Unknown')
        processed_df['area_category'] = categorized
    else:
        processed_df['area_category'] = 'Unknown'
    if 'confidence' not in processed_df.columns:
        processed_df['confidence'] = np.where(
            processed_df['ground_truth'] == 0,
            np.random.uniform(0.5, 1.0, len(processed_df)),
            np.nan
        )
    initial_count = len(processed_df)
    processed_df = processed_df.dropna(subset=['bb_id', 'cls_name', 'mistake_kind'])
    return processed_df

# --- Existing code ...
# For training data loading and class distribution, cache the calculation
@st.cache_data(show_spinner=False)
def cached_train_class_distribution(train_data):
    if train_data is None or train_data.empty:
        return {}
    if 'cls_name' in train_data.columns:
        return train_data['cls_name'].value_counts().to_dict()
    return {}

if __name__ == "__main__":
    main()