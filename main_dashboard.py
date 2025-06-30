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
warnings.filterwarnings('ignore')

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

class MultiExperimentAnalyzer:
    def __init__(self):
        """Initialize the multi-experiment analyzer"""
        self.experiments = {}  # Dict to store experiment data
        self.train_data = None
        self.main_experiment = None
        
    def add_experiment(self, name: str, data_file, experiment_id: str = None):
        """Add an experiment to the analyzer"""
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
    
    def set_train_data(self, train_file):
        """Set training data from Excel/CSV file"""
        try:
            # Read training data using the same safe file reading method
            train_data = self._safe_read_file(train_file)
            
            if train_data is not None and not train_data.empty:
                self.train_data = train_data
                # Process train data to get class distribution
                self.train_class_distribution = self._calculate_train_distribution()
                return True
            else:
                st.error("❌ No training data found in file or file is empty")
                return False
        except Exception as e:
            st.error(f"Error loading train data: {str(e)}")
            return False
    
    def _safe_read_file(self, file_path):
        """Safely read file with multiple encoding attempts"""
        if hasattr(file_path, 'name'):
            filename = file_path.name.lower()
            
            if filename.endswith(('.xlsx', '.xls')):
                try:
                    return pd.read_excel(file_path)
                except ImportError as e:
                    if 'openpyxl' in str(e):
                        st.error("📦 **Missing Dependency**: openpyxl is required to read Excel files.")
                        st.info("🔧 **Quick Fix**: Run this command in your terminal:")
                        st.code("pip install openpyxl", language="bash")
                        st.warning("⚠️ **Alternative**: Save your Excel file as CSV and upload that instead.")
                        return None
                    else:
                        st.error(f"Import error reading Excel file: {str(e)}")
                        return None
                except Exception as e:
                    st.error(f"Error reading Excel file: {str(e)}")
                    st.info("💡 **Tip**: Try saving your Excel file as CSV format and uploading that instead.")
                    return None
            
            # Try CSV with different encodings
            encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            
            for encoding in encodings_to_try:
                try:
                    file_path.seek(0)
                    for sep in ['\t', ',', ';']:
                        try:
                            file_path.seek(0)
                            df = pd.read_csv(file_path, encoding=encoding, sep=sep)
                            if len(df.columns) > 5:
                                st.success(f"✅ CSV loaded successfully (encoding: {encoding}, separator: '{sep}')")
                                return df
                        except Exception as csv_error:
                            continue
                except Exception as encoding_error:
                    continue
            
            st.error("❌ Could not read file. Please check:")
            st.info("• File format is supported (.xlsx, .xls, .csv)")
            st.info("• Excel files: Install openpyxl with `pip install openpyxl`")
            st.info("• CSV files: Check encoding (try saving as UTF-8)")
            return None
        
        return pd.read_excel(file_path) if isinstance(file_path, str) else None
    
    def process_experiment_data(self, df: pd.DataFrame, exp_id: str) -> pd.DataFrame:
        """Process experiment data to add derived columns"""
        try:
            processed_df = df.copy()
            processed_df['experiment_id'] = exp_id
            
            # Convert numeric columns with better error handling
            numeric_cols = ['area_px', 'frame', 'cls', 'ground_truth']
            for col in numeric_cols:
                if col in processed_df.columns:
                    processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
                else:
                    # Set default values for missing columns
                    if col == 'ground_truth':
                        processed_df[col] = 0  # Assume predictions if not specified
                    elif col == 'area_px':
                        processed_df[col] = 1000  # Default area
                    elif col == 'cls':
                        processed_df[col] = 0  # Default class ID
            
            # Ensure string columns
            string_cols = ['cls_name', 'bb_size', 'platform', 'mistake_kind', 'class_mistake']
            for col in string_cols:
                if col in processed_df.columns:
                    processed_df[col] = processed_df[col].astype(str)
                    processed_df[col] = processed_df[col].replace('nan', '-')
                else:
                    # Set default values for missing columns
                    if col == 'bb_size':
                        processed_df[col] = 'medium'
                    elif col == 'platform':
                        processed_df[col] = 'unknown'
                    elif col == 'class_mistake':
                        processed_df[col] = '-'
            
            # Create binary indicators
            processed_df['TP'] = (processed_df['mistake_kind'] == 'TP').astype(int)
            processed_df['FP'] = (processed_df['mistake_kind'] == 'FP').astype(int)
            processed_df['FN'] = (processed_df['mistake_kind'] == 'FN').astype(int)
            
            # Calculate area categories (15 ranges) with better error handling
            if 'area_px' in processed_df.columns and processed_df['area_px'].notna().any():
                processed_df['area_category'] = self._categorize_area(processed_df['area_px'])
            else:
                processed_df['area_category'] = 'Unknown'
            
            # Add confidence scores (placeholder if not available)
            if 'confidence' not in processed_df.columns:
                processed_df['confidence'] = np.where(
                    processed_df['ground_truth'] == 0,
                    np.random.uniform(0.5, 1.0, len(processed_df)),
                    np.nan
                )
            
            # Drop rows with critical missing data
            initial_count = len(processed_df)
            processed_df = processed_df.dropna(subset=['bb_id', 'cls_name', 'mistake_kind'])
            final_count = len(processed_df)
            
            if initial_count != final_count:
                st.warning(f"⚠️ Dropped {initial_count - final_count} rows with missing critical data")
            
            if len(processed_df) == 0:
                st.error("❌ No valid data remaining after processing")
                return None
            
            st.success(f"✅ Processed {len(processed_df)} records successfully")
            return processed_df
            
        except Exception as e:
            st.error(f"❌ Error processing data: {str(e)}")
            return None
    
    def _categorize_area(self, areas: pd.Series) -> pd.Series:
        """Categorize areas into 15 ranges"""
        # Remove NaN values for calculation
        valid_areas = areas.dropna()
        if len(valid_areas) == 0:
            return pd.Series(['Unknown'] * len(areas), index=areas.index)
        
        # Create 15 equal-width bins
        min_area = valid_areas.min()
        max_area = valid_areas.max()
        
        # Handle case where all areas are the same
        if min_area == max_area:
            return pd.Series([f"Area_{int(min_area)}"] * len(areas), index=areas.index)
        
        # Create bin edges
        bin_edges = np.linspace(min_area, max_area, 16)
        bin_labels = [f"{int(bin_edges[i])}-{int(bin_edges[i+1])}" for i in range(15)]
        
        # Add 'Unknown' to the categories
        bin_labels.append('Unknown')
        
        # Categorize
        categorized = pd.cut(areas, bins=bin_edges, labels=bin_labels[:-1], include_lowest=True)
        
        # Convert to string to avoid categorical issues, then handle NaN
        categorized = categorized.astype(str)
        categorized = categorized.replace('nan', 'Unknown')
        
        return categorized
    
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
                                   filters: Dict = None) -> pd.DataFrame:
        """Create confusion matrix data similar to the DAX code"""
        if experiment_ids is None:
            experiment_ids = list(self.experiments.keys())
        
        confusion_data = []
        
        for exp_id in experiment_ids:
            if exp_id not in self.experiments:
                continue
                
            exp_data = self.experiments[exp_id]['data'].copy()
            
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
                         analysis_mode: str = "All Data",
                         filters: Dict = None) -> Dict:
        """Calculate metrics for experiments with proper background filtering"""
        if experiment_ids is None:
            experiment_ids = list(self.experiments.keys())
        
        metrics = {}
        
        for exp_id in experiment_ids:
            if exp_id not in self.experiments:
                continue
            
            exp_data = self.experiments[exp_id]['data'].copy()
            
            # Apply filters if provided
            if filters:
                exp_data = self._apply_filters(exp_data, filters)
            
            if len(exp_data) == 0:
                continue
            
            # Apply analysis mode filtering
            if analysis_mode == "Background Only":
                # Only FP and FN with background
                exp_data = exp_data[
                    ((exp_data['mistake_kind'] == 'FP') & (exp_data['class_mistake'] == 'background')) |
                    ((exp_data['mistake_kind'] == 'FN') & (exp_data['class_mistake'] == 'background'))
                ]
            elif analysis_mode == "Non-Background Only":
                # Exclude any background-related mistakes
                exp_data = exp_data[
                    ~((exp_data['mistake_kind'].isin(['FP', 'FN'])) & (exp_data['class_mistake'] == 'background'))
                ]
            # For "All Data", no additional filtering
            
            if len(exp_data) == 0:
                continue
            
            # Calculate overall metrics from filtered data
            overall_tp = len(exp_data[exp_data['mistake_kind'] == 'TP'])
            overall_fp = len(exp_data[exp_data['mistake_kind'] == 'FP'])
            overall_fn = len(exp_data[exp_data['mistake_kind'] == 'FN'])
            
            precision = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0
            recall = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0
            
            metrics[exp_id] = {
                'precision': precision,
                'recall': recall,
                'f1': 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0,
                'tp_count': overall_tp,
                'fp_count': overall_fp,
                'fn_count': overall_fn,
                'filtered_data': exp_data  # Store for per-platform analysis
            }
        
        return metrics

def create_experiment_manager():
    """Create the experiment management interface"""
    st.sidebar.header("🔬 Experiment Management")
    
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
    
    # File uploads
    st.sidebar.subheader("📁 Upload Experiment Data")
    
    # Multiple experiment files
    exp_files = st.sidebar.file_uploader(
        "Upload Experiment Files", 
        type=['xlsx', 'xls', 'csv', 'tsv', 'txt'],
        accept_multiple_files=True,
        help="Upload Excel (.xlsx, .xls) or CSV files. If Excel doesn't work, save as CSV format."
    )
    
    # Process uploaded files
    if exp_files:
        for i, file in enumerate(exp_files):
            exp_name = st.sidebar.text_input(
                f"Name for {file.name}", 
                value=f"Experiment_{i+1}",
                key=f"exp_name_{i}"
            )
            
            if st.sidebar.button(f"Add {exp_name}", key=f"add_exp_{i}"):
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
        success = st.session_state.analyzer.set_train_data(train_file)
        if success:
            st.sidebar.success("✅ Training data loaded")
    
    # Show loaded experiments
    if experiment_names:
        st.sidebar.subheader("📊 Loaded Experiments")
        for exp_name in experiment_names:
            exp_info = st.session_state.analyzer.experiments[exp_name]
            st.sidebar.write(f"• **{exp_info['name']}**: {len(exp_info['data'])} records")
    
    return st.session_state.analyzer

def create_page_filters(analyzer, page_name="Analysis"):
    """Create filters at the top right of each page"""
    if not analyzer.experiments:
        return {}
    
    st.markdown(f"## {page_name}")
    
    # Get all unique values across experiments
    all_data = pd.concat([exp['data'] for exp in analyzer.experiments.values()], ignore_index=True)
    
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
            class_options = ['All Classes'] + classes
            selected_classes = st.multiselect(
                "🚢 Ship Classes",
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
    
    # Get all unique values across experiments
    all_data = pd.concat([exp['data'] for exp in analyzer.experiments.values()], ignore_index=True)
    
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

def confusion_matrix_page(analyzer):
    """Create confusion matrix analysis page"""
    # Create page-specific filters (no experiments, no classes)
    filters = create_confusion_matrix_filters(analyzer, "🔄 Confusion Matrix Analysis")
    
    if not analyzer.experiments:
        st.warning("Please upload experiment data first.")
        return
    
    # Create confusion matrix data with applied filters
    confusion_data = analyzer.create_confusion_matrix_data(
        experiment_ids=filters.get('experiments'),
        filters=filters
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
        fig = px.imshow(
            cm_pivot.values,
            x=cm_pivot.columns,
            y=cm_pivot.index,
            color_continuous_scale='Blues',
            title=f'Confusion Matrix - {selected_exp} (Count & Percentage) - Filtered Data',
            text_auto=False
        )
        
        # Add custom text annotations
        fig.update_traces(text=text_annotations, texttemplate="%{text}")
        
        fig.update_layout(
            xaxis_title="Predicted Class",
            yaxis_title="Actual Class",
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Interactive class analysis (MOVED HERE - between matrix and graphs)
        st.subheader("🔍 Interactive Class Analysis")
        
        all_classes = list(set(exp_confusion['predicted_class'].unique()) | 
                          set(exp_confusion['actual_class'].unique()))
        
        selected_class = st.selectbox(
            "Select Class for Detailed Analysis",
            options=['All Classes'] + sorted(all_classes),
            help="Select a class to see what it was confused with"
        )
        
        # Interactive class distribution analysis
        st.subheader("📊 Class Distribution Analysis")
        
        if selected_class != 'All Classes':
            st.info(f"🎯 **Showing confusion patterns for: {selected_class}** (Filtered Data)")
            
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
                        labels={'x': 'Predicted Class', 'y': 'Count'},
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
                    st.write(f"No data for actual class '{selected_class}' in filtered data")
            
            with col2:
                if len(col_data) > 0:
                    # What was actual when predicted was selected_class
                    actual_when_pred = col_data['actual_class'].value_counts()
                    fig_actual = px.bar(
                        x=actual_when_pred.index,
                        y=actual_when_pred.values,
                        title=f"What was ACTUAL when predicted '{selected_class}'",
                        labels={'x': 'Actual Class', 'y': 'Count'},
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
                    st.write(f"No data for predicted class '{selected_class}' in filtered data")
        
        else:
            # Show overall distributions when "All Classes" is selected
            st.info("🎯 **Showing overall distributions for all classes** (Filtered Data)")
            col1, col2 = st.columns(2)
            
            with col1:
                # Predicted class distribution
                pred_dist = exp_confusion['predicted_class'].value_counts()
                fig_pred = px.bar(
                    x=pred_dist.index,
                    y=pred_dist.values,
                    title="Overall Predicted Class Distribution",
                    labels={'x': 'Predicted Class', 'y': 'Count'}
                )
                fig_pred.update_xaxes(tickangle=45)
                st.plotly_chart(fig_pred, use_container_width=True)
            
            with col2:
                # Actual class distribution
                actual_dist = exp_confusion['actual_class'].value_counts()
                fig_actual = px.bar(
                    x=actual_dist.index,
                    y=actual_dist.values,
                    title="Overall Actual Class Distribution",
                    labels={'x': 'Actual Class', 'y': 'Count'}
                )
                fig_actual.update_xaxes(tickangle=45)
                st.plotly_chart(fig_actual, use_container_width=True)
        
        # Error type distribution and Platform Analysis (COMBINED - side by side)
        st.subheader("📊 Error Distribution & Platform Analysis")
        
        # Get filtered data for all experiments
        all_platform_data = []
        for exp_id in filters.get('experiments', []):
            if exp_id in analyzer.experiments:
                exp_data = analyzer.experiments[exp_id]['data'].copy()
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
                            title="Overall Error Type Distribution (Filtered Data)",
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
                            title="TP/FP/FN Distribution by Platform (Filtered)",
                            color_discrete_map={'TP': 'green', 'FP': 'red', 'FN': 'orange'}
                        )
                        
                        fig_platform.update_xaxes(tickangle=45)
                        st.plotly_chart(fig_platform, use_container_width=True)
                    else:
                        st.info("No platform data available")
                else:
                    st.info("No TP/FP/FN data available")

def overall_metrics_page(analyzer):
    """Overall performance metrics page with different data subsets"""
    # Create page-specific filters
    filters = create_page_filters(analyzer, "📊 Overall Performance Metrics")
    
    if not analyzer.experiments:
        st.warning("Please upload experiment data first.")
        return
    
    # Three analysis modes with corrected descriptions
    analysis_modes = {
        "All Data": "Analyze complete dataset",
        "Background Only": "Analyze only FP and FN with background class",
        "Non-Background Only": "Exclude all background-related FP and FN"
    }
    
    selected_mode = st.selectbox("Select Analysis Mode", list(analysis_modes.keys()))
    st.info(f"**{selected_mode}**: {analysis_modes[selected_mode]} - Using Filtered Data")
    
    # Calculate metrics for all experiments with applied filters
    experiment_ids = filters.get('experiments', [])
    
    if not experiment_ids:
        st.warning("No experiments selected.")
        return
    
    # Use the updated calculate_metrics method with filters
    metrics = analyzer.calculate_metrics(
        experiment_ids=experiment_ids,
        analysis_mode=selected_mode,
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
                    st.metric(
                        "Precision",
                        f"{exp_metrics['precision']:.3f}"
                    )
                    st.metric(
                        "Recall",
                        f"{exp_metrics['recall']:.3f}"
                    )
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
        title=f'Performance Comparison - {selected_mode} (Filtered Data)',
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
    
    # Per-class F1 Score breakdown
    st.subheader("📋 Per-Class F1 Score")
    
    # Calculate F1 scores per class using the filtered data from metrics
    class_f1_data = []
    for exp_id, exp_metrics in metrics.items():
        filtered_data = exp_metrics['filtered_data']
        
        # Calculate F1 per class
        for class_name in filtered_data['cls_name'].unique():
            class_data = filtered_data[filtered_data['cls_name'] == class_name]
            
            tp = len(class_data[class_data['mistake_kind'] == 'TP'])
            fp = len(class_data[class_data['mistake_kind'] == 'FP'])
            fn = len(class_data[class_data['mistake_kind'] == 'FN'])
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            class_f1_data.append({
                'Experiment': exp_id,
                'Class': class_name,
                'F1': f1
            })
    
    if class_f1_data:
        class_f1_df = pd.DataFrame(class_f1_data)
        
        fig_class = px.bar(
            class_f1_df,
            x='Class',
            y='F1',
            color='Experiment',
            barmode='group',
            title=f'F1 Score by Class - {selected_mode} (Filtered Data)'
        )
        
        st.plotly_chart(fig_class, use_container_width=True)
    
    # Per-Platform F1 Score breakdown
    st.subheader("🎥 Per-Platform F1 Score")
    
    # Calculate F1 scores per platform using the filtered data from metrics
    platform_f1_data = []
    for exp_id, exp_metrics in metrics.items():
        filtered_data = exp_metrics['filtered_data']
        
        # Calculate F1 per platform
        for platform_name in filtered_data['platform'].unique():
            platform_data = filtered_data[filtered_data['platform'] == platform_name]
            
            tp = len(platform_data[platform_data['mistake_kind'] == 'TP'])
            fp = len(platform_data[platform_data['mistake_kind'] == 'FP'])
            fn = len(platform_data[platform_data['mistake_kind'] == 'FN'])
            
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
            title=f'F1 Score by Platform - {selected_mode} (Filtered Data)'
        )
        
        st.plotly_chart(fig_platform, use_container_width=True)
        
        # Show platform details in expandable section
        with st.expander("📊 Platform Performance Details"):
            st.dataframe(platform_f1_df.round(3), use_container_width=True)

def fp_analysis_page(analyzer):
    """Detailed False Positive analysis page"""
    # Create page-specific filters
    filters = create_page_filters(analyzer, "🔴 False Positive Analysis")
    
    if not analyzer.experiments:
        st.warning("Please upload experiment data first.")
        return
    
    # Get all filtered data for comprehensive analysis
    experiment_ids = filters.get('experiments', [])
    all_filtered_data = []
    all_fp_data = []
    all_gt_data = []  # For experiment share calculation
    
    for exp_id in experiment_ids:
        if exp_id in analyzer.experiments:
            exp_data = analyzer.experiments[exp_id]['data'].copy()
            # Apply filters to the entire dataset first
            filtered_exp_data = analyzer._apply_filters(exp_data, filters)
            
            # Get FP data from filtered dataset
            fp_data = filtered_exp_data[filtered_exp_data['mistake_kind'] == 'FP'].copy()
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
        st.warning("No False Positive data available for selected filters.")
        return
    
    combined_fp = pd.concat([fp for fp in all_fp_data if len(fp) > 0], ignore_index=True)
    combined_filtered = pd.concat([data for data in all_filtered_data if len(data) > 0], ignore_index=True)
    combined_gt = pd.concat([gt for gt in all_gt_data if len(gt) > 0], ignore_index=True) if any(len(gt) > 0 for gt in all_gt_data) else pd.DataFrame()
    
    if combined_fp.empty:
        st.warning("No False Positive data matches the selected filters.")
        return
    
    # FP Rate and Precision Cards (side by side)
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
                    exp_tp = len(exp_filtered[exp_filtered['mistake_kind'] == 'TP'])
                    exp_fp = len(exp_filtered[exp_filtered['mistake_kind'] == 'FP'])
                    exp_total_pred = len(exp_filtered[exp_filtered['ground_truth'] == 0])
                    
                    fp_rate = exp_fp / exp_total_pred if exp_total_pred > 0 else 0
                    precision = exp_tp / (exp_tp + exp_fp) if (exp_tp + exp_fp) > 0 else 0
                    
                    with cols[col_idx]:
                        st.metric(f"🔴 FP Rate - {exp_id}", f"{fp_rate:.3f}")
                        st.metric(f"🎯 Precision - {exp_id}", f"{precision:.3f}")
    
    # FP Share vs Class Share (Fixed calculation)
    st.subheader("📈 FP Share vs Class Share Analysis")
    
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
        
        fig.update_layout(title="FP Share vs Class Share (Prior Probability) - Filtered Data")
        fig.update_xaxes(title_text="Ship Class")
        fig.update_yaxes(title_text="FP Share (%)", secondary_y=False)
        fig.update_yaxes(title_text="Class Share - Prior Probability (%)", secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add explanation
        with st.expander("📚 Understanding FP Share vs Class Share"):
            st.markdown("""
            **🎯 What this chart shows:**
            
            - **FP Share (Bars)**: What percentage of total False Positives does each class represent?
            - **Class Share (Line)**: What percentage of ground truth does each class represent? (Prior Probability)
            
            **📊 How to interpret:**
            
            - **FP Share > Class Share**: Class gets MORE false positives than expected based on its ground truth frequency
            - **FP Share < Class Share**: Class gets FEWER false positives than expected  
            - **FP Share ≈ Class Share**: Class gets false positives proportional to its ground truth frequency
            
            **💡 Example:**
            - If "Bulk" ships are 40% of ground truth but 60% of false positives → Model struggles with Bulk ships
            - If "Tanker" ships are 30% of ground truth but 10% of false positives → Model performs well on Tankers
            """)
    
    
    # Train Share (if available)
    if analyzer.train_data is not None and not analyzer.train_data.empty:
        st.subheader("📚 Training Data Distribution")
        train_dist = analyzer.train_class_distribution
        if train_dist:
            # Create bar chart for training distribution
            fig_train = px.bar(
                x=list(train_dist.keys()),
                y=list(train_dist.values()),
                title="Training Data Class Distribution",
                labels={'x': 'Ship Class', 'y': 'Count'},
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
    st.subheader("📊 FP Distribution by Experiment")
    
    # Create columns for side-by-side experiment comparison
    if len(experiment_ids) > 1:
        cols = st.columns(len(experiment_ids))
        
        for i, exp_id in enumerate(experiment_ids):
            # Get FP data for this specific experiment
            exp_fp_data = combined_fp[combined_fp['experiment_id'] == exp_id]
            
            if len(exp_fp_data) > 0:
                with cols[i]:
                    fp_dist = exp_fp_data['cls_name'].value_counts()
                    fig_dist = px.bar(
                        x=fp_dist.index,
                        y=fp_dist.values,
                        title=f"FP Distribution - {exp_id}",
                        labels={'x': 'Ship Class', 'y': 'FP Count'}
                    )
                    fig_dist.update_xaxes(tickangle=45)
                    fig_dist.update_layout(height=400)
                    st.plotly_chart(fig_dist, use_container_width=True)
                    
                    # Show total count
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
                title=f"FP Distribution - {exp_id} (Filtered Data)",
                labels={'x': 'Ship Class', 'y': 'FP Count'}
            )
            fig_dist.update_xaxes(tickangle=45)
            st.plotly_chart(fig_dist, use_container_width=True)
    
    # FP Area Distribution - Separate by Experiment
    st.subheader("📐 FP Area Distribution by Experiment")
    
    if 'area_category' in combined_fp.columns:
        if len(experiment_ids) > 1:
            cols = st.columns(len(experiment_ids))
            
            for i, exp_id in enumerate(experiment_ids):
                exp_fp_data = combined_fp[combined_fp['experiment_id'] == exp_id]
                
                if len(exp_fp_data) > 0:
                    with cols[i]:
                        fig_area = px.histogram(
                            exp_fp_data,
                            x='area_category',
                            title=f"FP Area Distribution - {exp_id}",
                            labels={'area_category': 'Area Category', 'count': 'FP Count'}
                        )
                        fig_area.update_xaxes(tickangle=45)
                        fig_area.update_layout(height=400)
                        st.plotly_chart(fig_area, use_container_width=True)
                else:
                    with cols[i]:
                        st.warning(f"No FP area data for {exp_id}")
        else:
            # Single experiment - full width
            exp_id = experiment_ids[0]
            exp_fp_data = combined_fp[combined_fp['experiment_id'] == exp_id]
            
            if len(exp_fp_data) > 0:
                fig_area = px.histogram(
                    exp_fp_data,
                    x='area_category',
                    title=f"FP Area Distribution - {exp_id} (Filtered Data)",
                    labels={'area_category': 'Area Category', 'count': 'FP Count'}
                )
                fig_area.update_xaxes(tickangle=45)
                st.plotly_chart(fig_area, use_container_width=True)
    
    # FP Heatmap - Separate by Experiment
    st.subheader("🔥 FP Heatmap (Platform vs Class) by Experiment")
    
    if len(combined_fp) > 0:
        if len(experiment_ids) > 1:
            cols = st.columns(len(experiment_ids))
            
            for i, exp_id in enumerate(experiment_ids):
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
            # Single experiment - full width
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
                        title=f"FP Heatmap - {exp_id} (Filtered Data)",
                        text_auto=True
                    )
                    st.plotly_chart(fig_heatmap, use_container_width=True)

def fn_analysis_page(analyzer):
    """Detailed False Negative analysis page (similar to FP but for FN)"""
    # Create page-specific filters
    filters = create_page_filters(analyzer, "🔵 False Negative Analysis")
    
    if not analyzer.experiments:
        st.warning("Please upload experiment data first.")
        return
    
    # Get all filtered data for comprehensive analysis
    experiment_ids = filters.get('experiments', [])
    all_filtered_data = []
    all_fn_data = []
    all_gt_data = []  # For class share calculation
    
    for exp_id in experiment_ids:
        if exp_id in analyzer.experiments:
            exp_data = analyzer.experiments[exp_id]['data'].copy()
            # Apply filters to the entire dataset first
            filtered_exp_data = analyzer._apply_filters(exp_data, filters)
            
            # Get FN data from filtered dataset
            fn_data = filtered_exp_data[filtered_exp_data['mistake_kind'] == 'FN'].copy()
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
        st.warning("No False Negative data available for selected filters.")
        return
    
    combined_fn = pd.concat([fn for fn in all_fn_data if len(fn) > 0], ignore_index=True)
    combined_filtered = pd.concat([data for data in all_filtered_data if len(data) > 0], ignore_index=True)
    combined_gt = pd.concat([gt for gt in all_gt_data if len(gt) > 0], ignore_index=True) if any(len(gt) > 0 for gt in all_gt_data) else pd.DataFrame()
    
    if combined_fn.empty:
        st.warning("No False Negative data matches the selected filters.")
        return
    
    # FN Rate and Recall Cards (side by side)
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
                    exp_tp = len(exp_filtered[exp_filtered['mistake_kind'] == 'TP'])
                    exp_fn = len(exp_filtered[exp_filtered['mistake_kind'] == 'FN'])
                    
                    fn_rate = exp_fn / (exp_fn + exp_tp) if (exp_fn + exp_tp) > 0 else 0
                    recall = exp_tp / (exp_tp + exp_fn) if (exp_tp + exp_fn) > 0 else 0
                    
                    with cols[col_idx]:
                        st.metric(f"🔵 FN Rate - {exp_id}", f"{fn_rate:.3f}")
                        st.metric(f"🎯 Recall - {exp_id}", f"{recall:.3f}")
    
    # FN Share vs Class Share (Similar to FP page)
    st.subheader("📈 FN Share vs Class Share Analysis")
    
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
        
        fig.update_layout(title="FN Share vs Class Share (Prior Probability) - Filtered Data")
        fig.update_xaxes(title_text="Ship Class")
        fig.update_yaxes(title_text="FN Share (%)", secondary_y=False)
        fig.update_yaxes(title_text="Class Share - Prior Probability (%)", secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add explanation
        with st.expander("📚 Understanding FN Share vs Class Share"):
            st.markdown("""
            **🎯 What this chart shows:**
            
            - **FN Share (Bars)**: What percentage of total False Negatives does each class represent?
            - **Class Share (Line)**: What percentage of ground truth does each class represent? (Prior Probability)
            
            **📊 How to interpret:**
            
            - **FN Share > Class Share**: Class gets MORE false negatives than expected based on its ground truth frequency
            - **FN Share < Class Share**: Class gets FEWER false negatives than expected  
            - **FN Share ≈ Class Share**: Class gets false negatives proportional to its ground truth frequency
            
            **💡 Example:**
            - If "Bulk" ships are 40% of ground truth but 60% of false negatives → Model misses Bulk ships more often
            - If "Tanker" ships are 30% of ground truth but 10% of false negatives → Model rarely misses Tankers
            """)
    
    # Train Share (if available)
    if analyzer.train_data is not None and not analyzer.train_data.empty:
        st.subheader("📚 Training Data Distribution")
        train_dist = analyzer.train_class_distribution
        if train_dist:
            # Create bar chart for training distribution
            fig_train = px.bar(
                x=list(train_dist.keys()),
                y=list(train_dist.values()),
                title="Training Data Class Distribution",
                labels={'x': 'Ship Class', 'y': 'Count'},
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
    st.subheader("📊 FN Distribution by Experiment")
    
    # Create columns for side-by-side experiment comparison
    if len(experiment_ids) > 1:
        cols = st.columns(len(experiment_ids))
        
        for i, exp_id in enumerate(experiment_ids):
            # Get FN data for this specific experiment
            exp_fn_data = combined_fn[combined_fn['experiment_id'] == exp_id]
            
            if len(exp_fn_data) > 0:
                with cols[i]:
                    fn_dist = exp_fn_data['cls_name'].value_counts()
                    fig_dist = px.bar(
                        x=fn_dist.index,
                        y=fn_dist.values,
                        title=f"FN Distribution - {exp_id}",
                        labels={'x': 'Ship Class', 'y': 'FN Count'}
                    )
                    fig_dist.update_xaxes(tickangle=45)
                    fig_dist.update_layout(height=400)
                    st.plotly_chart(fig_dist, use_container_width=True)
                    
                    # Show total count
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
                title=f"FN Distribution - {exp_id} (Filtered Data)",
                labels={'x': 'Ship Class', 'y': 'FN Count'}
            )
            fig_dist.update_xaxes(tickangle=45)
            st.plotly_chart(fig_dist, use_container_width=True)
    
    # FN Area Distribution - Separate by Experiment
    st.subheader("📐 FN Area Distribution by Experiment")
    
    if 'area_category' in combined_fn.columns:
        if len(experiment_ids) > 1:
            cols = st.columns(len(experiment_ids))
            
            for i, exp_id in enumerate(experiment_ids):
                exp_fn_data = combined_fn[combined_fn['experiment_id'] == exp_id]
                
                if len(exp_fn_data) > 0:
                    with cols[i]:
                        fig_area = px.histogram(
                            exp_fn_data,
                            x='area_category',
                            title=f"FN Area Distribution - {exp_id}",
                            labels={'area_category': 'Area Category', 'count': 'FN Count'}
                        )
                        fig_area.update_xaxes(tickangle=45)
                        fig_area.update_layout(height=400)
                        st.plotly_chart(fig_area, use_container_width=True)
                else:
                    with cols[i]:
                        st.warning(f"No FN area data for {exp_id}")
        else:
            # Single experiment - full width
            exp_id = experiment_ids[0]
            exp_fn_data = combined_fn[combined_fn['experiment_id'] == exp_id]
            
            if len(exp_fn_data) > 0:
                fig_area = px.histogram(
                    exp_fn_data,
                    x='area_category',
                    title=f"FN Area Distribution - {exp_id} (Filtered Data)",
                    labels={'area_category': 'Area Category', 'count': 'FN Count'}
                )
                fig_area.update_xaxes(tickangle=45)
                st.plotly_chart(fig_area, use_container_width=True)
    
    # FN Heatmap - Separate by Experiment
    st.subheader("🔥 FN Heatmap (Platform vs Class) by Experiment")
    
    if len(combined_fn) > 0:
        if len(experiment_ids) > 1:
            cols = st.columns(len(experiment_ids))
            
            for i, exp_id in enumerate(experiment_ids):
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
            # Single experiment - full width
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
                        title=f"FN Heatmap - {exp_id} (Filtered Data)",
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
    st.markdown("**Multi-Experiment Analysis with Interactive Filtering & Precision/Recall/F1 Metrics**")
    
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
                summary_analysis_page(analyzer)
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
        with st.expander("📚 **How to Use This Dashboard** (Click to expand)", expanded=True):
            st.markdown("""
            ## 🚀 **Quick Start Guide**
            
            ### **Step 1: Prepare Your Data** 📁
            - Each experiment = one Excel/CSV file
            - Required columns: `bb_id`, `frame`, `cls_name`, `mistake_kind`
            - Optional: `platform`, `bb_size`, `area_px`, `ground_truth`, `class_mistake`
            
            ### **Step 2: Upload Files** ⬆️
            1. Use **"Upload Experiment Files"** in the sidebar
            2. Select multiple Excel/CSV files (one per experiment)
            3. Give each experiment a meaningful name
            4. Click **"Add [Experiment Name]"** for each file
            
            ### **Step 3: Analyze** 📊
            - Use the tabs to explore different analyses
            - **NEW**: Start with the **Summary & Comparison** tab for overview analysis
            - Use the filters at the top of each page to focus on specific data
            - Click on charts for interactive drill-down
            
            ### **Step 4: Compare** 🔄
            - Select multiple experiments in page filters
            - Choose a "Main Experiment" for baseline comparison
            - View side-by-side metrics and visualizations
            - **NEW**: Use delta charts to see exact changes between experiments
            
            ---
            
            ## 📊 **New Summary Features**
            
            ### **Performance Gap Analysis** 📈
            - Compare best vs worst performing experiments
            - See percentage differences in key metrics
            - Identify which experiments need improvement
            
            ### **Best/Worst Classes** 🎯
            - Find strongest and weakest classes per experiment
            - See which classes need attention
            - Compare class performance across experiments
            
            ### **Class Improvement Analysis** 📊
            - Track which classes improved between experiments
            - Identify classes that degraded
            - See improvement percentages and trends
            
            ### **Delta/Improvement Charts** 🔄
            - Visualize exact changes between experiments
            - Compare any two experiments directly
            - See per-class performance changes
            - Interactive experiment selection for comparison
            
            ---
            
            ## 📋 **Data Format Example**
            
            Your Excel/CSV should look like this:
            """)
            
            # Show a sample data format
            sample_data = pd.DataFrame({
                'bb_id': [1, 2, 3, 4, 5],
                'frame': [1, 1, 2, 2, 3],
                'cls_name': ['Bulk', 'Merchant', 'Bulk', 'Tanker', 'Merchant'],
                'mistake_kind': ['TP', 'FP', 'TP', 'FN', 'TP'],
                'platform': ['Platform_A', 'Platform_A', 'Platform_B', 'Platform_B', 'Platform_A'],
                'bb_size': ['large', 'medium', 'large', 'small', 'medium'],
                'area_px': [15000, 8500, 16000, 2500, 7800],
                'ground_truth': [0, 0, 0, 1, 0],
                'class_mistake': ['-', 'Bulk', '-', '-', '-']
            })
            
            st.dataframe(sample_data, use_container_width=True)
            
            st.markdown("""
            ## 🎛️ **Understanding the Analysis**
            
            ### **Summary & Comparison Page** 📈 **(NEW!)**
            - **Performance Gap Analysis**: See % differences between best/worst experiments
            - **Best/Worst Classes**: Identify top and bottom performing classes
            - **Class Improvement**: Track which classes improved between experiments
            - **Delta Charts**: Compare any two experiments with interactive selection
            - **Improvement Tracking**: See exact F1-Score changes per class
            
            ### **Confusion Matrix Page** 🔄
            - Interactive confusion matrix with click-to-drill
            - Shows prediction vs actual classifications
            - Per-page filtering with collapsible controls
            
            ### **Overall Metrics Page** 📊
            - **Background Only**: Analyzes only FP and FN with background class
            - **Non-Background Only**: Excludes all background-related FP and FN
            - **All Data**: Complete dataset analysis
            - Precision, Recall, and F1-Score metrics
            - Per-class and per-platform F1 Score analysis
            
            ### **False Positive/Negative Pages** 🔴🔵
            - Deep-dive error analysis
            - Rate calculations and distributions
            - Platform and class-wise breakdowns
            - Interactive heatmaps and visualizations
            - FP/FN Share vs Class Share analysis
            - Side-by-side experiment comparisons
            
            ## 🔧 **Troubleshooting**
            
            **"Missing openpyxl" Error:**
            ```bash
            pip install openpyxl xlrd
            ```
            
            **"Cannot setitem on Categorical" Error:**
            - This is now fixed! Try uploading again.
            
            **"Missing required columns" Error:**
            - Check your file has: bb_id, frame, cls_name, mistake_kind
            - Column names must match exactly (case-sensitive)
            
            **File Won't Load:**
            - Try saving Excel as CSV UTF-8 format
            - Check for empty rows/columns
            - Ensure data is properly formatted
            """)
        
        # Installation requirements
        with st.expander("⚙️ Installation Requirements"):
            st.markdown("""
            **📦 Required Dependencies:**
            
            If you get errors when uploading Excel files, install these packages:
            ```bash
            pip install openpyxl xlrd
            ```
            
            **🔧 Full Installation Command:**
            ```bash
            pip install streamlit pandas numpy plotly scikit-learn openpyxl xlrd
            ```
            
            **💡 Alternative**: If you can't install openpyxl, save your Excel files as CSV format instead.
            """)
        
        # Show expected data format
        with st.expander("📋 Expected Data Format"):
            st.markdown("""
            **Excel/CSV Structure:**
            - `bb_id`: Unique bounding box identifier
            - `frame`: Frame number
            - `ground_truth`: 0 for predictions, 1 for ground truth
            - `platform`: Camera platform location
            - `cls`: Class ID number
            - `cls_name`: Ship class name
            - `bb_size`: Size category (small, medium, large)
            - `area_px`: Bounding box area in pixels
            - `mistake_kind`: TP, FP, FN, or '-'
            - `class_mistake`: Actual class if misclassified, otherwise '-'
              - For FP with background: `mistake_kind='FP'` and `class_mistake='background'`
              - For FN with background: `mistake_kind='FN'` and `class_mistake='background'`
            - `confidence`: Detection confidence (optional)
            
            **Training Excel/CSV Structure:**
            - Should have `cls_name` column for class distribution calculation
            - Can be any Excel/CSV file with class information
            """)
        
        # Feature overview
        with st.expander("🎯 Dashboard Features"):
            st.markdown("""
            **✨ Key Features:**
            
            1. **Multi-Experiment Support**: Upload and compare multiple experiments
            2. **NEW: Summary & Comparison Analysis**: Comprehensive experiment comparison
            3. **NEW: Performance Gap Analysis**: See % differences between experiments
            4. **NEW: Class Improvement Tracking**: Track class performance changes
            5. **NEW: Delta Charts**: Interactive experiment-to-experiment comparison
            6. **Interactive Confusion Matrix**: Click and drill-down analysis
            7. **Per-Page Filtering**: Individual filters on each analysis page
            8. **Precision/Recall/F1 Metrics**: Core performance metrics with filtering support
            9. **Advanced Filtering**: Platform, size, class filters with "All" options
            10. **FP/FN Deep Dive**: Comprehensive error analysis
            11. **Share Analysis**: FP/FN Share vs Class Share (Prior Probability)
            12. **Training Data Integration**: Compare test vs train distributions
            13. **NEW**: Per-Platform F1 Score Analysis
            
            **📊 Metrics Calculated:**
            - Precision, Recall, F1-Score
            - Per-class and per-platform F1 scores
            - Overall metrics with different background filtering modes
            - Error rate analysis
            - Prior probability comparisons
            - **NEW**: Performance gaps and improvement percentages
            - **NEW**: Best/worst class identification
            - **NEW**: Delta calculations between experiments
            
            **🎛️ Filtering System:**
            - **Per-Page Filters**: Each analysis page has its own filter controls
            - **Collapsible Interface**: Filters are in expandable sections to save space
            - **"All" Options**: Easy selection of all categories in each filter
            - **Real-Time Updates**: All visualizations update instantly with filter changes
            """)

if __name__ == "__main__":
    main()