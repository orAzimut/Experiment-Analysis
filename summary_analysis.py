import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

def create_summary_filters(analyzer, page_name="Summary Analysis"):
    """Create filters for summary page"""
    if not analyzer.experiments:
        return {}
    
    st.markdown(f"## {page_name}")
    
    # Get global super class mode
    use_super_class = getattr(st.session_state, 'global_super_class_mode', False)
    
    # Get all unique values across experiments (with super class consideration)
    all_data_list = []
    for exp_id in analyzer.experiments.keys():
        exp_data = analyzer.get_processed_data(exp_id, use_super_class)
        if exp_data is not None:
            all_data_list.append(exp_data)
    
    if not all_data_list:
        return {}
    
    all_data = pd.concat(all_data_list, ignore_index=True)
    
    # Create unique key prefix based on page name
    key_prefix = page_name.replace(" ", "_").replace("(", "").replace(")", "").replace("&", "and")
    
    # Create filter section at the top
    with st.expander("🎛️ **Filters** (Click to expand/collapse)", expanded=False):
        # Create 3 columns for filters
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
        'experiments': list(analyzer.experiments.keys()),
        'platforms': selected_platforms,
        'sizes': selected_sizes,
        'classes': selected_classes
    }

def filter_data_by_analysis_mode(data, analysis_mode):
    """Filter data based on global analysis mode"""
    if analysis_mode == "Detection":
        # Detection analysis: only detection-related data
        return data[
            (data['mistake_kind'] == 'TP') |
            ((data['mistake_kind'] == 'FP') & (data['class_mistake'] == 'background')) |
            ((data['mistake_kind'] == 'FN') & (data['class_mistake'] == 'background'))
        ]
    elif analysis_mode == "Classification":
        # Classification analysis: only classification-related data
        return data[
            (data['mistake_kind'] == 'TP') |
            ((data['mistake_kind'] == 'FP') & (data['class_mistake'] != 'background') & (data['class_mistake'] != '-'))
        ]
    else:  # "All Data"
        return data

def calculate_experiment_metrics(analyzer, filters, analysis_mode, use_super_class):
    """Calculate comprehensive metrics for all experiments with filters and analysis mode"""
    experiment_metrics = {}
    
    for exp_id in filters.get('experiments', []):
        if exp_id not in analyzer.experiments:
            continue
        
        # Get filtered data with super class consideration
        exp_data = analyzer.get_processed_data(exp_id, use_super_class)
        if exp_data is None:
            continue
            
        if filters:
            exp_data = analyzer._apply_filters(exp_data, filters)
        
        # Apply analysis mode filtering
        exp_data = filter_data_by_analysis_mode(exp_data, analysis_mode)
        
        if len(exp_data) == 0:
            continue
        
        # Calculate overall metrics based on analysis mode
        if analysis_mode == "Detection":
            tp_count = len(exp_data[exp_data['mistake_kind'] == 'TP'])
            fp_count = len(exp_data[
                (exp_data['mistake_kind'] == 'FP') & 
                (exp_data['class_mistake'] == 'background')
            ])
            fn_count = len(exp_data[
                (exp_data['mistake_kind'] == 'FN') & 
                (exp_data['class_mistake'] == 'background')
            ])
        elif analysis_mode == "Classification":
            tp_count = len(exp_data[exp_data['mistake_kind'] == 'TP'])
            fp_count = len(exp_data[
                (exp_data['mistake_kind'] == 'FP') & 
                (exp_data['class_mistake'] != 'background') &
                (exp_data['class_mistake'] != '-')
            ])
            fn_count = fp_count  # Mirror relationship in classification
        else:  # "All Data"
            tp_count = len(exp_data[exp_data['mistake_kind'] == 'TP'])
            fp_count = len(exp_data[exp_data['mistake_kind'] == 'FP'])
            fn_count = len(exp_data[exp_data['mistake_kind'] == 'FN'])
        
        precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
        recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calculate per-class metrics
        class_metrics = {}
        for cls in exp_data['cls_name'].unique():
            cls_data = exp_data[exp_data['cls_name'] == cls]
            
            if analysis_mode == "Detection":
                cls_tp = len(cls_data[cls_data['mistake_kind'] == 'TP'])
                cls_fp = len(cls_data[
                    (cls_data['mistake_kind'] == 'FP') & 
                    (cls_data['class_mistake'] == 'background')
                ])
                cls_fn = len(cls_data[
                    (cls_data['mistake_kind'] == 'FN') & 
                    (cls_data['class_mistake'] == 'background')
                ])
            elif analysis_mode == "Classification":
                cls_tp = len(cls_data[cls_data['mistake_kind'] == 'TP'])
                cls_fp = len(cls_data[
                    (cls_data['mistake_kind'] == 'FP') & 
                    (cls_data['class_mistake'] != 'background') &
                    (cls_data['class_mistake'] != '-')
                ])
                cls_fn = cls_fp  # Mirror relationship
            else:  # "All Data"
                cls_tp = len(cls_data[cls_data['mistake_kind'] == 'TP'])
                cls_fp = len(cls_data[cls_data['mistake_kind'] == 'FP'])
                cls_fn = len(cls_data[cls_data['mistake_kind'] == 'FN'])
            
            cls_precision = cls_tp / (cls_tp + cls_fp) if (cls_tp + cls_fp) > 0 else 0
            cls_recall = cls_tp / (cls_tp + cls_fn) if (cls_tp + cls_fn) > 0 else 0
            cls_f1 = 2 * cls_precision * cls_recall / (cls_precision + cls_recall) if (cls_precision + cls_recall) > 0 else 0
            
            class_metrics[cls] = {
                'precision': cls_precision,
                'recall': cls_recall,
                'f1': cls_f1,
                'tp': cls_tp,
                'fp': cls_fp,
                'fn': cls_fn,
                'total_detections': cls_tp + cls_fp,
                'total_ground_truth': cls_tp + cls_fn
            }
        
        experiment_metrics[exp_id] = {
            'overall': {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'tp': tp_count,
                'fp': fp_count,
                'fn': fn_count
            },
            'classes': class_metrics
        }
    
    return experiment_metrics

def calculate_class_distributions(analyzer, exp_id, filters, analysis_mode, use_super_class):
    """Calculate apriori distributions and FP/FN shares for classes with analysis mode"""
    if exp_id not in analyzer.experiments:
        return None
    
    # Get filtered data with super class consideration
    exp_data = analyzer.get_processed_data(exp_id, use_super_class)
    if exp_data is None:
        return None
    
    if filters:
        exp_data = analyzer._apply_filters(exp_data, filters)
    
    # Apply analysis mode filtering
    exp_data = filter_data_by_analysis_mode(exp_data, analysis_mode)
    
    if len(exp_data) == 0:
        return None
    
    # Calculate total ground truth instances per class (TP + FN)
    gt_counts = {}
    for cls in exp_data['cls_name'].unique():
        cls_data = exp_data[exp_data['cls_name'] == cls]
        if analysis_mode == "Detection":
            gt_count = len(cls_data[cls_data['mistake_kind'].isin(['TP'])])
            gt_count += len(cls_data[
                (cls_data['mistake_kind'] == 'FN') & 
                (cls_data['class_mistake'] == 'background')
            ])
        elif analysis_mode == "Classification":
            gt_count = len(cls_data[cls_data['mistake_kind'].isin(['TP'])])
            # For classification, FN is mirrored from FP
            gt_count += len(cls_data[
                (cls_data['mistake_kind'] == 'FP') & 
                (cls_data['class_mistake'] != 'background') &
                (cls_data['class_mistake'] != '-')
            ])
        else:  # "All Data"
            gt_count = len(cls_data[cls_data['mistake_kind'].isin(['TP', 'FN'])])
            
        if gt_count > 0:
            gt_counts[cls] = gt_count
    
    total_gt = sum(gt_counts.values())
    
    # Calculate apriori distribution (ground truth distribution)
    apriori_dist = {cls: count / total_gt for cls, count in gt_counts.items()} if total_gt > 0 else {}
    
    # Calculate FP and FN shares
    fp_counts = {}
    fn_counts = {}
    
    for cls in exp_data['cls_name'].unique():
        cls_data = exp_data[exp_data['cls_name'] == cls]
        
        if analysis_mode == "Detection":
            fp_counts[cls] = len(cls_data[
                (cls_data['mistake_kind'] == 'FP') & 
                (cls_data['class_mistake'] == 'background')
            ])
            fn_counts[cls] = len(cls_data[
                (cls_data['mistake_kind'] == 'FN') & 
                (cls_data['class_mistake'] == 'background')
            ])
        elif analysis_mode == "Classification":
            fp_counts[cls] = len(cls_data[
                (cls_data['mistake_kind'] == 'FP') & 
                (cls_data['class_mistake'] != 'background') &
                (cls_data['class_mistake'] != '-')
            ])
            fn_counts[cls] = fp_counts[cls]  # Mirror relationship
        else:  # "All Data"
            fp_counts[cls] = len(cls_data[cls_data['mistake_kind'] == 'FP'])
            fn_counts[cls] = len(cls_data[cls_data['mistake_kind'] == 'FN'])
    
    total_fp = sum(fp_counts.values())
    total_fn = sum(fn_counts.values())
    
    fp_shares = {cls: count / total_fp for cls, count in fp_counts.items()} if total_fp > 0 else {}
    fn_shares = {cls: count / total_fn for cls, count in fn_counts.items()} if total_fn > 0 else {}
    
    return {
        'apriori_dist': apriori_dist,
        'fp_shares': fp_shares,
        'fn_shares': fn_shares,
        'fp_counts': fp_counts,
        'fn_counts': fn_counts,
        'gt_counts': gt_counts
    }

def most_problematic_classes_analysis(analyzer, filters, analysis_mode, use_super_class):
    """Analyze most problematic classes based on apriori vs FP/FN share with analysis mode"""
    class_level = "Super Class" if use_super_class else "Subclass"
    st.subheader(f"🚨 Most Problematic Classes ({analysis_mode} Mode, {class_level})")
    
    for exp_id in filters.get('experiments', []):
        dist_data = calculate_class_distributions(analyzer, exp_id, filters, analysis_mode, use_super_class)
        if not dist_data:
            continue
        
        st.write(f"### {exp_id}")
        
        # Calculate gaps for each class
        problematic_classes = []
        
        for cls in dist_data['apriori_dist'].keys():
            apriori = dist_data['apriori_dist'].get(cls, 0)
            fp_share = dist_data['fp_shares'].get(cls, 0)
            fn_share = dist_data['fn_shares'].get(cls, 0)
            
            # Calculate gaps
            fp_gap = fp_share - apriori if fp_share > apriori else 0
            fn_gap = fn_share - apriori if fn_share > apriori else 0
            
            if fp_gap > 0 or fn_gap > 0:
                problematic_classes.append({
                    'Class': cls,
                    'Apriori %': f"{apriori * 100:.1f}%",
                    'FP Share %': f"{fp_share * 100:.1f}%",
                    'FN Share %': f"{fn_share * 100:.1f}%",
                    'FP Gap': fp_gap * 100,
                    'FN Gap': fn_gap * 100,
                    'Max Gap': max(fp_gap, fn_gap) * 100,
                    'Problem Type': 'FP' if fp_gap > fn_gap else 'FN',
                    'FP Count': dist_data['fp_counts'].get(cls, 0),
                    'FN Count': dist_data['fn_counts'].get(cls, 0),
                    'GT Count': dist_data['gt_counts'].get(cls, 0)
                })
        
        if problematic_classes:
            # Sort by max gap and get top 5
            problematic_df = pd.DataFrame(problematic_classes)
            problematic_df = problematic_df.sort_values('Max Gap', ascending=False).head(5)
            
            # Create table display
            display_df = problematic_df[['Class', 'Apriori %', 'FP Share %', 'FN Share %', 'Problem Type', 'FP Count', 'FN Count', 'GT Count']]
            
            # Style the dataframe
            def highlight_problem(row):
                styles = [''] * len(row)
                if row['Problem Type'] == 'FP':
                    styles[2] = 'background-color: #ffcccc; font-weight: bold'
                else:
                    styles[3] = 'background-color: #ffcccc; font-weight: bold'
                return styles
            
            styled_df = display_df.style.apply(highlight_problem, axis=1)
            st.dataframe(styled_df, use_container_width=True)
            
            # Drill-down by object size
            with st.expander(f"📏 Drill-down by Object Size - {exp_id}"):
                # Get raw data for size analysis
                raw_exp_data = analyzer.get_processed_data(exp_id, use_super_class)
                if raw_exp_data is not None:
                    if filters:
                        raw_exp_data = analyzer._apply_filters(raw_exp_data, filters)
                    
                    for size in sorted(raw_exp_data['bb_size'].unique()):
                        st.write(f"**{size.capitalize()} Objects:**")
                        
                        # Filter by size and recalculate
                        size_filters = filters.copy()
                        size_filters['sizes'] = [size]
                        size_dist_data = calculate_class_distributions(analyzer, exp_id, size_filters, analysis_mode, use_super_class)
                        
                        if size_dist_data:
                            size_problematic = []
                            
                            for cls in size_dist_data['apriori_dist'].keys():
                                apriori = size_dist_data['apriori_dist'].get(cls, 0)
                                fp_share = size_dist_data['fp_shares'].get(cls, 0)
                                fn_share = size_dist_data['fn_shares'].get(cls, 0)
                                
                                fp_gap = fp_share - apriori if fp_share > apriori else 0
                                fn_gap = fn_share - apriori if fn_share > apriori else 0
                                
                                if fp_gap > 0 or fn_gap > 0:
                                    size_problematic.append({
                                        'Class': cls,
                                        'Gap %': f"{max(fp_gap, fn_gap) * 100:.1f}%",
                                        'Type': 'FP' if fp_gap > fn_gap else 'FN',
                                        'Count': size_dist_data['fp_counts'].get(cls, 0) if fp_gap > fn_gap else size_dist_data['fn_counts'].get(cls, 0)
                                    })
                            
                            if size_problematic:
                                size_df = pd.DataFrame(size_problematic).sort_values('Gap %', ascending=False).head(3)
                                for _, row in size_df.iterrows():
                                    st.write(f"  • {row['Class']}: {row['Gap %']} ({row['Type']}: {row['Count']})")
                            else:
                                st.write("  • No problematic classes for this size")
        else:
            st.info(f"No problematic classes found for {exp_id} in {analysis_mode} mode ({class_level})")

def top_performing_classes_by_gap(analyzer, filters, experiments, analysis_mode, use_super_class):
    """Analyze top performing classes based on gap logic with analysis mode"""
    class_level = "Super Class" if use_super_class else "Subclass"
    st.write(f"### 🏆 Top Performing Classes ({analysis_mode} Mode, {class_level})")
    
    # Create columns for side-by-side display
    cols = st.columns(len(experiments))
    
    for idx, exp_id in enumerate(experiments):
        with cols[idx]:
            st.write(f"**{exp_id}**")
            
            dist_data = calculate_class_distributions(analyzer, exp_id, filters, analysis_mode, use_super_class)
            if not dist_data:
                st.info("No data available")
                continue
            
            # Calculate classes that perform well in BOTH FP and FN
            performing_classes = []
            
            for cls in dist_data['apriori_dist'].keys():
                apriori = dist_data['apriori_dist'].get(cls, 0)
                fp_share = dist_data['fp_shares'].get(cls, 0)
                fn_share = dist_data['fn_shares'].get(cls, 0)
                
                # Only consider as top performing if BOTH FP and FN shares are lower than apriori
                if fp_share < apriori and fn_share < apriori:
                    fp_improvement = (apriori - fp_share) * 100
                    fn_improvement = (apriori - fn_share) * 100
                    avg_improvement = (fp_improvement + fn_improvement) / 2
                    
                    performing_classes.append({
                        'Class': cls,
                        'Apriori %': f"{apriori * 100:.1f}%",
                        'FP Share %': f"{fp_share * 100:.1f}%",
                        'FN Share %': f"{fn_share * 100:.1f}%",
                        'Avg Improvement': avg_improvement,
                        'GT Count': dist_data['gt_counts'].get(cls, 0)
                    })
            
            if performing_classes:
                # Sort by average improvement and get top 5
                performing_df = pd.DataFrame(performing_classes)
                performing_df = performing_df.sort_values('Avg Improvement', ascending=False).head(5)
                display_df = performing_df[['Class', 'Apriori %', 'FP Share %', 'FN Share %', 'GT Count']]
                
                # Style the dataframe
                def highlight_good(val):
                    if 'Share %' in str(val):
                        return 'color: green'
                    return ''
                
                styled_df = display_df.style.applymap(highlight_good)
                st.dataframe(styled_df, use_container_width=True, height=230)
            else:
                st.info(f"No classes performing better than expected in both FP and FN ({analysis_mode} mode, {class_level})")

def top_worst_performing_by_f1(analyzer, filters, experiments, analysis_mode, use_super_class):
    """Show top and worst performing classes by F1 score with analysis mode"""
    class_level = "Super Class" if use_super_class else "Subclass"
    st.write(f"### 📊 Top/Worst Performing Classes by F1 Score ({analysis_mode} Mode, {class_level})")
    
    metrics = calculate_experiment_metrics(analyzer, filters, analysis_mode, use_super_class)
    
    # Create columns for side-by-side display
    cols = st.columns(len(experiments))
    
    for idx, exp_id in enumerate(experiments):
        with cols[idx]:
            st.write(f"**{exp_id}**")
            
            if exp_id not in metrics or not metrics[exp_id]['classes']:
                st.info("No data available")
                continue
            
            exp_metrics = metrics[exp_id]
            
            # Sort classes by F1 score
            class_f1_scores = [(cls, data['f1'], data['tp'], data['fp'], data['fn']) 
                              for cls, data in exp_metrics['classes'].items()]
            class_f1_scores.sort(key=lambda x: x[1], reverse=True)
            
            performance_data = []
            
            # Get top 5
            for i, (cls, f1, tp, fp, fn) in enumerate(class_f1_scores[:5]):
                performance_data.append({
                    'Rank': f'Top {i+1}',
                    'Class': cls,
                    'F1': f"{f1:.3f}",
                    'Prec': f"{tp/(tp+fp):.3f}" if (tp+fp) > 0 else "0.000",
                    'Rec': f"{tp/(tp+fn):.3f}" if (tp+fn) > 0 else "0.000",
                    'TP': tp,
                    'FP': fp,
                    'FN': fn
                })
            
            # Add separator
            if len(class_f1_scores) > 5:
                performance_data.append({
                    'Rank': '---',
                    'Class': '---',
                    'F1': '---',
                    'Prec': '---',
                    'Rec': '---',
                    'TP': '---',
                    'FP': '---',
                    'FN': '---'
                })
            
            # Get bottom 5 with at least 1 TP for meaningful metrics
            worst_classes_with_tp = [(cls, f1, tp, fp, fn) for cls, f1, tp, fp, fn in class_f1_scores if tp >= 1]
            worst_classes_with_tp.sort(key=lambda x: x[1])  # Sort ascending to get worst first
            
            # Get up to 5 worst classes with at least 1 TP
            bottom_classes = worst_classes_with_tp[:5]
            
            for i, (cls, f1, tp, fp, fn) in enumerate(bottom_classes):
                performance_data.append({
                    'Rank': f'Bottom {i+1}',
                    'Class': cls,
                    'F1': f"{f1:.3f}",
                    'Prec': f"{tp/(tp+fp):.3f}" if (tp+fp) > 0 else "0.000",
                    'Rec': f"{tp/(tp+fn):.3f}" if (tp+fn) > 0 else "0.000",
                    'TP': tp,
                    'FP': fp,
                    'FN': fn
                })
            
            if performance_data:
                perf_df = pd.DataFrame(performance_data)
                
                # Style the dataframe
                def style_row(row):
                    if 'Top' in str(row['Rank']):
                        return ['background-color: #e6f3e6'] * len(row)
                    elif 'Bottom' in str(row['Rank']):
                        return ['background-color: #ffe6e6'] * len(row)
                    elif '---' in str(row['Rank']):
                        return ['background-color: #f0f0f0'] * len(row)
                    return [''] * len(row)
                
                styled_df = perf_df.style.apply(style_row, axis=1)
                st.dataframe(styled_df, use_container_width=True, height=450)

def performance_gap_analysis(analyzer, filters, analysis_mode, use_super_class):
    """Analyze performance gaps between experiments with analysis mode"""
    class_level = "Super Class" if use_super_class else "Subclass"
    st.subheader(f"📊 Performance Gap Analysis ({analysis_mode} Mode, {class_level})")
    
    metrics = calculate_experiment_metrics(analyzer, filters, analysis_mode, use_super_class)
    
    if len(metrics) < 2:
        st.warning("Need at least 2 experiments for gap analysis.")
        return
    
    # Create comparison data
    comparison_data = []
    for exp_id, exp_metrics in metrics.items():
        comparison_data.append({
            'Experiment': exp_id,
            'Precision': exp_metrics['overall']['precision'],
            'Recall': exp_metrics['overall']['recall'],
            'F1-Score': exp_metrics['overall']['f1']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Calculate gaps
    gaps_data = []
    for metric in ['Precision', 'Recall', 'F1-Score']:
        best_exp = comparison_df.loc[comparison_df[metric].idxmax(), 'Experiment']
        worst_exp = comparison_df.loc[comparison_df[metric].idxmin(), 'Experiment']
        best_score = comparison_df[metric].max()
        worst_score = comparison_df[metric].min()
        gap_percent = ((best_score - worst_score) / worst_score * 100) if worst_score > 0 else 0
        
        gaps_data.append({
            'Metric': metric,
            'Best Experiment': best_exp,
            'Worst Experiment': worst_exp,
            'Best Score': f"{best_score:.3f}",
            'Worst Score': f"{worst_score:.3f}",
            'Gap %': f"{gap_percent:.1f}%"
        })
    
    gaps_df = pd.DataFrame(gaps_data)
    
    # Display gap analysis table
    st.write("**🏆 Performance Rankings & Gaps**")
    st.dataframe(gaps_df, use_container_width=True)
    
    return gaps_df, comparison_df

def experiment_to_experiment_changes(analyzer, filters, analysis_mode, use_super_class):
    """Show experiment-to-experiment changes with analysis mode"""
    class_level = "Super Class" if use_super_class else "Subclass"
    st.subheader(f"🔄 Experiment-to-Experiment Changes ({analysis_mode} Mode, {class_level})")
    
    metrics = calculate_experiment_metrics(analyzer, filters, analysis_mode, use_super_class)
    experiments = list(metrics.keys())
    
    if len(experiments) < 2:
        st.warning("Need at least 2 experiments for comparison.")
        return
    
    # Create pairwise comparison selector
    col1, col2 = st.columns(2)
    
    with col1:
        baseline_exp = st.selectbox(
            "Select Baseline Experiment",
            experiments,
            key="baseline_exp_summary"
        )
    
    with col2:
        comparison_exp = st.selectbox(
            "Select Comparison Experiment",
            [exp for exp in experiments if exp != baseline_exp],
            key="comparison_exp_summary"
        )
    
    if baseline_exp and comparison_exp:
        selected_experiments = [baseline_exp, comparison_exp]
        
        # Calculate deltas between selected experiments
        baseline_metrics = metrics[baseline_exp]['overall']
        comparison_metrics = metrics[comparison_exp]['overall']
        
        deltas = {
            'precision': comparison_metrics['precision'] - baseline_metrics['precision'],
            'recall': comparison_metrics['recall'] - baseline_metrics['recall'],
            'f1': comparison_metrics['f1'] - baseline_metrics['f1']
        }
        
        # Display delta metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            delta_val = deltas['precision']
            delta_pct = (delta_val / baseline_metrics['precision'] * 100) if baseline_metrics['precision'] > 0 else 0
            st.metric(
                "Precision Change",
                f"{comparison_metrics['precision']:.3f}",
                delta=f"{delta_val:+.3f} ({delta_pct:+.1f}%)"
            )
        
        with col2:
            delta_val = deltas['recall']
            delta_pct = (delta_val / baseline_metrics['recall'] * 100) if baseline_metrics['recall'] > 0 else 0
            st.metric(
                "Recall Change",
                f"{comparison_metrics['recall']:.3f}",
                delta=f"{delta_val:+.3f} ({delta_pct:+.1f}%)"
            )
        
        with col3:
            delta_val = deltas['f1']
            delta_pct = (delta_val / baseline_metrics['f1'] * 100) if baseline_metrics['f1'] > 0 else 0
            st.metric(
                "F1-Score Change",
                f"{comparison_metrics['f1']:.3f}",
                delta=f"{delta_val:+.3f} ({delta_pct:+.1f}%)"
            )
        
        # Per-class deltas table
        st.write(f"### 📋 Per-Class Performance Changes ({class_level})")
        
        # Get common classes
        baseline_classes = set(metrics[baseline_exp]['classes'].keys())
        comparison_classes = set(metrics[comparison_exp]['classes'].keys())
        common_classes = baseline_classes.intersection(comparison_classes)
        
        if common_classes:
            class_deltas = []
            
            for cls in common_classes:
                baseline_f1 = metrics[baseline_exp]['classes'][cls]['f1']
                comparison_f1 = metrics[comparison_exp]['classes'][cls]['f1']
                delta = comparison_f1 - baseline_f1
                delta_pct = (delta / baseline_f1 * 100) if baseline_f1 > 0 else 0
                
                class_deltas.append({
                    'Class': cls,
                    'Baseline F1': f"{baseline_f1:.3f}",
                    'Comparison F1': f"{comparison_f1:.3f}",
                    'Delta': f"{delta:+.3f}",
                    'Delta %': f"{delta_pct:+.1f}%"
                })
            
            class_deltas_df = pd.DataFrame(class_deltas)
            class_deltas_df = class_deltas_df.sort_values('Delta %', ascending=False)
            
            # Style the dataframe
            def style_delta(val):
                if '+' in str(val) and val != '+0.0%' and val != '+0.000':
                    return 'color: green; font-weight: bold'
                elif '-' in str(val):
                    return 'color: red; font-weight: bold'
                return ''
            
            styled_df = class_deltas_df.style.applymap(style_delta, subset=['Delta', 'Delta %'])
            st.dataframe(styled_df, use_container_width=True)
        else:
            st.warning("No common classes found between selected experiments.")
        
        st.markdown("---")
        
        # Add Top Performing Classes by Gap Logic
        top_performing_classes_by_gap(analyzer, filters, selected_experiments, analysis_mode, use_super_class)
        
        st.markdown("---")
        
        # Add Top/Worst Performing Classes by F1 Score
        top_worst_performing_by_f1(analyzer, filters, selected_experiments, analysis_mode, use_super_class)

def summary_analysis_page(analyzer):
    """Main summary analysis page with global analysis mode support"""
    # Create page-specific filters
    filters = create_summary_filters(analyzer, "📈 Summary & Comparison Analysis")
    
    # Get global analysis mode and super class mode
    analysis_mode = getattr(st.session_state, 'global_analysis_mode', 'All Data')
    use_super_class = getattr(st.session_state, 'global_super_class_mode', False)
    
    # Show analysis mode info
    class_level = "Super Class" if use_super_class else "Subclass"
    
    if analysis_mode == "Detection":
        st.info(f"**🔍 Detection Analysis** ({class_level}): Summary focused on detection performance (something vs nothing)")
    elif analysis_mode == "Classification":
        st.info(f"**🏷️ Classification Analysis** ({class_level}): Summary focused on classification performance among detected objects")
    else:
        st.info(f"**📊 All Data Analysis** ({class_level}): Complete summary including all detection and classification errors")
    
    if not analyzer.experiments:
        st.warning("Please upload experiment data first.")
        return
    
    if len(analyzer.experiments) < 2:
        st.info("💡 **Tip**: Upload multiple experiments to see comparison analysis. Currently showing single experiment analysis.")
    
    # Add overview section
    st.markdown("---")
    
    # Run analyses in order
    try:
        # 1. Most Problematic Classes (near headline as requested)
        most_problematic_classes_analysis(analyzer, filters, analysis_mode, use_super_class)
        
        st.markdown("---")
        
        # 2. Performance Gap Analysis (without graph)
        if len(analyzer.experiments) >= 2:
            gap_results = performance_gap_analysis(analyzer, filters, analysis_mode, use_super_class)
            
            st.markdown("---")
            
            # 3. Experiment-to-Experiment Changes (includes top performing and F1 analyses)
            experiment_to_experiment_changes(analyzer, filters, analysis_mode, use_super_class)
        else:
            st.info("🔍 **Multiple Experiments Required**: Upload more experiments to see gap analysis and experiment comparisons.")
    
    except Exception as e:
        st.error(f"Error in summary analysis: {str(e)}")
        st.info("💡 **Tip**: Make sure your data has the required columns and proper format.")

if __name__ == "__main__":
    st.error("This is a module file. Please run main_dashboard.py instead.")