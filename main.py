import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure package import paths
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(base_dir, 'credit_risk_engine'))
sys.path.insert(0, os.path.join(base_dir, 'src'))

try:
    from src.pipeline import CreditRiskPipeline
except ImportError:
    from pipeline import CreditRiskPipeline

def generate_visualizations(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 11})

    # 1. ROC Curve Plot
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    prim_roc = results['performance']['primary']['roc_curve']
    comp_roc = results['performance']['comparison']['roc_curve']
    prim_auc = results['performance']['primary']['auc_roc']
    comp_auc = results['performance']['comparison']['auc_roc']
    
    ax.plot(prim_roc['fpr'], prim_roc['tpr'], color='#2563eb', lw=2.5, 
            label=f"Primary Scorecard (AUC = {prim_auc:.3f})")
    ax.plot(comp_roc['fpr'], comp_roc['tpr'], color='#10b981', lw=2, linestyle='--',
            label=f"Gradient Boosted Trees (AUC = {comp_auc:.3f})")
    ax.plot([0, 1], [0, 1], color='#94a3b8', linestyle=':', lw=1.5, label="Random Guess (AUC = 0.500)")
    
    ax.set_title("Credit Risk Model Performance - Receiver Operating Characteristic (ROC)", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    ax.legend(loc="lower right", frameon=True)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    plt.tight_layout()
    roc_path = os.path.join(output_dir, 'roc_curve.png')
    plt.savefig(roc_path)
    plt.close()

    # 2. SHAP Global Feature Importance
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    feat_imp = results['global_explainability']['feature_importance'][:10]
    names = [f['feature'] for f in feat_imp][::-1]
    values = [f['importance'] for f in feat_imp][::-1]
    directions = [f['direction'] for f in feat_imp][::-1]
    
    colors = ['#ef4444' if 'Positive' in d else ('#10b981' if 'Negative' in d else '#64748b') for d in directions]
    bars = ax.barh(names, values, color=colors, height=0.65, edgecolor='black', linewidth=0.5)
    
    ax.set_title("Explainable AI: Global SHAP Feature Importance (Mean |SHAP Value|)", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Mean Absolute Impact on Model Default Log-Odds", fontsize=11)
    
    # Custom legend for risk directions
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#ef4444', label='Increases Default Risk (Higher Value -> Higher Risk)'),
        Patch(facecolor='#10b981', label='Protective Factor (Higher Value -> Lower Risk)'),
    ]
    ax.legend(handles=legend_elements, loc="lower right", frameon=True)
    plt.tight_layout()
    shap_path = os.path.join(output_dir, 'shap_global_importance.png')
    plt.savefig(shap_path)
    plt.close()

    # 3. Confusion Matrix Heatmap
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    cm = results['performance']['primary']['confusion_matrix']
    cm_matrix = np.array([[cm['tn'], cm['fp']], [cm['fn'], cm['tp']]])
    
    sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                xticklabels=['Pred Good (0)', 'Pred Default (1)'],
                yticklabels=['Actual Good (0)', 'Actual Default (1)'],
                annot_kws={'size': 14, 'fontweight': 'bold'})
    ax.set_title(f"Confusion Matrix (Threshold = {results['model_info']['decision_threshold']:.2f})", fontsize=12, fontweight='bold', pad=10)
    plt.tight_layout()
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path)
    plt.close()

    # 4. Portfolio Optimization & Cutoff Curve
    fig, ax1 = plt.subplots(figsize=(9, 5), dpi=150)
    cutoff = results['portfolio_impact']['cutoff_optimization_curve']
    th_vals = [c['threshold'] for c in cutoff]
    profits = [c['net_profit'] / 1000.0 for c in cutoff]
    appr_rates = [c['approval_rate'] for c in cutoff]
    
    color_prof = '#2563eb'
    ax1.set_xlabel('Decision Cutoff Threshold (Max Allowed Default Risk)', fontsize=11)
    ax1.set_ylabel('Net Expected Profit ($k)', color=color_prof, fontsize=11)
    line1 = ax1.plot(th_vals, profits, color=color_prof, lw=2.5, marker='o', label='Net Profit ($k)')
    ax1.tick_params(axis='y', labelcolor=color_prof)
    
    ax2 = ax1.twinx()
    color_appr = '#10b981'
    ax2.set_ylabel('Portfolio Approval Rate (%)', color=color_appr, fontsize=11)
    line2 = ax2.plot(th_vals, appr_rates, color=color_appr, lw=2, linestyle='--', marker='s', label='Approval Rate (%)')
    ax2.tick_params(axis='y', labelcolor=color_appr)
    
    # Highlight max profit
    max_idx = int(np.argmax(profits))
    opt_th = th_vals[max_idx]
    opt_profit = profits[max_idx]
    ax1.axvline(opt_th, color='#f59e0b', linestyle=':', lw=2, label=f'Optimal Cutoff ({opt_th:.2f})')
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='lower center', frameon=True)
    ax1.set_title("FinTech Portfolio Optimization: Net Profit vs Approval Cutoff Threshold", fontsize=12, fontweight='bold', pad=12)
    plt.tight_layout()
    cutoff_path = os.path.join(output_dir, 'cutoff_optimization.png')
    plt.savefig(cutoff_path)
    plt.close()

    # 5. Sample Loan Applicant Waterfall Plot
    if len(results['instance_explanations']) > 0:
        sample = results['instance_explanations'][0]
        fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
        wf = sample['explanation']['waterfall_decomposition']
        
        step_names = [s['feature'] for s in wf]
        step_contribs = [s['contribution'] for s in wf]
        colors = ['#ef4444' if c > 0 else '#10b981' for c in step_contribs]
        
        y_pos = np.arange(len(step_names))
        ax.barh(y_pos, step_contribs, color=colors, height=0.6, edgecolor='black', linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(step_names)
        ax.axvline(0, color='black', lw=1)
        ax.set_xlabel("Contribution to Default Log-Odds (SHAP Value)", fontsize=11)
        ax.set_title(f"Explainability Waterfall for Loan {sample['loan_id']} | Score: {sample['credit_score']} ({sample['risk_grade']})",
                     fontsize=12, fontweight='bold', pad=12)
        plt.tight_layout()
        wf_path = os.path.join(output_dir, 'applicant_waterfall.png')
        plt.savefig(wf_path)
        plt.close()

    print(f"Generated 5 diagnostic visualization plots in {output_dir}")

def run_cli():
    parser = argparse.ArgumentParser(description="FinTech Explainable Machine Learning Credit Risk System")
    parser.add_argument('--dataset', type=str, default='/working_dir/c_f089660531aacfaf/data/lending_club_sample.csv',
                        help="Path to CSV dataset or 'german' / 'give_me_credit'")
    parser.add_argument('--model', type=str, default='logistic', choices=['logistic', 'tree'],
                        help="Model architecture ('logistic' or 'tree')")
    parser.add_argument('--threshold', type=float, default=0.20,
                        help="Approval cutoff decision threshold (default: 0.20)")
    parser.add_argument('--output_dir', type=str, default='/working_dir/c_f089660531aacfaf/artifacts',
                        help="Directory to save output metrics and plots")
    args = parser.parse_args()

    dataset_path = args.dataset
    if dataset_path == 'german':
        dataset_path = '/working_dir/c_f089660531aacfaf/data/german_credit_sample.csv'
    elif dataset_path == 'give_me_credit':
        dataset_path = '/working_dir/c_f089660531aacfaf/data/give_me_some_credit_sample.csv'

    print("=" * 70)
    print("FINTECH EXPLAINABLE MACHINE LEARNING SYSTEM FOR CREDIT RISK")
    print("=" * 70)
    print(f"Loading Dataset: {dataset_path}")
    print(f"Model: {args.model}")
    print(f"Cutoff Threshold: {args.threshold}")

    pipeline = CreditRiskPipeline(model_type=args.model)
    results = pipeline.run(dataset_path, threshold=args.threshold)

    # Save JSON results
    json_path = os.path.join(args.output_dir, 'results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved complete run results to {json_path}")

    # Generate charts
    plots_dir = os.path.join(args.output_dir, 'plots')
    generate_visualizations(results, plots_dir)

    # Print Executive Summary Table
    perf = results['performance']['primary']
    port = results['portfolio_impact']
    print("\n" + "=" * 70)
    print("EXECUTIVE SUMMARY & MODEL METRICS")
    print("=" * 70)
    print(f"Total Applications Analyzed : {results['data_summary']['total_rows']}")
    print(f"Dataset Default Rate        : {results['data_summary']['default_rate']:.2f}%")
    print(f"Model AUC-ROC               : {perf['auc_roc']:.4f}")
    print(f"Model Precision             : {perf['precision']:.4f}")
    print(f"Model Recall (Sensitivity)  : {perf['recall']:.4f}")
    print(f"Brier Score Calibration     : {perf['brier_score']:.4f}")
    print(f"Approval Rate at Threshold  : {port['approval_rate_pct']:.2f}% ({port['approved_count']} approved)")
    print(f"Portfolio Origination Vol   : ${port['total_originated_volume']:,.2f}")
    print(f"Expected Interest Revenue   : ${port['expected_interest_revenue']:,.2f}")
    print(f"Expected Default Losses     : ${port['expected_default_loss']:,.2f}")
    print(f"Net Expected Portfolio Profit: ${port['net_expected_profit']:,.2f} (ROI: {port['roi_pct']:.2f}%)")
    print("=" * 70)

    print("\nTop 5 Risk Factors (SHAP Global Attribution):")
    for f in results['global_explainability']['feature_importance'][:5]:
        print(f"  • {f['feature']:<25} | SHAP: {f['importance']:.4f} | {f['direction']}")
    print("=" * 70)

if __name__ == '__main__':
    run_cli()
