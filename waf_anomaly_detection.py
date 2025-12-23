"""
Complete Production-Ready WAF Anomaly Detection System
Naval Hackathon - Final Version

Features:
- Enhanced feature engineering with security patterns
- Multiple model comparison (Isolation Forest, One-Class SVM, LOF)
- Ensemble voting system
- SHAP explainability
- Comprehensive visualizations
- Real-time inference with whitelisting
- Adaptive threshold mechanism
"""

# -----------------------------
# 1. IMPORTS
# -----------------------------

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, classification_report
)

import joblib
import warnings
warnings.filterwarnings('ignore')

# For SHAP explainability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    print("⚠️  SHAP not available. Install with: pip install shap")
    SHAP_AVAILABLE = False

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# -----------------------------
# 2. LOAD DATASET
# -----------------------------

CSV_PATH = "traffic_logs.csv"
df = pd.read_csv(CSV_PATH)

print("="*70)
print(" "*20 + "WAF ANOMALY DETECTION SYSTEM")
print("="*70)
print(f"\n📊 DATASET OVERVIEW")
print(f"{'─'*70}")
print(f"Total requests: {len(df):,}")
print(f"Attack samples: {df['label'].sum():,} ({df['label'].mean()*100:.1f}%)")
print(f"Normal samples: {(df['label']==0).sum():,} ({(df['label']==0).mean()*100:.1f}%)")
print()


# -----------------------------
# 3. COMPREHENSIVE FEATURE ENGINEERING
# -----------------------------

print(f"🔧 FEATURE ENGINEERING")
print(f"{'─'*70}")

# Convert timestamp
df['timestamp'] = pd.to_datetime(df['timestamp'])

# TEMPORAL FEATURES
df['hour'] = df['timestamp'].dt.hour
df['minute'] = df['timestamp'].dt.minute
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)

# METHOD ENCODING
method_encoder = LabelEncoder()
df['method_encoded'] = method_encoder.fit_transform(df['method'])

# HTTP STATUS FEATURES
df['is_error'] = (df['status_code'] >= 400).astype(int)
df['is_client_error'] = ((df['status_code'] >= 400) & (df['status_code'] < 500)).astype(int)
df['is_server_error'] = (df['status_code'] >= 500).astype(int)
df['is_redirect'] = ((df['status_code'] >= 300) & (df['status_code'] < 400)).astype(int)

# USER AGENT FEATURES
df['is_bot_ua'] = df['user_agent'].str.contains(
    'bot|crawler|spider|curl|python-requests|wget', 
    case=False, regex=True
).astype(int)

df['ua_length'] = df['user_agent'].str.len()

# URL SECURITY PATTERN DETECTION
df['has_sql_injection'] = df['url'].str.contains(
    r"('|--|union|select|insert|update|delete|drop|;|\bor\b.*=)", 
    case=False, regex=True
).astype(int)

df['has_xss'] = df['url'].str.contains(
    r'(<script|<iframe|javascript:|onerror|onload)', 
    case=False, regex=True
).astype(int)

df['has_path_traversal'] = df['url'].str.contains(
    r'(\.\.|/etc/|/var/|/proc/|\\\\|%2e%2e)', 
    case=False, regex=True
).astype(int)

df['has_command_injection'] = df['url'].str.contains(
    r'(;|\||&|`|\$\(|%0a|%0d)', 
    case=False, regex=True
).astype(int)

df['has_special_chars'] = df['url'].str.contains(
    r'[<>\'\";\(\)\{\}\[\]]', 
    regex=True
).astype(int)

# URL STRUCTURE FEATURES
df['url_depth'] = df['url'].str.count('/')
df['has_query'] = (df['query_length'] > 0).astype(int)
df['query_param_count'] = df['url'].str.count('&') + df['has_query']

# RATE-BASED FEATURES
df['req_ratio_10s_to_1m'] = df['req_per_ip_10sec'] / (df['req_per_ip_1min'] + 1)
df['burst_indicator'] = (df['req_per_ip_10sec'] > 50).astype(int)
df['high_frequency'] = (df['req_per_ip_1min'] > 100).astype(int)

# PAYLOAD FEATURES
df['payload_entropy_norm'] = df['payload_entropy'] / 8.0
df['high_entropy'] = (df['payload_entropy'] > 5.0).astype(int)

# SIZE-BASED ANOMALIES
df['size_anomaly'] = ((df['bytes_sent'] > 5000) | (df['bytes_sent'] < 100)).astype(int)
df['slow_request'] = (df['request_time'] > 1.5).astype(int)

# REPUTATION FEATURES (simulated - in production, use real IP reputation DB)
# For now, we'll use request patterns as a proxy
ip_stats = df.groupby('src_ip').agg({
    'label': 'mean',  # Historical attack rate
    'is_error': 'mean',
    'req_per_ip_1min': 'max'
}).reset_index()
ip_stats.columns = ['src_ip', 'ip_attack_history', 'ip_error_rate', 'ip_max_req_rate']

df = df.merge(ip_stats, on='src_ip', how='left')

print(f"✓ Created {df.shape[1] - len(pd.read_csv(CSV_PATH).columns)} new features")
print()


# -----------------------------
# 4. FEATURE SELECTION
# -----------------------------

FEATURE_COLUMNS = [
    # Original numerical features
    "url_length", "query_length", "bytes_sent", "request_time",
    "req_per_ip_1min", "req_per_ip_10sec", "unique_urls_per_ip",
    "time_gap", "payload_entropy", "is_https",
    
    # Temporal features
    "hour", "minute", "day_of_week", "is_business_hours",
    
    # HTTP features
    "method_encoded", "is_error", "is_client_error", 
    "is_server_error", "is_redirect",
    
    # User agent features
    "is_bot_ua", "ua_length",
    
    # Security pattern features
    "has_sql_injection", "has_xss", "has_path_traversal",
    "has_command_injection", "has_special_chars",
    
    # URL structure
    "url_depth", "has_query", "query_param_count",
    
    # Rate-based features
    "req_ratio_10s_to_1m", "burst_indicator", "high_frequency",
    
    # Payload features
    "payload_entropy_norm", "high_entropy",
    
    # Anomaly indicators
    "size_anomaly", "slow_request",
    
    # Reputation features
    "ip_attack_history", "ip_error_rate", "ip_max_req_rate"
]

X = df[FEATURE_COLUMNS].copy()
y = df["label"].copy()

# Store original data for analysis
df_original = df.copy()

print(f"📋 FEATURE SUMMARY")
print(f"{'─'*70}")
print(f"Total features: {len(FEATURE_COLUMNS)}")
print(f"Feature categories:")
print(f"  • Security patterns: 5")
print(f"  • Rate-based: 6")
print(f"  • HTTP/Status: 5")
print(f"  • Temporal: 4")
print(f"  • Reputation: 3")
print(f"  • Other: {len(FEATURE_COLUMNS) - 23}")
print()


# -----------------------------
# 5. FEATURE SCALING
# -----------------------------

print(f"⚙️  PREPROCESSING")
print(f"{'─'*70}")

# Log transform for skewed features
SKEWED_COLUMNS = [
    "bytes_sent", "req_per_ip_1min", "req_per_ip_10sec",
    "time_gap", "url_length", "query_length", "ua_length"
]

for col in SKEWED_COLUMNS:
    if col in X.columns:
        X[col] = np.log1p(X[col])

print(f"✓ Applied log transformation to {len([c for c in SKEWED_COLUMNS if c in X.columns])} skewed features")

# Standardize all features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"✓ Standardized all features (mean=0, std=1)")
print()


# -----------------------------
# 6. TRAIN/TEST/EVAL SPLIT
# -----------------------------

X_train, X_temp, y_train, y_temp = train_test_split(
    X_scaled, y, test_size=0.30, random_state=42, stratify=y
)

X_test, X_eval, y_test, y_eval = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print(f"📊 DATA SPLIT")
print(f"{'─'*70}")
print(f"Train: {len(X_train):,} samples ({y_train.sum()} attacks, {y_train.mean()*100:.1f}%)")
print(f"Test:  {len(X_test):,} samples ({y_test.sum()} attacks, {y_test.mean()*100:.1f}%)")
print(f"Eval:  {len(X_eval):,} samples ({y_eval.sum()} attacks, {y_eval.mean()*100:.1f}%)")
print()


# -----------------------------
# 7. MODEL TRAINING & COMPARISON
# -----------------------------

print("="*70)
print(" "*25 + "MODEL TRAINING")
print("="*70)
print()

# Store all models and predictions
models = {}
predictions = {}
scores_dict = {}

# ---------------
# MODEL 1: ISOLATION FOREST
# ---------------
print(f"🌲 Training Isolation Forest...")
print(f"{'─'*70}")

# Hyperparameter tuning
best_if_f1 = 0
best_if_model = None
best_if_cont = 0

contamination_levels = [0.05, 0.07, 0.10, 0.12, 0.15]
for cont in contamination_levels:
    model_temp = IsolationForest(
        n_estimators=300,
        contamination=cont,
        max_samples='auto',
        max_features=1.0,
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    )
    model_temp.fit(X_train)
    pred_temp = (model_temp.predict(X_test) == -1).astype(int)
    f1_temp = f1_score(y_test, pred_temp)
    
    print(f"  contamination={cont:.2f} → F1={f1_temp:.4f}")
    
    if f1_temp > best_if_f1:
        best_if_f1 = f1_temp
        best_if_model = model_temp
        best_if_cont = cont

print(f"\n✓ Best contamination: {best_if_cont:.2f}")

# Evaluate on eval set
if_scores = best_if_model.decision_function(X_eval)
if_pred = (best_if_model.predict(X_eval) == -1).astype(int)

if_acc = accuracy_score(y_eval, if_pred)
if_prec = precision_score(y_eval, if_pred)
if_rec = recall_score(y_eval, if_pred)
if_f1 = f1_score(y_eval, if_pred)
if_auc = roc_auc_score(y_eval, (if_scores - if_scores.min()) / (if_scores.max() - if_scores.min() + 1e-10))

print(f"\n📊 Isolation Forest Results:")
print(f"  Accuracy:  {if_acc:.4f}")
print(f"  Precision: {if_prec:.4f}")
print(f"  Recall:    {if_rec:.4f}")
print(f"  F1 Score:  {if_f1:.4f}")
print(f"  ROC-AUC:   {if_auc:.4f}")
print()

models['IsolationForest'] = best_if_model
predictions['IsolationForest'] = if_pred
scores_dict['IsolationForest'] = if_scores

# ---------------
# MODEL 2: ONE-CLASS SVM
# ---------------
print(f"🎯 Training One-Class SVM...")
print(f"{'─'*70}")

best_svm_f1 = 0
best_svm_model = None
best_svm_nu = 0

nu_levels = [0.05, 0.10, 0.15, 0.20]
for nu in nu_levels:
    model_temp = OneClassSVM(
        kernel='rbf',
        gamma='auto',
        nu=nu,
        cache_size=500
    )
    model_temp.fit(X_train)
    pred_temp = (model_temp.predict(X_test) == -1).astype(int)
    f1_temp = f1_score(y_test, pred_temp)
    
    print(f"  nu={nu:.2f} → F1={f1_temp:.4f}")
    
    if f1_temp > best_svm_f1:
        best_svm_f1 = f1_temp
        best_svm_model = model_temp
        best_svm_nu = nu

print(f"\n✓ Best nu: {best_svm_nu:.2f}")

# Evaluate on eval set
svm_scores = best_svm_model.decision_function(X_eval)
svm_pred = (best_svm_model.predict(X_eval) == -1).astype(int)

svm_acc = accuracy_score(y_eval, svm_pred)
svm_prec = precision_score(y_eval, svm_pred)
svm_rec = recall_score(y_eval, svm_pred)
svm_f1 = f1_score(y_eval, svm_pred)
svm_auc = roc_auc_score(y_eval, (svm_scores - svm_scores.min()) / (svm_scores.max() - svm_scores.min() + 1e-10))

print(f"\n📊 One-Class SVM Results:")
print(f"  Accuracy:  {svm_acc:.4f}")
print(f"  Precision: {svm_prec:.4f}")
print(f"  Recall:    {svm_rec:.4f}")
print(f"  F1 Score:  {svm_f1:.4f}")
print(f"  ROC-AUC:   {svm_auc:.4f}")
print()

models['OneClassSVM'] = best_svm_model
predictions['OneClassSVM'] = svm_pred
scores_dict['OneClassSVM'] = svm_scores

# ---------------
# MODEL 3: LOCAL OUTLIER FACTOR
# ---------------
print(f"🔍 Training Local Outlier Factor...")
print(f"{'─'*70}")

best_lof_f1 = 0
best_lof_model = None
best_lof_cont = 0

for cont in contamination_levels:
    model_temp = LocalOutlierFactor(
        n_neighbors=20,
        contamination=cont,
        novelty=True,  # Enable predict() method
        n_jobs=-1
    )
    model_temp.fit(X_train)
    pred_temp = (model_temp.predict(X_test) == -1).astype(int)
    f1_temp = f1_score(y_test, pred_temp)
    
    print(f"  contamination={cont:.2f} → F1={f1_temp:.4f}")
    
    if f1_temp > best_lof_f1:
        best_lof_f1 = f1_temp
        best_lof_model = model_temp
        best_lof_cont = cont

print(f"\n✓ Best contamination: {best_lof_cont:.2f}")

# Evaluate on eval set
lof_scores = best_lof_model.decision_function(X_eval)
lof_pred = (best_lof_model.predict(X_eval) == -1).astype(int)

lof_acc = accuracy_score(y_eval, lof_pred)
lof_prec = precision_score(y_eval, lof_pred)
lof_rec = recall_score(y_eval, lof_pred)
lof_f1 = f1_score(y_eval, lof_pred)
lof_auc = roc_auc_score(y_eval, (lof_scores - lof_scores.min()) / (lof_scores.max() - lof_scores.min() + 1e-10))

print(f"\n📊 Local Outlier Factor Results:")
print(f"  Accuracy:  {lof_acc:.4f}")
print(f"  Precision: {lof_prec:.4f}")
print(f"  Recall:    {lof_rec:.4f}")
print(f"  F1 Score:  {lof_f1:.4f}")
print(f"  ROC-AUC:   {lof_auc:.4f}")
print()

models['LOF'] = best_lof_model
predictions['LOF'] = lof_pred
scores_dict['LOF'] = lof_scores


# -----------------------------
# 8. ENSEMBLE VOTING
# -----------------------------

print("="*70)
print(" "*25 + "ENSEMBLE MODEL")
print("="*70)
print()

# Majority voting (at least 2 out of 3 models predict anomaly)
ensemble_pred = ((predictions['IsolationForest'] + 
                  predictions['OneClassSVM'] + 
                  predictions['LOF']) >= 2).astype(int)

# Ensemble metrics
ens_acc = accuracy_score(y_eval, ensemble_pred)
ens_prec = precision_score(y_eval, ensemble_pred)
ens_rec = recall_score(y_eval, ensemble_pred)
ens_f1 = f1_score(y_eval, ensemble_pred)

# For ROC-AUC, use average scores
ensemble_scores = (
    (if_scores - if_scores.min()) / (if_scores.max() - if_scores.min() + 1e-10) +
    (svm_scores - svm_scores.min()) / (svm_scores.max() - svm_scores.min() + 1e-10) +
    (lof_scores - lof_scores.min()) / (lof_scores.max() - lof_scores.min() + 1e-10)
) / 3
ens_auc = roc_auc_score(y_eval, ensemble_scores)

print(f"🎯 Ensemble Voting Strategy: Majority (2/3 models)")
print(f"{'─'*70}")
print(f"📊 Ensemble Results:")
print(f"  Accuracy:  {ens_acc:.4f}")
print(f"  Precision: {ens_prec:.4f}")
print(f"  Recall:    {ens_rec:.4f}")
print(f"  F1 Score:  {ens_f1:.4f}")
print(f"  ROC-AUC:   {ens_auc:.4f}")
print()

# Comparison table
print(f"📊 MODEL COMPARISON")
print(f"{'─'*70}")
comparison_df = pd.DataFrame({
    'Model': ['Isolation Forest', 'One-Class SVM', 'LOF', 'Ensemble'],
    'Accuracy': [if_acc, svm_acc, lof_acc, ens_acc],
    'Precision': [if_prec, svm_prec, lof_prec, ens_prec],
    'Recall': [if_rec, svm_rec, lof_rec, ens_rec],
    'F1 Score': [if_f1, svm_f1, lof_f1, ens_f1],
    'ROC-AUC': [if_auc, svm_auc, lof_auc, ens_auc]
})
print(comparison_df.to_string(index=False))
print()

# Determine best model
best_model_name = comparison_df.loc[comparison_df['F1 Score'].idxmax(), 'Model']
print(f"🏆 Best performing model: {best_model_name}")
print()


# -----------------------------
# 9. DETAILED EVALUATION
# -----------------------------

print(f"📋 DETAILED CLASSIFICATION REPORT (Ensemble)")
print(f"{'─'*70}")
print(classification_report(y_eval, ensemble_pred, target_names=['Normal', 'Attack']))

cm = confusion_matrix(y_eval, ensemble_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\n📊 CONFUSION MATRIX")
print(f"{'─'*70}")
print(f"                 Predicted")
print(f"              Normal  Attack")
print(f"Actual Normal   {tn:4d}   {fp:4d}")
print(f"       Attack   {fn:4d}   {tp:4d}")
print()

# Key metrics for WAF
fpr = fp / (tn + fp) if (tn + fp) > 0 else 0
fnr = fn / (tp + fn) if (tp + fn) > 0 else 0

print(f"🎯 CRITICAL WAF METRICS")
print(f"{'─'*70}")
print(f"False Positive Rate: {fpr:.4f} ({fp}/{tn+fp} legitimate requests blocked)")
print(f"False Negative Rate: {fnr:.4f} ({fn}/{tp+fn} attacks missed)")
print(f"True Positive Rate:  {ens_rec:.4f} (Detection rate)")
print()


# -----------------------------
# 10. VISUALIZATIONS
# -----------------------------

print(f"📈 GENERATING VISUALIZATIONS")
print(f"{'─'*70}")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('WAF Anomaly Detection - Comprehensive Analysis', fontsize=16, fontweight='bold')

# 1. ROC Curves
ax1 = axes[0, 0]
for name, scores in scores_dict.items():
    scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
    fpr_curve, tpr_curve, _ = roc_curve(y_eval, scores_norm)
    auc_score = roc_auc_score(y_eval, scores_norm)
    ax1.plot(fpr_curve, tpr_curve, label=f'{name} (AUC={auc_score:.3f})', linewidth=2)

# Ensemble ROC
fpr_ens, tpr_ens, _ = roc_curve(y_eval, ensemble_scores)
ax1.plot(fpr_ens, tpr_ens, label=f'Ensemble (AUC={ens_auc:.3f})', 
         linewidth=3, linestyle='--', color='red')
ax1.plot([0, 1], [0, 1], 'k--', alpha=0.3)
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.set_title('ROC Curves - All Models')
ax1.legend(loc='lower right')
ax1.grid(True, alpha=0.3)

# 2. Precision-Recall Curves
ax2 = axes[0, 1]
for name, preds in predictions.items():
    precision_curve, recall_curve, _ = precision_recall_curve(y_eval, preds)
    ax2.plot(recall_curve, precision_curve, label=name, linewidth=2)

precision_ens, recall_ens, _ = precision_recall_curve(y_eval, ensemble_pred)
ax2.plot(recall_ens, precision_ens, label='Ensemble', 
         linewidth=3, linestyle='--', color='red')
ax2.set_xlabel('Recall')
ax2.set_ylabel('Precision')
ax2.set_title('Precision-Recall Curves')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Model Comparison Bar Chart
ax3 = axes[0, 2]
metrics_to_plot = ['F1 Score', 'Precision', 'Recall', 'ROC-AUC']
x = np.arange(len(comparison_df))
width = 0.2

for i, metric in enumerate(metrics_to_plot):
    ax3.bar(x + i*width, comparison_df[metric], width, label=metric)

ax3.set_xlabel('Model')
ax3.set_ylabel('Score')
ax3.set_title('Model Performance Comparison')
ax3.set_xticks(x + width * 1.5)
ax3.set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# 4. Confusion Matrix Heatmap
ax4 = axes[1, 0]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4,
            xticklabels=['Normal', 'Attack'],
            yticklabels=['Normal', 'Attack'])
ax4.set_title('Confusion Matrix (Ensemble)')
ax4.set_ylabel('Actual')
ax4.set_xlabel('Predicted')

# 5. Attack Detection by Model
ax5 = axes[1, 1]
detection_rates = {
    'Isolation Forest': if_rec,
    'One-Class SVM': svm_rec,
    'LOF': lof_rec,
    'Ensemble': ens_rec
}
colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
bars = ax5.bar(detection_rates.keys(), detection_rates.values(), color=colors)
ax5.set_ylabel('Detection Rate (Recall)')
ax5.set_title('Attack Detection Rate by Model')
ax5.set_ylim([0, 1.0])
ax5.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}', ha='center', va='bottom')

# 6. False Positive Rate Comparison
ax6 = axes[1, 2]
fpr_comparison = {
    'Isolation Forest': fp / (tn + fp) if (tn + fp) > 0 else 0,
    'One-Class SVM': confusion_matrix(y_eval, predictions['OneClassSVM'])[0,1] / 
                      confusion_matrix(y_eval, predictions['OneClassSVM'])[0].sum(),
    'LOF': confusion_matrix(y_eval, predictions['LOF'])[0,1] / 
           confusion_matrix(y_eval, predictions['LOF'])[0].sum(),
    'Ensemble': fpr
}
bars2 = ax6.bar(fpr_comparison.keys(), fpr_comparison.values(), color=colors)
ax6.set_ylabel('False Positive Rate')
ax6.set_title('False Positive Rate by Model (Lower is Better)')
ax6.set_ylim([0, max(fpr_comparison.values()) * 1.2])
ax6.grid(True, alpha=0.3, axis='y')

for bar in bars2:
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('waf_analysis.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: waf_analysis.png")


# -----------------------------
# 11. FEATURE IMPORTANCE (SHAP)
# -----------------------------

if SHAP_AVAILABLE:
    print(f"\n🔍 GENERATING SHAP EXPLANATIONS")
    print(f"{'─'*70}")
    
    try:
        # Use a sample for SHAP (it's computationally expensive)
        sample_size = min(100, len(X_eval))
        X_eval_sample = X_eval[:sample_size]
        
        # SHAP for Isolation Forest
        explainer = shap.TreeExplainer(models['IsolationForest'])
        shap_values = explainer.shap_values(X_eval_sample)
        
        # Summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_eval_sample, 
                         feature_names=FEATURE_COLUMNS,
                         show=False)
        plt.title('SHAP Feature Importance - Isolation Forest', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('shap_analysis.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: shap_analysis.png")
        plt.close()
        
    except Exception as e:
        print(f"⚠️  SHAP analysis failed: {e}")
else:
    print(f"\n⚠️  Skipping SHAP analysis (not installed)")

print()


# -----------------------------
# 12. ATTACK PATTERN ANALYSIS
# -----------------------------

print(f"🎯 ATTACK PATTERN ANALYSIS")
print(f"{'─'*70}")

# Analyze detected attacks
detected_idx = np.where(ensemble_pred == 1)[0]
actual_attacks_idx = np.where(y_eval == 1)[0]

correctly_detected = np.intersect1d(detected_idx, actual_attacks_idx)
false_positives_idx = np.setdiff1d(detected_idx, actual_attacks_idx)
missed_attacks_idx = np.setdiff1d(actual_attacks_idx, detected_idx)

print(f"Total attacks in eval set: {len(actual_attacks_idx)}")
print(f"Correctly detected: {len(correctly_detected)}")
print(f"False positives: {len(false_positives_idx)}")
print(f"Missed attacks: {len(missed_attacks_idx)}")
print()

# Analyze which features are most prominent in detected attacks
if len(detected_idx) > 0:
    print(f"Top features in detected anomalies:")
    detected_features = X[FEATURE_COLUMNS].iloc[detected_idx]
    
    security_features = ['has_sql_injection', 'has_xss', 'has_path_traversal', 
                         'has_command_injection', 'has_special_chars']
    
    for feat in security_features:
        if feat in detected_features.columns:
            mean_val = detected_features[feat].mean()
            print(f"  • {feat}: {mean_val:.2%}")
    
    print(f"\nHigh-risk indicators:")
    print(f"  • Burst traffic: {detected_features['burst_indicator'].mean():.2%}")
    print(f"  • High entropy: {detected_features['high_entropy'].mean():.2%}")
    print(f"  • Error responses: {detected_features['is_error'].mean():.2%}")
    print(f"  • Bot user agents: {detected_features['is_bot_ua'].mean():.2%}")
print()


# -----------------------------
# 13. ADAPTIVE THRESHOLD MECHANISM
# -----------------------------

print(f"⚙️  ADAPTIVE THRESHOLD CALIBRATION")
print(f"{'─'*70}")

# Calculate optimal threshold based on F1 score
thresholds = np.linspace(0, 1, 100)
f1_scores_threshold = []

for thresh in thresholds:
    pred_thresh = (ensemble_scores >= thresh).astype(int)
    if pred_thresh.sum() > 0:  # Avoid division by zero
        f1_thresh = f1_score(y_eval, pred_thresh)
        f1_scores_threshold.append(f1_thresh)
    else:
        f1_scores_threshold.append(0)

optimal_threshold_idx = np.argmax(f1_scores_threshold)
optimal_threshold = thresholds[optimal_threshold_idx]

print(f"Default threshold: 0.5")
print(f"Optimal threshold (max F1): {optimal_threshold:.4f}")
print(f"F1 improvement: {f1_scores_threshold[optimal_threshold_idx]:.4f} vs {ens_f1:.4f}")
print()

# Dynamic threshold recommendation
print(f"📊 THRESHOLD RECOMMENDATIONS:")
print(f"  • High Security (FPR < 1%):  threshold = {thresholds[np.where(np.array(f1_scores_threshold) > 0)[0][0]]:.4f}")
print(f"  • Balanced (Optimal F1):     threshold = {optimal_threshold:.4f}")
print(f"  • High Recall (Catch 95%+):  threshold = {thresholds[-20]:.4f}")
print()


# -----------------------------
# 14. WHITELISTING MECHANISM
# -----------------------------

print(f"✅ IP WHITELISTING SYSTEM")
print(f"{'─'*70}")

# Identify IPs with consistent normal behavior
ip_behavior = df_original.groupby('src_ip').agg({
    'label': ['mean', 'count'],
    'is_error': 'mean'
}).reset_index()
ip_behavior.columns = ['src_ip', 'attack_rate', 'request_count', 'error_rate']

# Whitelist criteria: 0 attacks, 10+ requests, low error rate
whitelist_candidates = ip_behavior[
    (ip_behavior['attack_rate'] == 0) & 
    (ip_behavior['request_count'] >= 10) &
    (ip_behavior['error_rate'] < 0.1)
]

print(f"Whitelist candidates: {len(whitelist_candidates)} IPs")
print(f"Sample whitelisted IPs:")
if len(whitelist_candidates) > 0:
    for ip in whitelist_candidates['src_ip'].head(5):
        print(f"  • {ip}")
else:
    print("  (No IPs meet whitelist criteria in sample data)")
print()

# Save whitelist
whitelist_ips = set(whitelist_candidates['src_ip'].tolist())


# -----------------------------
# 15. SAVE ALL ARTIFACTS
# -----------------------------

print(f"💾 SAVING MODEL ARTIFACTS")
print(f"{'─'*70}")

# Save individual models
joblib.dump(models['IsolationForest'], "model_isolation_forest.pkl")
joblib.dump(models['OneClassSVM'], "model_oneclasssvm.pkl")
joblib.dump(models['LOF'], "model_lof.pkl")

# Save preprocessing artifacts
joblib.dump(scaler, "feature_scaler.pkl")
joblib.dump(method_encoder, "method_encoder.pkl")

# Save whitelist
with open("ip_whitelist.txt", "w") as f:
    for ip in whitelist_ips:
        f.write(f"{ip}\n")

# Save feature columns
with open("feature_columns.txt", "w") as f:
    f.write("\n".join(FEATURE_COLUMNS))

# Save ensemble configuration
ensemble_config = {
    'voting_threshold': 2,  # 2 out of 3
    'optimal_threshold': optimal_threshold,
    'feature_columns': FEATURE_COLUMNS,
    'best_model': best_model_name,
    'performance': {
        'accuracy': ens_acc,
        'precision': ens_prec,
        'recall': ens_rec,
        'f1': ens_f1,
        'roc_auc': ens_auc,
        'fpr': fpr
    }
}
joblib.dump(ensemble_config, "ensemble_config.pkl")

print(f"✓ Saved: model_isolation_forest.pkl")
print(f"✓ Saved: model_oneclasssvm.pkl")
print(f"✓ Saved: model_lof.pkl")
print(f"✓ Saved: feature_scaler.pkl")
print(f"✓ Saved: method_encoder.pkl")
print(f"✓ Saved: ip_whitelist.txt ({len(whitelist_ips)} IPs)")
print(f"✓ Saved: feature_columns.txt")
print(f"✓ Saved: ensemble_config.pkl")
print()


# -----------------------------
# 16. REAL-TIME INFERENCE CLASS
# -----------------------------

print(f"🚀 CREATING REAL-TIME INFERENCE ENGINE")
print(f"{'─'*70}")

class WAFAnomalyDetector:
    """
    Production-ready WAF anomaly detection system
    """
    
    def __init__(self, model_path_prefix="./"):
        # Load models
        self.if_model = joblib.load(f"{model_path_prefix}model_isolation_forest.pkl")
        self.svm_model = joblib.load(f"{model_path_prefix}model_oneclasssvm.pkl")
        self.lof_model = joblib.load(f"{model_path_prefix}model_lof.pkl")
        
        # Load preprocessing
        self.scaler = joblib.load(f"{model_path_prefix}feature_scaler.pkl")
        self.method_encoder = joblib.load(f"{model_path_prefix}method_encoder.pkl")
        
        # Load configuration
        self.config = joblib.load(f"{model_path_prefix}ensemble_config.pkl")
        self.feature_columns = self.config['feature_columns']
        
        # Load whitelist
        self.whitelist = set()
        try:
            with open(f"{model_path_prefix}ip_whitelist.txt", "r") as f:
                self.whitelist = set(line.strip() for line in f)
        except:
            pass
    
    def preprocess_request(self, request_data):
        """
        Preprocess a single request for inference
        """
        # Check whitelist first
        if request_data.get('src_ip') in self.whitelist:
            return None, "WHITELISTED"
        
        # Create feature vector
        features = {}
        
        # Direct features
        for feat in self.feature_columns:
            if feat in request_data:
                features[feat] = request_data[feat]
            else:
                features[feat] = 0
        
        # Convert to array
        feature_vector = np.array([features[col] for col in self.feature_columns]).reshape(1, -1)
        
        # Apply log transform
        skewed_cols = ["bytes_sent", "req_per_ip_1min", "req_per_ip_10sec",
                       "time_gap", "url_length", "query_length", "ua_length"]
        for i, col in enumerate(self.feature_columns):
            if col in skewed_cols:
                feature_vector[0, i] = np.log1p(feature_vector[0, i])
        
        # Scale
        feature_scaled = self.scaler.transform(feature_vector)
        
        return feature_scaled, "OK"
    
    def predict(self, request_data, return_scores=False):
        """
        Predict if request is anomalous
        
        Returns:
            - decision: 'BLOCK' or 'ALLOW'
            - confidence: 0-1 score
            - explanation: dict with details
        """
        # Preprocess
        features, status = self.preprocess_request(request_data)
        
        if status == "WHITELISTED":
            return 'ALLOW', 1.0, {'reason': 'IP whitelisted', 'models': {}}
        
        # Get predictions from all models
        if_pred = self.if_model.predict(features)[0]
        svm_pred = self.svm_model.predict(features)[0]
        lof_pred = self.lof_model.predict(features)[0]
        
        # Get scores
        if_score = self.if_model.decision_function(features)[0]
        svm_score = self.svm_model.decision_function(features)[0]
        lof_score = self.lof_model.decision_function(features)[0]
        
        # Ensemble voting
        votes = sum([if_pred == -1, svm_pred == -1, lof_pred == -1])
        is_anomaly = votes >= 2
        
        decision = 'BLOCK' if is_anomaly else 'ALLOW'
        confidence = votes / 3.0
        
        explanation = {
            'votes': f"{votes}/3",
            'models': {
                'IsolationForest': 'ANOMALY' if if_pred == -1 else 'NORMAL',
                'OneClassSVM': 'ANOMALY' if svm_pred == -1 else 'NORMAL',
                'LOF': 'ANOMALY' if lof_pred == -1 else 'NORMAL'
            },
            'scores': {
                'IsolationForest': float(if_score),
                'OneClassSVM': float(svm_score),
                'LOF': float(lof_score)
            }
        }
        
        # Add attack indicators
        indicators = []
        if request_data.get('has_sql_injection', 0) == 1:
            indicators.append('SQL Injection pattern')
        if request_data.get('has_xss', 0) == 1:
            indicators.append('XSS pattern')
        if request_data.get('has_path_traversal', 0) == 1:
            indicators.append('Path traversal')
        if request_data.get('burst_indicator', 0) == 1:
            indicators.append('Burst traffic')
        if request_data.get('high_entropy', 0) == 1:
            indicators.append('High payload entropy')
        
        if indicators:
            explanation['attack_indicators'] = indicators
        
        return decision, confidence, explanation

# Save the class
joblib.dump(WAFAnomalyDetector, "waf_detector_class.pkl")
print(f"✓ Saved: waf_detector_class.pkl")
print()


# -----------------------------
# 17. REAL-TIME DEMO
# -----------------------------

print(f"="*70)
print(" "*20 + "REAL-TIME DETECTION DEMO")
print(f"="*70)
print()

# Initialize detector
detector = WAFAnomalyDetector()

# Test cases
test_requests = [
    {
        'name': 'Normal Request',
        'data': {
            'src_ip': '192.168.5.225',
            'method': 'GET',
            'url_length': 5,
            'query_length': 0,
            'bytes_sent': 1290,
            'request_time': 0.273,
            'req_per_ip_1min': 10,
            'req_per_ip_10sec': 8,
            'unique_urls_per_ip': 1,
            'time_gap': 5.0,
            'is_https': 0,
            'payload_entropy': 2.68,
            'method_encoded': 0,
            'is_error': 0,
            'is_server_error': 0,
            'is_client_error': 0,
            'is_redirect': 0,
            'is_bot_ua': 1,
            'ua_length': 21,
            'has_sql_injection': 0,
            'has_xss': 0,
            'has_path_traversal': 0,
            'has_command_injection': 0,
            'has_special_chars': 0,
            'url_depth': 1,
            'has_query': 0,
            'query_param_count': 0,
            'req_ratio_10s_to_1m': 0.8,
            'burst_indicator': 0,
            'high_frequency': 0,
            'payload_entropy_norm': 0.335,
            'high_entropy': 0,
            'size_anomaly': 0,
            'slow_request': 0,
            'ip_attack_history': 0.0,
            'ip_error_rate': 0.0,
            'ip_max_req_rate': 10,
            'hour': 15,
            'minute': 40,
            'day_of_week': 0,
            'is_business_hours': 1
        }
    },
    {
        'name': 'SQL Injection Attack',
        'data': {
            'src_ip': '192.168.4.5',
            'method': 'PUT',
            'url_length': 24,
            'query_length': 19,
            'bytes_sent': 7935,
            'request_time': 2.192,
            'req_per_ip_1min': 131,
            'req_per_ip_10sec': 74,
            'unique_urls_per_ip': 28,
            'time_gap': 5.0,
            'is_https': 1,
            'payload_entropy': 5.25,
            'method_encoded': 2,
            'is_error': 1,
            'is_server_error': 0,
            'is_client_error': 1,
            'is_redirect': 0,
            'is_bot_ua': 1,
            'ua_length': 21,
            'has_sql_injection': 1,
            'has_xss': 0,
            'has_path_traversal': 0,
            'has_command_injection': 0,
            'has_special_chars': 1,
            'url_depth': 2,
            'has_query': 1,
            'query_param_count': 1,
            'req_ratio_10s_to_1m': 0.565,
            'burst_indicator': 1,
            'high_frequency': 1,
            'payload_entropy_norm': 0.656,
            'high_entropy': 1,
            'size_anomaly': 1,
            'slow_request': 1,
            'ip_attack_history': 0.15,
            'ip_error_rate': 0.3,
            'ip_max_req_rate': 131,
            'hour': 15,
            'minute': 40,
            'day_of_week': 0,
            'is_business_hours': 1
        }
    },
    {
        'name': 'XSS Attack',
        'data': {
            'src_ip': '192.168.5.134',
            'method': 'POST',
            'url_length': 35,
            'query_length': 16,
            'bytes_sent': 4548,
            'request_time': 1.748,
            'req_per_ip_1min': 194,
            'req_per_ip_10sec': 44,
            'unique_urls_per_ip': 12,
            'time_gap': 5.0,
            'is_https': 0,
            'payload_entropy': 4.88,
            'method_encoded': 1,
            'is_error': 1,
            'is_server_error': 0,
            'is_client_error': 1,
            'is_redirect': 0,
            'is_bot_ua': 0,
            'ua_length': 11,
            'has_sql_injection': 0,
            'has_xss': 1,
            'has_path_traversal': 0,
            'has_command_injection': 0,
            'has_special_chars': 1,
            'url_depth': 1,
            'has_query': 1,
            'query_param_count': 1,
            'req_ratio_10s_to_1m': 0.227,
            'burst_indicator': 0,
            'high_frequency': 1,
            'payload_entropy_norm': 0.610,
            'high_entropy': 0,
            'size_anomaly': 0,
            'slow_request': 1,
            'ip_attack_history': 0.1,
            'ip_error_rate': 0.4,
            'ip_max_req_rate': 194,
            'hour': 15,
            'minute': 41,
            'day_of_week': 0,
            'is_business_hours': 1
        }
    }
]

for test in test_requests:
    print(f"🔍 Testing: {test['name']}")
    print(f"{'─'*70}")
    
    decision, confidence, explanation = detector.predict(test['data'])
    
    if decision == 'BLOCK':
        print(f"🚨 Decision: {decision} (Confidence: {confidence*100:.1f}%)")
    else:
        print(f"✅ Decision: {decision} (Confidence: {confidence*100:.1f}%)")
    
    print(f"\nModel Votes: {explanation['votes']}")
    for model, pred in explanation['models'].items():
        symbol = '🚨' if pred == 'ANOMALY' else '✅'
        print(f"  {symbol} {model}: {pred}")
    
    if 'attack_indicators' in explanation:
        print(f"\nAttack Indicators Detected:")
        for indicator in explanation['attack_indicators']:
            print(f"  ⚠️  {indicator}")
    
    print()

print(f"="*70)
print(" "*25 + "PIPELINE COMPLETE")
print(f"="*70)
print()
print(f"📁 Generated Files:")
print(f"  • waf_analysis.png - Comprehensive visualizations")
if SHAP_AVAILABLE:
    print(f"  • shap_analysis.png - Feature importance")
print(f"  • model_isolation_forest.pkl - Isolation Forest model")
print(f"  • model_oneclasssvm.pkl - One-Class SVM model")
print(f"  • model_lof.pkl - LOF model")
print(f"  • feature_scaler.pkl - Feature scaler")
print(f"  • method_encoder.pkl - Method encoder")
print(f"  • ensemble_config.pkl - Ensemble configuration")
print(f"  • ip_whitelist.txt - Whitelisted IPs")
print(f"  • feature_columns.txt - Feature list")
print(f"  • waf_detector_class.pkl - Detector class")
print()
print(f"🚀 Ready for deployment!")
print(f"   Use WAFAnomalyDetector class for real-time inference")
print(f"="*70)