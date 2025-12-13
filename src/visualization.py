import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud


sns.set_style("whitegrid")
custom_params = {
    "text.color": "#003366", "axes.labelcolor": "#003366",
    "xtick.color": "#003366", "ytick.color": "#003366",
    "axes.titlecolor": "#003366", "font.family": "sans-serif"
}
plt.rcParams.update(custom_params)

def create_dist_plot(data, title, color, xlabel):
    """Generates Histogram with KDE (From Phase 1)"""
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_data = [x for x in data if x < 500] 
    sns.histplot(plot_data, ax=ax, kde=True, color=color, bins=30)
    ax.set_title(title, fontweight='bold', fontsize=14)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency")
    plt.tight_layout()
    return fig

def plot_wordcloud(text, title, font_path='arial'):
    """Generates WordCloud (From Phase 2)"""
    wc = WordCloud(width=700, height=350, background_color='white', font_path=font_path, colormap='viridis').generate(text)
    fig = plt.figure(figsize=(9,5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    return fig

def plot_confusion_matrix(cm, classes, title="Confusion Matrix"):
    """Plots Confusion Matrix (From Phase 4)"""
    fig = plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.tight_layout()
    return fig

def plot_roc_curve(fpr, tpr, roc_auc, model_name):
    """Plots ROC Curve"""
    fig = plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend()
    return fig