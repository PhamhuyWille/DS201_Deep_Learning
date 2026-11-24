from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator


def plot_learning_curves(histories, output_dir):
    """Save train loss and dev ROUGE-L curves without opening a GUI window."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    figure = Figure(figsize=(11, 4))
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 2)

    labels = {
        'none': 'LSTM without attention',
        'luong': 'Luong attention',
        'bahdanau': 'Bahdanau attention',
    }
    for mode, history in histories.items():
        epochs = [record['epoch'] for record in history]
        label = labels.get(mode, mode)
        axes[0].plot(epochs, [r['train_loss'] for r in history],
                     marker='o', label=label)
        axes[1].plot(epochs, [r['rougeL_f1'] for r in history],
                     marker='o', label=label)

    axes[0].set_title('Train loss per token')
    axes[0].set_ylabel('Cross-entropy loss')
    axes[1].set_title('Dev ROUGE-L F1 (greedy)')
    axes[1].set_ylabel('ROUGE-L F1')
    for axis in axes:
        axis.set_xlabel('Epoch')
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.grid(alpha=0.3)
        axis.legend()

    figure.tight_layout()
    output_path = output_dir / 'learning_curves.png'
    figure.savefig(output_path, dpi=180)
    return output_path
