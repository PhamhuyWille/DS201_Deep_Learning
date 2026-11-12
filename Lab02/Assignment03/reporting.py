import csv
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix


def write_csv(path, rows, fields):
    with path.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_results(run_dir, classes, scores, history, layers, summary, config, title=None):
    title = title or 'ResNet-18'
    run_dir.mkdir(parents=True, exist_ok=True)
    labels = list(range(len(classes)))
    report = classification_report(scores['y_true'], scores['y_pred'], labels=labels,
                                   target_names=classes, output_dict=True, zero_division=0)
    per_class = [{'class': name, **report[name]} for name in classes]
    write_csv(run_dir / 'per_class.csv', per_class,
              ['class', 'precision', 'recall', 'f1-score', 'support'])
    write_csv(run_dir / 'history.csv', history, list(history[0]))
    write_csv(run_dir / 'layers.csv',
              [dict(zip(['layer', 'output', 'parameters'], row)) for row in layers],
              ['layer', 'output', 'parameters'])
    with (run_dir / 'metrics.json').open('w', encoding='utf-8') as handle:
        json.dump({'summary': summary, 'config': config, 'classes': classes,
                   'classification_report': report}, handle, ensure_ascii=False, indent=2)
    cm = confusion_matrix(scores['y_true'], scores['y_pred'], labels=labels)
    with (run_dir / 'confusion_matrix.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.writer(handle)
        writer.writerow(['true / predicted', *classes])
        writer.writerows([name, *row] for name, row in zip(classes, cm.tolist()))
    fig, ax = plt.subplots(figsize=(15, 13))
    image = ax.imshow(cm, cmap='Blues')
    fig.colorbar(image, ax=ax)
    ax.set(xticks=labels, yticks=labels, xticklabels=classes, yticklabels=classes,
           xlabel='Predicted label', ylabel='True label', title=f'{title} - test confusion matrix')
    plt.setp(ax.get_xticklabels(), rotation=90)
    threshold = cm.max() / 2 if cm.size else 0
    for i in labels:
        for j in labels:
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=6,
                    color='white' if cm[i, j] > threshold else 'black')
    fig.tight_layout()
    fig.savefig(run_dir / 'confusion_matrix.png', dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots()
    epochs = [h['epoch'] for h in history]
    ax.plot(epochs, [h['train_loss'] for h in history], label='Train')
    ax.plot(epochs, [h['val_loss'] for h in history], label='Validation')
    ax.set(xlabel='Epoch', ylabel='Cross-entropy loss', title=title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / 'loss.png', dpi=150)
    plt.close(fig)
    print(classification_report(scores['y_true'], scores['y_pred'], labels=labels,
                                target_names=classes, digits=4, zero_division=0))


def save_comparison(output_dir, summaries):
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / 'comparison.csv', summaries, list(summaries[0]))
    by_mode = {row['augmentation']: row for row in summaries}
    with (output_dir / 'comparison.md').open('w', encoding='utf-8') as handle:
        handle.write('| Augmentation | Params | Best epoch | Seconds/epoch | Accuracy | Precision macro | Recall macro | F1 macro | Loss divergence epoch |\n')
        handle.write('| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n')
        for row in summaries:
            handle.write(f'| {row["augmentation"]} | {row["parameters"]:,} | {row["best_epoch"]} '
                         f'| {row["seconds_per_epoch"]:.2f} | {row["accuracy"]:.4f} '
                         f'| {row["precision_macro"]:.4f} | {row["recall_macro"]:.4f} '
                         f'| {row["f1_macro"]:.4f} | {row["loss_divergence_epoch"] or "N/A"} |\n')
        if 'no_aug' in by_mode and 'aug' in by_mode:
            delta = by_mode['aug']['f1_macro'] - by_mode['no_aug']['f1_macro']
            handle.write(f'\nMacro-F1 difference (augmentation - no augmentation): **{delta:+.4f}**\n')
            handle.write('\nLoss divergence epoch is a heuristic; inspect the loss plots before reporting overfitting.\n')
    if 'no_aug' in by_mode and 'aug' in by_mode:
        print(f"Macro-F1 delta (aug - no_aug): {by_mode['aug']['f1_macro'] - by_mode['no_aug']['f1_macro']:+.4f}")
