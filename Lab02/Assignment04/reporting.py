"""Compatible with Bài 1's save_results/save_comparison format."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix


def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_results(run_dir, classes, scores, history, layers, summary, config, title=None):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    title = title or f'ResNet-50 {summary["mode"]}'
    labels = list(range(len(classes)))
    report = classification_report(scores['y_true'], scores['y_pred'], labels=labels,
                                   target_names=classes, output_dict=True, zero_division=0)
    per_class = [{'class': cls, **report[cls]} for cls in classes]
    write_csv(run_dir / 'per_class.csv', per_class,
              ['class', 'precision', 'recall', 'f1-score', 'support'])
    write_csv(run_dir / 'history.csv', history, list(history[0]))
    write_csv(run_dir / 'layers.csv',
              [dict(zip(['layer', 'output', 'parameters'], row)) for row in layers],
              ['layer', 'output', 'parameters'])
    with (run_dir / 'metrics.json').open('w', encoding='utf-8') as f:
        json.dump({'summary': summary, 'config': config, 'classes': classes,
                   'classification_report': report}, f, indent=2, ensure_ascii=False)
    cm = confusion_matrix(scores['y_true'], scores['y_pred'], labels=labels)
    with (run_dir / 'confusion_matrix.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['true / predicted', *classes])
        w.writerows([name, *row] for name, row in zip(classes, cm.tolist()))
    fig, ax = plt.subplots(figsize=(15, 13))
    image = ax.imshow(cm, cmap='Blues')
    fig.colorbar(image, ax=ax)
    ax.set(xticks=labels, yticks=labels, xticklabels=classes, yticklabels=classes,
           xlabel='Predicted label', ylabel='True label', title=f'{title} - test confusion matrix')
    plt.setp(ax.get_xticklabels(), rotation=90)
    threshold = cm.max()/2
    for i in labels:
        for j in labels:
            ax.text(j, i, str(cm[i,j]), ha='center', va='center', fontsize=6,
                    color='white' if cm[i,j] > threshold else 'black')
    fig.tight_layout()
    fig.savefig(run_dir / 'confusion_matrix.png', dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots()
    ax.plot([h['epoch'] for h in history], [h['train_loss'] for h in history], label='Train')
    ax.plot([h['epoch'] for h in history], [h['val_loss'] for h in history], label='Validation')
    ax.set(xlabel='Epoch', ylabel='Cross-entropy loss', title=title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / 'loss.png', dpi=150)
    plt.close(fig)
    print(classification_report(scores['y_true'], scores['y_pred'], labels=labels,
                                target_names=classes, digits=4, zero_division=0))


def save_comparison(output_dir, summaries):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = ['mode', 'parameters', 'trainable_parameters', 'lr', 'best_epoch',
              'best_val_f1', 'seconds_per_epoch', 'accuracy', 'precision_macro',
              'recall_macro', 'f1_macro']
    write_csv(output_dir / 'comparison.csv', summaries, fields)
    with (output_dir / 'comparison.md').open('w', encoding='utf-8') as f:
        f.write('| Mode | Parameters | Trainable | LR | Best epoch | Seconds/epoch | Accuracy | Precision macro | Recall macro | F1 macro |\n')
        f.write('| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n')
        for x in summaries:
            f.write(f'| {x["mode"]} | {x["parameters"]:,} | {x["trainable_parameters"]:,} | {x["lr"]:.0e} '
                    f'| {x["best_epoch"]} | {x["seconds_per_epoch"]:.2f} | {x["accuracy"]:.4f} '
                    f'| {x["precision_macro"]:.4f} | {x["recall_macro"]:.4f} | {x["f1_macro"]:.4f} |\n')


def save_four_model_table(output_dir, lab_dir, summaries):
    """Fill results found on disk; leave unknown values empty rather than fabricate."""
    import json
    paths = [('LeNet-5 (Bài 1)', lab_dir/'Assignment01/results/avg/metrics.json'),
             ('GoogLeNet (Bài 2)', lab_dir/'Assignment02/results/metrics.json'),
             ('ResNet-18 (Bài 3)', lab_dir/'Assignment03/results/no_aug/metrics.json')]
    rows = []
    for name, path in paths:
        result = {}
        if path.is_file():
            try: result = json.loads(path.read_text(encoding='utf-8')).get('summary', {})
            except (OSError, ValueError): pass
        rows.append({'model': name, 'parameters': result.get('parameters', ''),
                     'accuracy': result.get('accuracy', ''), 'precision_macro': result.get('precision_macro', ''),
                     'recall_macro': result.get('recall_macro', ''), 'f1_macro': result.get('f1_macro', '')})
    for x in summaries:
        rows.append({'model': f'ResNet-50 ({x["mode"]})', 'parameters': x['parameters'],
                     **{k: x[k] for k in ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']}})
    write_csv(Path(output_dir)/'four_models.csv', rows, list(rows[0]))
