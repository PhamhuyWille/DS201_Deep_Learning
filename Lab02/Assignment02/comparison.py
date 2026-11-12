"""Compare measured GoogLeNet results with available LeNet runs."""

import json
from pathlib import Path
from Assignment01.reporting import write_csv


def save_comparison(output_dir, lenet_dir, summary, config):
    fields = ['model', 'parameters', 'seconds_per_epoch', 'accuracy', 'f1_macro', 'status']
    rows = []
    for pool in ['avg', 'max']:
        path = lenet_dir / pool / 'metrics.json'
        row = dict.fromkeys(fields, '')
        row.update(model=f'LeNet-5 ({pool})', parameters=339421, status='Pending run')
        if path.is_file():
            saved = json.loads(path.read_text(encoding='utf-8'))
            row.update({k: saved['summary'][k] for k in fields if k in saved['summary']})
            other = saved.get('config', {})
            differences = [k for k in ['seed', 'epochs', 'batch_size', 'lr']
                           if other.get(k) != config.get(k)]
            differences += [k for k in ['train_dir', 'test_dir']
                            if Path(other.get(k, '')).resolve() != Path(config[k]).resolve()]
            row['status'] = 'Different config: ' + ', '.join(differences) if differences else 'Matching recorded config'
        rows.append(row)
    rows.append({k: summary[k] for k in fields if k in summary} | {'status': 'Current run'})
    write_csv(output_dir / 'comparison.csv', rows, fields)
    with (output_dir / 'comparison.md').open('w', encoding='utf-8') as handle:
        handle.write('| Model | Parameters | Seconds/epoch | Accuracy | Macro-F1 | Status |\n')
        handle.write('| --- | ---: | ---: | ---: | ---: | --- |\n')
        for row in rows:
            values = [f'{row[k]:.4f}' if isinstance(row[k], float) else str(row[k]) for k in fields]
            handle.write('| ' + ' | '.join(values) + ' |\n')
        handle.write('\nTimes include train and validation, excluding checkpoint saving. '
                     'Compare runs on the same hardware and with the same preprocessing. '
                     'Matching recorded config does not verify unchanged data files or transforms.\n')
