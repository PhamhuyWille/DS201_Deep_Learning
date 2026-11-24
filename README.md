# 🧠 PRACTICE DS201 - DEEP LEARNING FOR DATA SCIENCE

### University of Information Technology (UIT - VNUHCM)

This repository documents my learning journey and hands-on practice in **DS201 - Deep Learning for Data Science**. It focuses on understanding neural networks through
implementation and applying deep learning models to image classification,
Vietnamese sentiment analysis, named entity recognition, and English-to-Vietnamese
machine translation.

---

## 👨‍🏫 Instructor

- **BSc. Nguyễn Thị Hoàng Anh**
- Faculty of Information Science and Engineering, University of Information Technology (UIT - VNUHCM)

---

## 📌 Objectives

- Understand the fundamentals of neural networks and deep learning
- Implement and train models using **PyTorch**
- Apply neural networks to handwritten digit and Vietnamese food image
  classification
- Build recurrent models for Vietnamese sentiment analysis and named entity
  recognition
- Compare LSTM encoder–decoder models with Luong and Bahdanau attention for
  English-to-Vietnamese translation
- Evaluate models using classification metrics, entity-level F1, ROUGE-L, and BLEU

---

## 📂 Project Structure

### 🔹 Lab01 - Simple Neural Networks

- Implement simple neural networks in a Jupyter notebook
- Compare one-layer and three-layer MLP models using SGD
- Dataset: **MNIST**
- Notebook: [`Lab01/24520689.ipynb`](Lab01/24520689.ipynb)

---

### 🔹 Lab02 - Vietnamese Food Classification

- Prepare image data and split the training data into training and validation
  sets
- Train and evaluate image classification models with PyTorch
- Dataset: **VinaFood21**, organized into class folders under `train/` and
  `test/`
- Experiment summary and results table are available in the
  [Lab02 guide](Lab02/README.md)

#### Assignment 01 - LeNet-5

- Compare AvgPool and MaxPool LeNet-5 with the best validation-F1 checkpoint
- Export four test metrics, per-class F1, and a 21-by-21 confusion matrix
- Script: [`Lab02/Assignment01/assignment_01.py`](Lab02/Assignment01/assignment_01.py)
- Usage and outputs: [`Assignment 01 guide`](Lab02/Assignment01/README.md)

#### Assignment 02 - GoogLeNet

- Train a reduced GoogLeNet with three Inception blocks and global average pooling
- Save and reload the best validation-F1 checkpoint
- Report macro F1, precision, recall, and accuracy on the test set
- Script: [`Lab02/Assignment02/assignment02.py`](Lab02/Assignment02/assignment02.py)
- Usage and outputs: [`Assignment 02 guide`](Lab02/Assignment02/README.md)

#### Assignment 03 - ResNet-18

- Train ResNet-18 from scratch with 96×96 or 112×112 images
- Compare training with and without data augmentation
- Export training curves, per-class metrics, and confusion matrices
- The implementation and detailed guide are available in
  [Assignment 03](Lab02/Assignment03/README.md)

#### Assignment 04 - Transfer Learning with ResNet-50

- Use the pretrained Hugging Face model `microsoft/resnet-50` with 224×224 images
- Compare feature extraction with full fine-tuning for 21 food classes
- Export experiment reports and a cross-model comparison table
- The implementation and detailed guide are available in
  [Assignment 04](Lab02/Assignment04/README.md)

Shared image preprocessing for Assignments 01 and 02 is implemented in
[`Lab02/data_preprocessing.py`](Lab02/data_preprocessing.py).

---

### 🔹 Lab03 - LSTM, GRU, and BiLSTM

- Train five-layer LSTM and GRU models for three-class sentiment classification
  on **UIT-VSFC**
- Compare sentiment models with a TF-IDF + Logistic Regression baseline
- Train a five-layer BiLSTM for named entity recognition on **PhoNER**
- Use 300-dimensional embeddings, hidden size 256, dropout 0.3, and Adam for
  10 epochs, selecting checkpoints by dev F1
- Report sentiment macro-F1 and NER entity micro-F1 with strict IOB2 evaluation
- The notebook includes training curves, per-class reports, and error analysis.
  [Lab03 notebook](Lab03/24520689.ipynb)

---

### 🔹 Lab04 - Encoder–Decoder LSTM and Attention

- Build English-to-Vietnamese translation models with a three-layer LSTM encoder
  and decoder
- Compare three configurations: no attention (`none`), Luong global-general
  attention (`luong`), and Bahdanau additive attention (`bahdanau`)
- Normalize text with Unicode NFC, lowercase, and whitespace tokenization;
  keep sentence pairs with 1–50 tokens per language and build separate source
  and target vocabularies from the filtered training set
- Select checkpoints by dev ROUGE-L F1, breaking ties with dev negative
  log-likelihood; reload the selected weights before test evaluation
- Evaluate greedy translations using mean sentence ROUGE-L F1 and corpus BLEU
  with SacreBLEU (`tokenize='none'`)
- Training script: [`Lab04/src/main.py`](Lab04/src/main.py)
- Notebooks: [Assignments](Lab04/assignments.ipynb) ·
  [Demo](Lab04/Lab4_Demo.ipynb)
- Results and analysis: [Lab04 report](Lab04/RESULTS.md) ·
  [Training log](Lab04/results/results.log) ·
  [Learning curves](Lab04/results/learning_curves.png)

---

## 📊 Recorded Results

The classification and NER values below come from the saved Lab02 reports and
Lab03 notebook outputs.

| Lab   | Task                     | Model                        | Test accuracy | Test F1 | F1 averaging              |
| ----- | ------------------------ | ---------------------------- | ------------: | ------: | ------------------------- |
| Lab02 | Food classification      | ResNet-50 fine-tuning        |        82.43% |  0.8251 | Macro                     |
| Lab03 | Sentiment classification | TF-IDF + Logistic Regression |        88.98% |  0.6483 | Macro                     |
| Lab03 | Sentiment classification | LSTM                         |        89.99% |  0.7700 | Macro                     |
| Lab03 | Sentiment classification | GRU                          |        90.37% |  0.7650 | Macro                     |
| Lab03 | Named entity recognition | BiLSTM                       |             — |  0.8930 | Entity micro, strict IOB2 |

The complete Lab02 comparison and Lab03 experiment outputs are available in the
[Lab02 results summary](Lab02/README.md) and
[Lab03 notebook](Lab03/24520689.ipynb).

### Lab04 translation results

All three models completed 10 epochs in the recorded run, with epoch 10 selected
for each model. ROUGE-L F1 uses a 0–1 scale; BLEU uses a 0–100 scale.

| Model | Dev ROUGE-L F1 | Test ROUGE-L F1 | Test BLEU |
| ----- | ------------: | -------------: | --------: |
| LSTM without attention | 0.2308 | 0.2350 | 2.6231 |
| LSTM + Luong attention | **0.3021** | **0.3164** | **6.0928** |
| LSTM + Bahdanau attention | 0.2988 | 0.3128 | 5.5817 |

Luong performed best in this recorded run. The small gap from Bahdanau has not
been evaluated across multiple seeds. Translation samples and Bahdanau attention
heatmaps remain to be added to the analysis. See the
[Lab04 report](Lab04/RESULTS.md) for evaluation details and limitations.

---

## ⚙️ Tech Stack

- Python
- PyTorch and torchvision
- NumPy, pandas, and Matplotlib
- Scikit-learn
- Hugging Face Transformers
- Seqeval
- ROUGE Score and SacreBLEU
- JupyterLab

---

## 🚀 Getting Started

Use Python 3.12 or newer for the current Lab04 source, which reuses quote
characters inside f-string expressions. Install the packages used across the labs
in your Python environment.

```powershell
python -m pip install torch torchvision scikit-learn numpy pandas matplotlib tqdm transformers seqeval rouge-score sacrebleu jupyterlab
```

### Lab01 and Lab03 notebooks

Open PowerShell in the repository root and launch JupyterLab.

```powershell
python -m jupyterlab
```

Open `Lab01/24520689.ipynb` or `Lab03/24520689.ipynb` and run the cells in order.
For Lab03, use `Lab03` as the kernel's working directory so the relative `data/`
and `artifacts/` paths resolve correctly. Prepare the JSON Lines files using
this layout.

```text
Lab03/
├── 24520689.ipynb
├── data/
│   ├── uit-vsfc/
│   │   ├── UIT-VSFC-train.json
│   │   ├── UIT-VSFC-dev.json
│   │   └── UIT-VSFC-test.json
│   └── phoner/
│       ├── train_syllable.json
│       ├── dev_syllable.json
│       └── test_syllable.json
└── artifacts/
```

Lab03 saves the selected model weights as `lstm.pt`, `gru.pt`, and
`bilstm_ner.pt`, and exports `summary_results.csv` under `Lab03/artifacts/`.

### Lab02 training scripts

Place VinaFood21 images in `Lab02/dataset/train/<class>/` and
`Lab02/dataset/test/<class>/`. From the repository root, run the following
commands in PowerShell. Each script configures its module path automatically.

```powershell
python Lab02\Assignment01\assignment_01.py --epochs 10 --pool avg
python Lab02\Assignment01\assignment_01.py --epochs 10 --pool max
python Lab02\Assignment02\assignment02.py --epochs 10
python Lab02\Assignment03\assignment_03.py --epochs 10 --mode no_aug
python Lab02\Assignment03\assignment_03.py --epochs 10 --mode aug
```

Use `--train-dir` and `--test-dir` to override the default dataset paths.
The training scripts select CUDA when available and otherwise use the CPU.
Each assignment writes checkpoints and reports under its own `results/`
directory, with separate subdirectories for experiment modes.

Assignment 04 currently accesses `args.num_workers` without declaring the
corresponding command-line argument, causing an `AttributeError` before
training. Once that issue is fixed, its entry point supports
`--mode feature_extraction`, `--mode fine_tuning`, and `--mode both`. The initial
run downloads pretrained weights and preprocessing configuration from Hugging
Face. The current status and recorded results are documented in the
[Lab02 guide](Lab02/README.md).

### Lab04 translation experiments

Prepare the three parallel-text files with this layout. Each file must contain a
JSON array of objects with `english` and `vietnamese` string fields.

```text
Lab04/
├── data/
│   ├── small-train.json
│   ├── small-dev.json
│   └── small-test.json
├── src/
│   ├── main.py
│   ├── preprocessing.py
│   ├── model.py
│   ├── engine.py
│   └── plotting.py
└── results/
```

Run the following command from the repository root in PowerShell. It trains all
three configurations sequentially and selects CUDA when available, otherwise CPU.

```powershell
python Lab04\src\main.py --epochs 10 --batch-size 32 --output-dir Lab04\results --checkpoint Lab04\checkpoint
```

The default model uses 256-dimensional embeddings, hidden size 256, three LSTM
layers, dropout 0.3, Adam with learning rate 0.001, and seed 42. Use `--train-dir`,
`--dev-dir`, and `--test-dir` to override the input paths.

The command writes `none.pt`, `luong.pt`, and `bahdanau.pt` under
`Lab04/checkpoint/`, and `learning_curves.png` under `Lab04/results/`. Metrics are
printed to the console; the script does not automatically create `results.log`.
Without the explicit output arguments above, checkpoints and plots default to
`Lab04/src/checkpoint/` and `Lab04/src/results/` respectively. Dataset directories
and model weight files are excluded from version control by `.gitignore`.

---

## ✍️ Author

- Full name: **Phạm Đình Quang Huy**
- Data Science @ UIT
- Email: 24520689@gm.uit.edu.vn

---

## ⭐ Notes

This repository is for learning and practice. The Lab02 dataset is VinaFood21,
introduced in the paper _VinaFood21: A Novel Dataset for Evaluating Vietnamese
Food Recognition_ (RIVF 2021). See the
[dataset documentation](https://github.com/nguyenvd-uit/uit-together-dataset/blob/main/VinaFood21.md)
for details and citation requirements.
