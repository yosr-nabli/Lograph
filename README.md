# 🚀 Running the Experiment

This project uses a **Conda-based reproducible environment**  
(Python 3.12, PyTorch 2.7.0, scikit-learn 1.7.2).

---

## 1️⃣ Environment Setup (Required)

### Create the Conda Environment

From the project root directory, run:

```bash
conda env create -f environment.yml
```
This will create a Conda environment named:masterthesis
Step 2: Activate the Environment
```bash
conda activate masterthesis
```
Step 3: Verify Installation (Optional but Recommended)
```bash
python --version
python -c "import torch; print(torch.__version__)"
python -c "import sklearn; print(sklearn.__version__)"
```

Expected versions:
```bash

Python 3.12

PyTorch 2.7.0

scikit-learn 1.7.2
```

📊 2. Download the Dataset

Download public log datasets from:
```bash
https://github.com/logpai/loghub
```

Place the downloaded datasets inside the project's:
```bash
data/
```

directory.

⚙️ 3. Configure the Experiment

Edit the configuration file:
```bash
config.py
```

This file stores the experimental settings, including:

-Dataset paths

--Model paths

Training schemes

-Hyperparameters

-Grouping strategies

-Adjust these settings according to your experimental setup.

🧠 4. Run Training

To start training the model:
```bash
python main.py
```