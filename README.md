# Baseline RecBole

Sequential recommendation models benchmark using [RecBole](https://github.com/RUCAIBox/RecBole) on HPC clusters.

## Project Structure

### Core Files
- **`main.py`** - Training and evaluation entry point
- **`utils.py`** - Environment setup and utility functions
- **`slurm_baseline.sh`** - SLURM job array script (7 models × 3 datasets)
- **`sync_git.sh`** - Git sync with auto-commit

### Configuration
- **`config/overall.yaml`** - Global settings (hyperparameters, metrics, W&B)

### Data Directories
- **`data/`** - Datasets with embeddings and interaction files

## Usage

```bash
# Single experiment
python main.py --model SASRec --dataset Baby_Products --log_wandb

# Batch experiments on cluster
sbatch slurm_baseline.sh

# Sync to git
./sync_git.sh "commit message"
```

## Models & Datasets

**Models:** SASRec, BERT4Rec, FEARec, SINE, CORE, SASRecCPR, TedRec  
**Datasets:** Baby_Products, Industrial_and_Scientific, Office_Products
