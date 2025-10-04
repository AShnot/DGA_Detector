# TinyBERT for DGA Detection

Train a compact TinyBERT classifier to detect DGA vs benign domains on large-scale JSONL data.

## Data Format

JSONL with keys `domain`, `threat`:

```
{"domain": "lespedi", "threat": "benign"}
{"domain": "qgcyquqgygcsaausi", "threat": "dga"}
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --no-deps -r dga_tinybert/requirements.txt
# If tokenizers fails on Python 3.13, training uses slow tokenizer via use_fast=False
pip install --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple torch torchvision torchaudio
pip install 'dill==0.3.8' 'huggingface-hub==0.35.3' 'packaging==25.0' regex 'requests==2.32.5' 'safetensors==0.6.2' 'aiohttp==3.12.15' 'xxhash==3.6.0' 'multiprocess==0.70.16' 'pyarrow==21.0.0' 'psutil==7.1.0' 'annotated-types==0.7.0' 'pydantic-core==2.23.4' 'fsspec[http]==2024.6.1'
```

## Training

```bash
python dga_tinybert/train.py \
  --train_file data/dga.jsonl \
  --output_dir outputs/tinybert-dga \
  --model_name prajwal1/bert-tiny \
  --batch_size 256 \
  --num_epochs 2 \
  --fp16
```

Options:
- `--eval_file` to pass a separate eval file; otherwise a stratified split is used.
- `--max_length` default 64; domains are short so 32–64 works.
- `--num_proc` to speed up preprocessing.

## Inference

```bash
python dga_tinybert/predict.py --model_dir outputs/tinybert-dga --text lespedi qgcyquqgygcsaausi
```

Output per line:

```
{"domain": "lespedi", "label": "benign", "confidence": 0.993}
{"domain": "qgcyquqgygcsaausi", "label": "dga", "confidence": 0.987}
```

Or from file:

```bash
python dga_tinybert/predict.py --model_dir outputs/tinybert-dga --input data/sample.jsonl
```

## Notes for 16M samples
## ExtraHop Dataset

Download the dataset and convert to JSONL:

```bash
git clone https://github.com/ExtraHop/DGA-Detection-Training-Dataset.git data/extrahop

# Convert recursively; writes {domain, threat} JSONL
python dga_tinybert/convert_extrahop.py \
  --input_dir data/extrahop \
  --output_jsonl data/dga_extrahop.jsonl \
  --shuffle

# Train on the converted file
python dga_tinybert/train.py \
  --train_file data/dga_extrahop.jsonl \
  --output_dir outputs/tinybert-dga-extrahop \
  --model_name prajwal1/bert-tiny \
  --batch_size 256 \
  --num_epochs 2 \
  --fp16
```

The converter scans `.csv`, `.txt` files and normalizes labels to `benign` or `dga`, validates domain format, removes duplicates, and optionally shuffles.
## ExtraHop dataset usage

- Clone the dataset repo:
  ```bash
  git clone https://github.com/ExtraHop/DGA-Detection-Training-Dataset.git data/extrahop
  ```

- Convert to our JSONL (`domain`, `threat`) using the converter:
  ```bash
  python dga_tinybert/prepare_extrahop.py \
    --input_root data/extrahop \
    --output data/dga.jsonl
  ```

  Options:
  - `--dedup` to remove duplicates (uses memory).
  - You can pass explicit files by label:
    ```bash
    python dga_tinybert/prepare_extrahop.py \
      --dga data/extrahop/path/to/dga.csv \
      --benign data/extrahop/path/to/benign.csv \
      --output data/dga.jsonl
    ```

- Train using the produced file:
  ```bash
  python dga_tinybert/train.py --train_file data/dga.jsonl --output_dir outputs/tinybert-dga --model_name prajwal1/bert-tiny --batch_size 256 --num_epochs 2 --fp16
  ```

### Обучение напрямую из JSON.gz

Если у вас сжатый JSONL (`.json.gz`) формата `{"domain": ..., "threat": ...}`, используйте встроенный загрузчик:

```bash
python dga_tinybert/train.py \
  --train_file data/dga.json.gz \
  --output_dir outputs/tinybert-dga \
  --model_name prajwal1/bert-tiny \
  --batch_size 256 \
  --num_epochs 2 \
  --max_samples 5000000   # опционально ограничить число образцов
```

Примечание: загрузчик `.gz` сразу создаёт колонки `domain` и `labels` (0=benign, 1=dga), поэтому нормализация меток и маппинг в скрипте пропускаются для этого пути.
- Prefer `--batch_size 512`+ with gradient accumulation if VRAM limited.
- Use `--num_proc` for dataset mapping and `--bf16/--fp16` for mixed precision.
- Consider sharded training data and `--save_steps` large (e.g., 50k).
- Set `WANDB_DISABLED=true` or `report_to none` to reduce overhead.
- If CPU RAM is a bottleneck, stream with `load_dataset(..., streaming=True)` and switch to manual trainer loop.
 - Add `--streaming` flag to enable HuggingFace streaming on huge JSONL without loading all to RAM.
