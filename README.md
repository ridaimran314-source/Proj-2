# Pakistani Politician Image Classification (Project 2)

Multi-class CNN face classification for Pakistani public figures, self-collected dataset, course-compliant split and evaluation.

The number of classes is defined by your folder names under `dataset/raw/` / `splits/**/train/` (whatever you configure in `configs/classes.yaml`).

## Dataset workflow

1. **Define classes**  
   Edit `configs/classes.yaml` so the 16 `slug` values match the people you will collect (folder names = slugs).

2. **Create empty raw folders**  
   ```bash
   pip install -r requirements.txt
   python scripts/init_raw_folders.py
   ```

3. **Collect images** (Google Images, Wikipedia, official news, government pages)  
   Save face photos under `dataset/raw/<slug>/`.  
   **Minimum 80 images per class** before splitting.

### Automated seed from Wikimedia Commons (optional)

`scripts/download_commons_images.py` pulls **freely licensed** files from [Wikimedia Commons](https://commons.wikimedia.org/) using the official API. Run this on your own PC (respect [User-Agent policy](https://foundation.wikimedia.org/wiki/Policy:User-Agent_policy); use a real contact URL or email):

```bash
set COMMONS_CONTACT_URL=https://github.com/YOUR_ORG/YOUR_REPO
python scripts/download_commons_images.py --classes configs/classes.yaml --max-per-class 120
```

Commons categories are **noisy** (crowds, logos, wrong person). **Delete bad files**, add missing faces, and document sources in your report. Each file on Commons has its own license—keep attribution as required by those licenses.

### Google Images (supported APIs only — do not scrape `google.com`)

Automated **HTML scraping of Google Images is not allowed** by Google’s terms. This repo uses **allowed** APIs instead:

| Provider | Credentials | Notes |
|----------|-------------|--------|
| **google_cse** (default) | `GOOGLE_API_KEY` + `GOOGLE_CSE_ID` | [Custom Search JSON API](https://developers.google.com/custom-search/v1/overview) with `searchType=image`. Create a [Programmable Search Engine](https://programmablesearchengine.google.com/) with **Image search** and search the whole web. |
| **serpapi** | `SERPAPI_API_KEY` | [SerpApi](https://serpapi.com/) Google Images engine (paid plan / limited free searches). |

Per-class search text: optional `google_query` in `configs/classes.yaml`, otherwise `"{display_name} face portrait"`.

```powershell
$env:GOOGLE_API_KEY="your-key"
$env:GOOGLE_CSE_ID="your-cx-id"
python scripts/download_google_images.py --provider google_cse --replace --max-per-class 90
```

`--replace` deletes existing `.jpg/.png/.webp` in each `dataset/raw/<slug>/` (use after backing up anything you want to keep). Then run `verify_counts.py` and `split_dataset.py --clean`.

### Collaborating groups (extra classes)

If instructors approved merged work: **+10 classes per additional group** (e.g. two groups → **26** classes, three → **36**). Use `configs/classes_collab_26.yaml` or `configs/classes_collab_36.yaml` and pass `--classes ...` to `init_raw_folders.py`, `download_commons_images.py`, `verify_counts.py`, and `split_dataset.py`. Edit YAML so the extra people match what you filed with the course.

4. **Verify counts**  
   ```bash
   python scripts/verify_counts.py
   ```

5. **Split 75% / 15% / 10%** (stratified per class; copies files)  
   ```bash
   python scripts/split_dataset.py --clean
   ```

6. **Augmentation (implemented in code)**  
   Training uses rotation, horizontal flip, brightness/contrast/saturation jitter, zoom/crop (`RandomResizedCrop` + `RandomAffine` scale), **only on `splits/train`** (`src/politician_cnn/data.py`). Validation and test use resize + center crop only.

## Training (≥2 pretrained CNNs)

**Data location:** put your **real** images under **`train/`**, **`val/`**, and **`test/`** (one subfolder per class; identical class names in all three).

This repo’s `dvc repro train_models` stage expects the nested layout **`splits/splits/{train,val,test}`** (matches `python scripts/split_dataset.py` when it writes into `splits/splits/`).

If your folders are directly under `splits/{train,val,test}` instead, pass `--data-root splits` when training.

If you maintain `splits/` by hand, **do not** run `python scripts/split_dataset.py --clean` unless you intend to **wipe and rebuild** splits from `dataset/raw/`.

After `splits/train`, `splits/val`, and `splits/test` are ready:

```bash
pip install -r requirements.txt
python scripts/train_models.py --data-root splits/splits --epochs 25 --batch-size 16 --models resnet50 efficientnet_b0
```

**Transfer learning defaults (speed + accuracy on small datasets):** the trainer freezes the pretrained backbone for the first `--backbone-freeze-epochs` (default **8**) to train the classifier head, then unfreezes the full model using `--finetune-lr` (defaults to **0.2 × --lr** if omitted). It also supports inverse-frequency `--class-weights auto` (default) to help imbalanced classes.

**MLflow** is **on by default** (writes to `./mlruns`). Override with `MLFLOW_TRACKING_URI` or `--mlflow-experiment`. CI / smoke: add `--no-mlflow`.

Optional architectures (see `src/politician_cnn/models.py`): `resnet101`, `resnet152`, `efficientnet_b0`–`b4`, `vgg16`, `vgg19`.

**Outputs** (under `artifacts/<architecture>/`):

- `best.pt` — best validation-weights checkpoint  
- `test_metrics.json` — overall accuracy, per-class precision/recall/F1, 90% requirement flag  
- `train_val_curves.png` — loss and accuracy vs epoch  
- `confusion_matrix_heatmap.png` — test confusion matrix  
- `top_misclassified/` — five highest-confidence wrong predictions (copies + `top_misclassified.json`)  

`artifacts/comparison_summary.json` compares all models from one run.

**90% test accuracy** depends on data quality and training budget; tune epochs, learning rate, and add more real face images if you are below the bar.

## Repository hygiene (course)

- Add required **GitHub contributors** from the brief (instructor + TAs) and all group members.  
- Use a **unique, relevant repo name** when you publish.  
- Keep **continuous commits** (data collection, training, frontend, MLOps if Category A).

## Category A (MLOps)

### DVC (dataset / pipeline versioning)

```bash
pip install dvc
dvc init
# Optional: dvc remote add -d myremote s3://your-bucket/dvcstore   # or SSH / GDrive per DVC docs
dvc repro train_models
```

- Edit **`params.yaml`** for epochs, batch size, learning rate.  
- **`dvc.yaml`** defines stage `train_models` (deps: code + `splits/splits/train|val|test`, outs: `artifacts/`, metrics: `comparison_summary.json`).  
- Track large raw data separately, e.g. `dvc add dataset/raw` after populating, then `dvc push`.  
- Commit **`dvc.lock`**, **`dvc.yaml`**, and **`.dvc/config`** (not `.dvc/cache/`).

### MLflow (experiment tracking)

Training logs params, per-epoch metrics, test metrics, artifacts (curves, confusion matrix JSON, checkpoint), and a **`torch_model`** artifact per architecture. Open UI:

```bash
mlflow ui --backend-store-uri file:./mlruns
```

### Airflow (optional orchestration)

- DAG: `airflow/dags/pak_politician_train.py` runs `dvc repro train_models`.  
- Set Airflow Variable **`project2_root`** to the repo path on the worker; install **DVC** + project deps on the worker image.  
- Install Airflow separately (`pip install apache-airflow`) — not bundled in `requirements.txt` to keep the training env light.

### CI/CD (GitHub Actions)

Workflow **`.github/workflows/ci.yml`**: CPU PyTorch smoke tests + **Docker build** of the API image on push/PR to `main`/`master`.

### Docker + EC2 (serving API)

- **Image:** `docker/Dockerfile.api` (FastAPI + Uvicorn, CPU PyTorch).  
- **Local:** `docker compose up --build` (expects `artifacts/resnet50/best.pt`; adjust volume in `docker-compose.yml`).  
- **EC2:** follow **`deploy/EC2.md`** for install, `scp` checkpoint, `docker run`, and security group.

Endpoints: `GET /health`, `POST /predict` (multipart field `file` = image).

## Next steps

- Add a **simple test frontend** that calls `/predict` for rubric demos.
