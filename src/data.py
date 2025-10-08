import os
import io
import csv
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

import pandas as pd
from PIL import Image
from datasets import Dataset, DatasetDict, Features, ClassLabel, Value, Image as HFImage

@dataclass
class ManifestColumns:
    image_path: str = "image_path"  # absolute or repo-relative path to image
    label: str = "label"            # class string, e.g., "Dress", "T-Shirt"
    xmin: str = "xmin"              # optional
    ymin: str = "ymin"
    xmax: str = "xmax"
    ymax: str = "ymax"

def _crop_if_bbox(img: Image.Image, row: pd.Series, cols: ManifestColumns) -> Image.Image:
    # Only crop if all bbox columns exist and are not NA
    needed = [cols.xmin, cols.ymin, cols.xmax, cols.ymax]
    if all(c in row and pd.notna(row[c]) for c in needed):
        x1, y1, x2, y2 = map(int, [row[cols.xmin], row[cols.ymin], row[cols.xmax], row[cols.ymax]])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = max(x1 + 1, x2), max(y1 + 1, y2)
        return img.crop((x1, y1, x2, y2))
    return img

def load_from_manifest_csv(
    csv_path: str,
    root_dir: Optional[str] = None,
    val_split: float = 0.2,
    seed: int = 42,
    crop_with_bbox: bool = True,
    label_map: Optional[List[str]] = None,
    cols: ManifestColumns = ManifestColumns()
) -> DatasetDict:
    """
    Load a dataset from a CSV manifest compatible with DeepFashion2 subsets.

    CSV required columns:
      - image_path  (absolute or relative to root_dir)
      - label       (class name string)
    Optional columns for cropping:
      - xmin, ymin, xmax, ymax

    Example CSV row:
      image_path,label,xmin,ymin,xmax,ymax
      images/img_001.jpg,Dress,34,12,210,300

    Args:
      csv_path: Path to manifest CSV.
      root_dir: If image_path is relative, it will be joined with root_dir.
      val_split: Fraction for validation split.
      crop_with_bbox: If True and bbox present, crop around bbox.
      label_map: Optional list of class names to lock label space/order.

    Returns:
      DatasetDict with 'train' and 'validation' splits and HF Image feature.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Normalize image paths
    def resolve_path(p: str) -> str:
        p = str(p)
        return p if os.path.isabs(p) or root_dir is None else os.path.join(root_dir, p)

    df[cols.image_path] = df[cols.image_path].map(resolve_path)

    # Build label space
    classes = label_map or sorted(df[cols.label].unique().tolist())
    class_to_id = {c: i for i, c in enumerate(classes)}

    # Build HF features
    features = Features({
        "image": HFImage(),
        "label": ClassLabel(names=classes),
        cols.image_path: Value("string")
    })
    for opt_col in [cols.xmin, cols.ymin, cols.xmax, cols.ymax]:
        if opt_col in df.columns:
            features[opt_col] = Value("int64")

    # Load images lazily & crop if requested
    def gen():
        for _, row in df.iterrows():
            path = row[cols.image_path]
            with Image.open(path) as img:
                img = img.convert("RGB")
                if crop_with_bbox:
                    img = _crop_if_bbox(img, row, cols)
                yield {
                    "image": img,
                    "label": class_to_id[row[cols.label]],
                    cols.image_path: path,
                    **({cols.xmin: int(row.get(cols.xmin, 0))} if cols.xmin in df.columns else {}),
                    **({cols.ymin: int(row.get(cols.ymin, 0))} if cols.ymin in df.columns else {}),
                    **({cols.xmax: int(row.get(cols.xmax, 0))} if cols.xmax in df.columns else {}),
                    **({cols.ymax: int(row.get(cols.ymax, 0))} if cols.ymax in df.columns else {}),
                }

    full = Dataset.from_generator(gen, features=features)

    # Split
    full = full.shuffle(seed=seed)
    split = full.train_test_split(test_size=val_split, seed=seed)
    return DatasetDict({"train": split["train"], "validation": split["test"]})
