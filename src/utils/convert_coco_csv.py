import json
import csv
import pandas as pd
from pathlib import Path


coco_ann = "data/COCO/annotations_trainval2014/annotations/captions_train2014.json"
images_dir = "data/COCO/Train/Images/train2014"
out_csv = "coco_train.csv"

with open(coco_ann, "r", encoding="utf-8") as f:
    data = json.load(f)


id_to_file = {img["id"]: img["file_name"] for img in data["images"]}

rows = []
for ann in data["annotations"]:
    img_id = ann["image_id"]
    caption = ann["caption"].strip()
    file_name = id_to_file[img_id]
    img_path = f"{images_dir}/{file_name}"
    rows.append([img_path, caption])

with open(out_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["image", "caption"])
    writer.writerows(rows)

print(f"Saved {len(rows)} rows to {out_csv}")


df = pd.read_csv("data/COCO/Train/coco_train.csv")
df2 = pd.read_csv("data/COCO/Validate/coco_val.csv")


df["image"] = (
    df["image"]
    .astype(str)
    .str.replace(r"\.jpg$", "", regex=True)
    .str.zfill(12)
    .radd("COCO_train2014_")
    .add(".jpg")
)
df2["image"] = df2["image"].str.replace(r"^data/COCO/val2017/", "", regex=True)

df.to_csv("data/COCO/Train/coco_train.csv", index=False)
df2.to_csv("data/COCO/Validate/coco_val.csv", index=False)

csv_in = "data/COCO/Train/coco_train.csv"
csv_out = "data/COCO/Train/coco_train_clean.csv"
root_dir = Path("data/COCO/Train/Images/train2014")

df = pd.read_csv(csv_in)

# 1) normalize column names to what your code expects
rename_map = {}
if "file_name" in df.columns and "image" not in df.columns:
    rename_map["file_name"] = "image"
if "text" in df.columns and "caption" not in df.columns:
    rename_map["text"] = "caption"
if "sentence" in df.columns and "caption" not in df.columns:
    rename_map["sentence"] = "caption"
df = df.rename(columns=rename_map)

# 2) keep only required columns (if extras exist)
required = ["image", "caption"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# 3) drop empty captions/images
df["image"] = df["image"].astype(str).str.strip()
df["caption"] = df["caption"].astype(str).str.strip()
df = df[(df["image"] != "") & (df["caption"] != "")]

# 4) if CSV has numeric ids, build COCO filename format
#    adjust prefix/year as needed for your files
needs_format = df["image"].str.fullmatch(r"\d+(\.0)?").fillna(False)
df.loc[needs_format, "image"] = (
    df.loc[needs_format, "image"]
      .str.replace(".0", "", regex=False)
      .str.zfill(12)
      .radd("COCO_train2014_")
      .add(".jpg")
)

# 5) keep basename only (matches your dataset behavior)
df["image"] = df["image"].apply(lambda p: Path(p).name)

# 6) drop rows whose image file does not exist on disk
exists_mask = df["image"].apply(lambda fn: (root_dir / fn).exists())
missing_count = (~exists_mask).sum()
df_clean = df[exists_mask].copy()

print(f"Original rows: {len(df)}")
print(f"Dropped missing images: {missing_count}")
print(f"Final rows: {len(df_clean)}")

df_clean.to_csv(csv_out, index=False)
print(f"Saved: {csv_out}")