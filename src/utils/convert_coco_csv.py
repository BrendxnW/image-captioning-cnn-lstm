import json
import csv
import pandas as pd
from pathlib import Path


coco_ann = "data/COCO/annotations_trainval2017/annotations/captions_val2017.json"
images_dir = "data/COCO/val2017"
out_csv = "coco_val.csv"

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
    writer.writerow(["image_path", "caption"])
    writer.writerows(rows)

print(f"Saved {len(rows)} rows to {out_csv}")


df = pd.read_csv("data/COCO/Train/coco_train.csv")
df2 = pd.read_csv("data/COCO/Validate/coco_val.csv")


df["image"] = df["image"].str.replace(r"^data/COCO/train2017/", "", regex=True)
df2["image"] = df2["image"].str.replace(r"^data/COCO/val2017/", "", regex=True)

df.to_csv("data/COCO/Train/coco_train.csv", index=False)
df2.to_csv("data/COCO/Validate/coco_val.csv", index=False)