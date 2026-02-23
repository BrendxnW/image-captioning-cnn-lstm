import json
import csv
from pathlib import Path

# paths (adjust to your folder structure)
coco_ann = "data/COCO/annotations_trainval2017/annotations/captions_val2017.json"
images_dir = "data/COCO/val2017"  # folder where images live
out_csv = "coco_val.csv"

with open(coco_ann, "r", encoding="utf-8") as f:
    data = json.load(f)

# map image_id -> filename
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