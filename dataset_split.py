# dataset_split.py
# from ultralytics.data.utils import download
from pathlib import Path
import shutil, random, os

def prepare_coco_splits(base_dir="coco_yolo", num_clients=3):
    # 1. Download COCO in YOLO format
    # download("coco128")  # tiny subset of COCO (128 images) for quick tests
    src = Path("datasets/coco128")

    # 2. Copy into client folders
    for client_id in range(num_clients):
        dst = Path(base_dir) / f"client{client_id+1}"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)

    # 3. Simulate Non-IID splits: distribute images unevenly
    all_imgs = list((src / "images/train2017").glob("*.jpg"))
    random.shuffle(all_imgs)
    chunk_size = len(all_imgs) // num_clients
    for client_id in range(num_clients):
        client_imgs = all_imgs[client_id*chunk_size:(client_id+1)*chunk_size]
        client_dir = Path(base_dir) / f"client{client_id+1}"

        # Directories for this client
        client_img_dir = client_dir / "images/train2017"
        client_lbl_dir = client_dir / "labels/train2017"

        # Keep only assigned images/labels
        for img in all_imgs:
            if img not in client_imgs:
                img_file = client_img_dir / img.name
                lbl_file = client_lbl_dir / (img.stem + ".txt")

                if img_file.exists():
                    os.remove(img_file)
                if lbl_file.exists():
                    os.remove(lbl_file)


if __name__ == "__main__":
    prepare_coco_splits()
    print("COCO split into 3 clients inside coco_yolo/")
