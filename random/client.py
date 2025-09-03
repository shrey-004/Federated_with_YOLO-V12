# client.py
import flwr as fl
from ultralytics import YOLO
import torch
from pathlib import Path
from collections import OrderedDict

# ---- Utility: weights conversion ----
def get_weights(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_weights(model, parameters):
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), parameters)})
    model.load_state_dict(state_dict, strict=True)

# ---- Client class ----
class YOLOClient(fl.client.NumPyClient):
    def __init__(self, client_id):
        self.client_id = client_id
        # self.model = YOLO("yolov12n.pt")  # nano version for quick test
        self.model = YOLO("yolov8n.pt")   # nano pretrained

        # self.model = YOLO("yolov12n.yaml")  # start from scratch (nano version config)
        self.dataset = f"coco_yolo/client{client_id}/coco128.yaml"

    def get_parameters(self, config):
        return get_weights(self.model.model)

    def fit(self, parameters, config):
        set_weights(self.model.model, parameters)
        # Train for 1 epoch on this client’s dataset
        self.model.train(data=self.dataset, epochs=1, imgsz=320, device=0)
        return get_weights(self.model.model), 100, {}  # len(dataset) placeholder

    def evaluate(self, parameters, config):
        set_weights(self.model.model, parameters)
        results = self.model.val(data=self.dataset)
        metrics = {"mAP50": results.box.map50, "precision": results.box.mp, "recall": results.box.mr}
        return float(results.box.loss), 50, metrics  # dummy num_examples

# ---- Run client ----
if __name__ == "__main__":
    import sys
    cid = int(sys.argv[1])  # pass client id: 1, 2, or 3
    fl.client.start_numpy_client(
        server_address="0.0.0.0:8080",
        client=YOLOClient(cid)
    )
