
import torch
import torchvision.models as models

NUM_CLASSES = 6

model = models.resnet18()

in_features = model.fc.in_features  # 512

model.fc = torch.nn.Sequential(
    torch.nn.Dropout(0.4),
    torch.nn.Linear(in_features, 256),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.2),
    torch.nn.Linear(256, NUM_CLASSES)
)

checkpoint = torch.load("trashnet_resnet18.pt", map_location="cpu")

model.load_state_dict(checkpoint["model_state_dict"])

model.eval()

dummy = torch.randn(1,3,224,224)

torch.onnx.export(
    model,
    dummy,
    "outputs/trashnet_resnet18.onnx",
    input_names=["image"],
    output_names=["logits"],
    dynamic_axes={
        "image": {0: "batch"},
        "logits": {0: "batch"}
    },
    opset_version=13
)

print("ONNX export successful")