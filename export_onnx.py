import torch
from predict import Net   # reuse your model architecture

model = Net()

state = torch.load("snn_spam_model.pth", map_location="cpu")
model.load_state_dict(state, strict=False)

model.eval()

dummy_input = torch.randn(1,900)

torch.onnx.export(
    model,
    dummy_input,
    "smartshield_snn.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=11
)

print("ONNX model exported successfully")