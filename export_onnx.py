import torch
import torch.nn as nn

class Net(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(917,128)
        self.fc2 = nn.Linear(128,2)

        self.beta = 0.9
        self.threshold = 1.0
        self.num_steps = 25

    def forward(self,x):

        batch_size = x.size(0)

        mem1 = torch.zeros(batch_size,128)
        mem2 = torch.zeros(batch_size,2)

        spk_sum = torch.zeros(batch_size,2)

        for _ in range(self.num_steps):

            mem1 = self.beta * mem1 + self.fc1(x)
            spk1 = (mem1 > self.threshold).float()
            mem1 = mem1 * (mem1 <= self.threshold)

            mem2 = self.beta * mem2 + self.fc2(spk1)
            spk2 = (mem2 > self.threshold).float()
            mem2 = mem2 * (mem2 <= self.threshold)

            spk_sum += spk2

        return spk_sum


model = Net()
model.load_state_dict(torch.load("snn_spam_model.pth",map_location="cpu"))
model.eval()

dummy = torch.randn(1,917)

torch.onnx.export(
    model,
    dummy,
    "model.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=11
)

print("ONNX model exported")