import torch

if torch.cuda.is_available():
    device_idx = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(device_idx)
    device_type = "cuda"
else:
    device_idx = 0
    device_name = "cpu"
    device_type = "cpu"

print(f"DEVICE: {device_idx} - {device_name} ({device_type})")