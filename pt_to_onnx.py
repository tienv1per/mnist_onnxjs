import torch

from inference_model import Net

def main():
    pytorch_model = Net()
    pytorch_model.load_state_dict(torch.load("pt_model.pt"))
    pytorch_model.eval()
    dummy_input = torch.zeros(280 * 280 * 4)
    torch.onnx.export(pytorch_model, dummy_input, "onnx_model.onnx", verbose=True, dynamic_axes={'input' : {0 : 'batch_size'}, 'output' : {0 : 'batch_size'}})


if __name__ == "__main__":
    main()
