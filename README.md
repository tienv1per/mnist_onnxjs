Run Pytorch model in browser with Javascript by converting it to ONNX form and then loading that ONNX model in the browser using onnx.js. This is the demo of handwritten digit recognition model trained on MNIST dataset.</br>

## Installation
### Clone this repo
```console
git clone https://github.com/tienv1per/mnist_onnxjs.git
```

### Install Dependencies
```console
cd mnist_onnxjs
pip install -r requirements.txt
```

## Usage
Run 
```console
python3 -m http.server
```
with pretrained onnx model onnx_model_onnx.</br></br>
If you want to train the model from scratch, run
```console
python3 train.py
```
And then convert it to ONNX format by running
```console
python3 pt_to_onnx.py
```
This will dump `onnx_model.onnx` file. And then you could use that model to make prediction 

### Notes
This is just a simple demo by converting Pytorch models to ONNX format. If you want to improve the accuracy of this, then modified the architecture of the model, make it more sophisticated and then you will get better result.</br>
If you see this repo helpful for you, give me a star :v.