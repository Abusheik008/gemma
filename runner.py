import time
import torch
import torch.onnx
from PIL import Image
from torch2trt import TRTModule, torch2trt
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_id = "model"

print("Loading the pretrained model")
model = PaliGemmaForConditionalGeneration.from_pretrained(model_id)
model.eval().to(device)

# Prepare dummy inputs for ONNX export
# Ensure the dummy input matches the expected input format and type
dummy_input_ids = torch.randint(0, 30522, (1, 128)).to(device)  # Replace 30522 with the vocab size
dummy_attention_mask = torch.ones_like(dummy_input_ids).to(device)

onnx_path = "model.onnx"
print("Exporting the model to ONNX format")
torch.onnx.export(
    model, 
    (dummy_input_ids, dummy_attention_mask), 
    onnx_path, 
    input_names=["input_ids", "attention_mask"], 
    output_names=["output"],
    opset_version=11
)

# Load the ONNX model
import onnx
print("Loading the ONNX model")
onnx_model = onnx.load(onnx_path)

# Convert the ONNX model to TensorRT
print("Converting the ONNX model to TensorRT")
model_trt = torch2trt(model, [dummy_input_ids, dummy_attention_mask], max_workspace_size=1<<25, fp16_mode=True)

# Save the TensorRT model
trt_path = 'model_trt.pth'
print("Saving the TensorRT model")
torch.save(model_trt.state_dict(), trt_path)

# Load the TensorRT model
print("Loading the TensorRT model")
model_trt = TRTModule()
model_trt.load_state_dict(torch.load(trt_path))
model_trt.to(device)

print("Loading the processor")
processor = PaliGemmaProcessor.from_pretrained(model_id)

def generate_caption(prompt, image_path):
    raw_image = Image.open(image_path)
    inputs = processor(prompt, raw_image, return_tensors="pt").to(device)
    
    start_time = time.time()
    with torch.no_grad():  # Disable gradient calculation
        output = model_trt.generate(**inputs, max_new_tokens=300)
    elapsed_time = time.time() - start_time

    print(elapsed_time, "inference seconds")
    return processor.decode(output[0], skip_special_tokens=True)[len(prompt):]

def main():
    prompt = "MORE_DETAILED_CAPTION"
    image_file = "test1.jpeg"
    
    start_time = time.time()
    caption = generate_caption(prompt, image_file)
    elapsed_time = time.time() - start_time

    print(caption)
    print(f"{elapsed_time:.2f} seconds")
    minutes, seconds = divmod(elapsed_time, 60)
    if minutes > 0:
        print(f"{int(minutes)} minutes and {int(seconds)} seconds")
    else:
        print(f"{int(seconds)} seconds")

if __name__ == "__main__":
    main()
