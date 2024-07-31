import time
import torch
from PIL import Image
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor, BitsAndBytesConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_id = "model"

print("Planning to launch the model...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

print("Loading the pretrained model")

model_4bit = PaliGemmaForConditionalGeneration.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)

print("loading the processor")
processor = PaliGemmaProcessor.from_pretrained(model_id)

def generate_caption(prompt, image_path):
    raw_image = Image.open(image_path)
    inputs = processor(prompt, raw_image, return_tensors="pt").to(device)
    
    start_time = time.time()
    with torch.no_grad():  # Disable gradient calculation
        output = model_4bit.generate(**inputs, max_new_tokens=300)
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
