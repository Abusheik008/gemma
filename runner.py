import time
import torch
from PIL import Image
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor, BitsAndBytesConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_id = "model"
model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16)
processor = PaliGemmaProcessor.from_pretrained(model_id)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
model_4bit = PaliGemmaForConditionalGeneration.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)

def generate_caption(prompt, image_path):
    raw_image = Image.open(image_path)
    inputs = processor(prompt, raw_image, return_tensors="pt").to(device)
    output = model_4bit.generate(**inputs, max_new_tokens=20)
    return processor.decode(output[0], skip_special_tokens=True)[len(prompt):]

def main():
    start_time = time.time()

    prompt = "MORE_DETAILED_CAPTION"
    image_file = "/content/drive/MyDrive/gemma/gemma/Beam_data/test1.jpeg"
    caption = generate_caption(prompt, image_file)

    print(caption)

    elapsed_time = time.time() - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    if minutes > 0:
        print(f"{int(minutes)} minutes and {int(seconds)} seconds")
    else:
        print(f"{int(seconds)} seconds")

if __name__ == "__main__":
    main()
