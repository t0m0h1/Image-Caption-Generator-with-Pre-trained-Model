from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests
import torch

# Load an image (use your own file or URL)
img_path = "https://example.com/path/to/your/image.jpg"
image = Image.open(requests.get(img_path, stream=True).raw).convert('RGB')

# Load BLIP processor and model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Preprocess and generate caption
inputs = processor(images=image, return_tensors="pt")
with torch.no_grad():
    out = model.generate(**inputs)
caption = processor.decode(out[0], skip_special_tokens=True)

print("Generated Caption:", caption)
