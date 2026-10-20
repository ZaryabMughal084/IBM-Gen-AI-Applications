import gradio as gr
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

MODEL_NAME = "Salesforce/blip-image-captioning-base"

processor = BlipProcessor.from_pretrained(MODEL_NAME)
model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def caption_image(image):
    if image is None:
        return "Please upload an image."

    inputs = processor(images=image.convert("RGB"), return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=50)

    return processor.decode(output[0], skip_special_tokens=True)

demo = gr.Interface(
    fn=caption_image,
    inputs=gr.Image(type="pil", label="Upload a photo"),
    outputs=gr.Textbox(label="Generated caption"),
    title="AI Image Captioning",
    description="Upload an image to generate a description using the BLIP model.",
)

if __name__ == "__main__":
    demo.launch()
