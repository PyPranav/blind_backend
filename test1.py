from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
import requests
import time
import threading
import base64
import io

def analyze_image(image_base64, processor, model):
    image_bytes = io.BytesIO(base64.b64decode(image_base64))

    # Load the image from the byte stream
    image = Image.open(image_bytes)
    width, height = image.size
    new_width = width // 2
    new_height = height // 2
    image = image.resize((new_width, new_height))

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        print(
                f"Detected {model.config.id2label[label.item()]} with confidence "
                f"{round(score.item(), 3)} at location {box}"
            )

    # ... (rest of the processing remains the same)

# Main execution with multithreading
start_time = time.time()
image_base64_strings = [
    "iVBORw0KGgoAAAANSUhEUgAAABgAAAAYCAYAAADgdz34AAABwklEQVRIS+2Vz0oDQRSGv8V"]
threads = []
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

for url in image_base64_strings:
    thread = threading.Thread(target=analyze_image, args=(url, processor, model))
    threads.append(thread)
    thread.start()

for thread in threads:
    thread.join()  # Wait for all threads to finish

end_time = time.time()

execution_time = end_time - start_time
print(f"Execution time: {execution_time:.2f} seconds")
