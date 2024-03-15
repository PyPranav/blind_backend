from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
import requests

import time

start_time = time.time()  # Capture start time

# Your code to be timed goes here




url = "chutiya 1.jpeg"
image = Image.open(url)
width, height = image.size

# Calculate the new size (half the width and height)
new_width = width 
new_height = height 

# Resize the image
image1 = image.resize((new_width, new_height))

url = "chutiya 2.jpeg"
image = Image.open(url)
width, height = image.size

# Calculate the new size (half the width and height)
new_width = width 
new_height = height 

# Resize the image
image2 = image.resize((new_width, new_height))

url = "chutiya 3.jpeg"
image = Image.open(url)
width, height = image.size

# Calculate the new size (half the width and height)
new_width = width 
new_height = height 

# Resize the image
image3 = image.resize((new_width, new_height))

url = "chutiya.jpeg"
image = Image.open(url)
width, height = image.size

# Calculate the new size (half the width and height)
new_width = width 
new_height = height 

# Resize the image
image4 = image.resize((new_width, new_height))

# you can specify the revision tag if you don't want the timm dependency
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

for im in [image1,image2, image3, image4]:
    inputs = processor(images=im, return_tensors="pt")
    outputs = model(**inputs)

    # convert outputs (bounding boxes and class logits) to COCO API
    # let's only keep detections with score > 0.9
    target_sizes = torch.tensor([im.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        print(
                f"Detected {model.config.id2label[label.item()]} with confidence "
                f"{round(score.item(), 3)} at location {box}"
            )

end_time = time.time()  # Capture end time

execution_time = end_time - start_time

print(f"Execution time: {execution_time:.2f} seconds")