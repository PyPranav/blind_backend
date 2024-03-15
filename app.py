from flask import render_template
from flask import Flask, redirect, request, url_for, jsonify
import os
import io
from PIL import Image
import time
import threading
import base64
import torch
import json
from transformers import DetrImageProcessor, DetrForObjectDetection, BlipProcessor, BlipForConditionalGeneration
from openai import OpenAI
import requests




app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY") or os.urandom(24)

processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")


def analyze_image(coord,image, processor, model, result):
    # width, height = image.size
    # new_width = width // 2
    # new_height = height // 2
    # image = image.resize((new_width, new_height))

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]
    # print(results, coord)

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        # print(
        #         f"Detected {model.config.id2label[label.item()]} with confidence for {coord}"
        #         f"{round(score.item(), 3)} at location {box}"
        #     )
        if(str(coord) not in result):
            result[str(coord)] = []
        result[str(coord)].append(model.config.id2label[label.item()])

@app.route('/analyze', methods=['POST'])    
def analyze():
    if request.method == 'POST':
        form = request.get_json()
        image_base64 = form['image']
        start_time = time.time()


        # with open('im.txt', 'r') as f:
        #     image_base64 = f.read()

        image_bytes = io.BytesIO(base64.b64decode(image_base64))

        # Load the image from the byte stream
        image = Image.open(image_bytes)


        width, height = image.size
        print("dims:", width, height)
        rows,cols = 3,3
        # Calculate sub-image dimensions
        sub_image_width = int(width / cols)
        sub_image_height = int(height / rows)

        # Initialize list to store sub-images
        sub_images = []
        threads = []
        result = {}



        # Loop through rows and columns
        for y in range(rows):
            for x in range(cols):
                # Define coordinates for the current sub-image
                left = x * sub_image_width
                top = y * sub_image_height
                right = left + sub_image_width
                bottom = top + sub_image_height

                # Crop the sub-image from the original image
                sub_image = image.crop((left, top, right, bottom))
                sub_image = sub_image.convert("RGB")

                # Append the sub-image to the list
                sub_images.append(sub_image)
                if (y==2) or (y==1 and x==1):
                    thread = threading.Thread(target=analyze_image, args=((x,y),sub_image, processor, model, result))
                    threads.append(thread)
                    thread.start()

                # sub_image = sub_image.convert("RGB")
                # sub_image.save(f'./out/sub_image_{y}_{x}.jpeg')
        for thread in threads:
            thread.join()

        
        # print("result:)",result)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time:.2f} seconds")

        return jsonify(result)
   
   
   
    return jsonify({})

@app.route('/llm_analysis', methods=['POST'])
def llm_analyze():
    start_time = time.time()

    data = analyze()
    res = json.loads(data.data)

    form = request.get_json()
    try:
        content = form['content'] 
    except Exception as e:
        print("Errrrrrorrrrrr",e)
        content = 1
    # Example: reuse your existing OpenAI setup

    # Point to the local server
    left = None
    right = None
    front = None
    ahead = None

    if "(0, 2)" in res:
       left  = ", ".join(set(res["(0, 2)"]))
    if "(2, 2)" in res:
       right  = ", ".join(set(res["(2, 2)"]))
    if "(1, 2)" in res:
        front = ", ".join(set(res["(1, 2)"]))
    
    if "(1, 1)" in res:
        ahead = ", ".join(set(res["(1, 1)"]))
    result = ""
    if front is None:
        result+="The path ahead is clear. You can move forward. "
    else:
        result+="The path ahead is blocked by "+"obstacle"+". You cannot move forward. "

    if left is not None:
        if content:
            result+="There is a "+"obstacle"+" to your close left. "
    elif front is not None:
        result+="Path to your left is clear you may go to left. "

    if right is not None:
        if content:
            result+="There is a "+"obstacle"+" to your close right. "
    elif front is not None and left is not None:
        result+="Path to your right is clear you may go to right. "
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"utli Execution time: {execution_time:.2f} seconds")
    return jsonify({"response": result})

@app.route('/cap', methods=['POST'])
def cap():
    if request.method == 'POST':
        form = request.get_json()
        image_base64 = form['image']

        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

        # img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' 
        # raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
        raw_image = Image.open(io.BytesIO(base64.b64decode(image_base64))).convert('RGB')

# conditional image captioning
        text = "An image of "
        inputs = processor(raw_image, text, return_tensors="pt")

        out = model.generate(**inputs)
        print(processor.decode(out[0], skip_special_tokens=True))

# unconditional image captioning
        inputs = processor(raw_image, return_tensors="pt")

        out = model.generate(**inputs)
        # print(processor.decode(out[0], skip_special_tokens=True))
        return jsonify({"response": processor.decode(out[0], skip_special_tokens=True)})






if __name__ == "__main__":
    app.run(debug=True)