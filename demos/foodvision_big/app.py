import gradio as gr 
import os
import torch 

from pathlib import Path
from model import create_effnetb2_model
from timeit import default_timer as timer 
from typing import Tuple, Dict

# Setup class names
with open("class_names.txt", "r") as f: 
    class_names = [food_name.strip() for food_name in f.readlines()]

# 2. Model and transforms preparation
effnetb2, effnetb2_transforms = create_effnetb2_model(num_classes=len(class_names),
                                                      seed=42)

# Load save weights
effnetb2.load_state_dict(
    torch.load(
        f="pretrained_effnetb2_feature_extractor_pizza_steak_sushi.pth",
        map_location=torch.device("cpu")
    )
)

# 3. Predict function

def predict(img, model):
    # Start a timer
    start_time = timer()

    # Transform the input image for use with EffnetB2
    img = model(img).unsqueeze(0) # Unsqueeze = add batch dimension on 0th dim

    # Put model into eval mode, make prediction
    effnetb2.eval()
    with torch.inference_mode():
        # Pass transformed image through the model and turn the prediction logits into probs
        pred_probs = torch.softmax(effnetb2(img), dim=1)

    # Create a prediction label and prediction probability dictionary
    pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}

    # Calculate pred time
    end_time = timer()
    pred_time = round(end_time - start_time, 4)

    # Return pred dict and pred time
    return pred_labels_and_probs, pred_time

# 4. Gradio app

# Create title, description, and article
title = "FoodVision Big"
description = "An EfficientNetB2 feature extractor computer vision model to classify food images from Food101 dataset"
article = "Created at [09. PyTorch Model Deployment] (website)"

# Create example list
example_list = [["examples/" + example] for example in os.listdir("examples")]

# Create the Gradio demo
demo = gr.Interface(fn=predict,
                    inputs=gr.Image(type="pil"),
                    outputs=[gr.Label(num_top_classes=5, label="Predictions"),
                             gr.Number(label="Prediction Time (s)")],
                    examples=example_list,
                    title=title,
                    description=description,
                    article=article)

# Launch the demo
demo.launch() # generate a publically shareable URL
