import numpy as np
import tensorflow as tf
import argparse
import json
from process_and_predict import predict
import sys
import tensorflow_hub as hub

if __name__ == '__main__':
    print('Predicting the image. Please wait....')
    
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path")
    parser.add_argument("final_model")
    parser.add_argument("top_k", default = 3)
    parser.add_argument("category_names", default = "label_map.json")
    args = parser.parse_args()
    
    image_path = args.image_path
    model = tf.keras.models.load_model(args.final_model ,custom_objects={'KerasLayer':hub.KerasLayer} )
    top_k = int(args.top_k)

    with open(args.category_names, 'r') as f:
        class_names = json.load(f)
   
    probs, indices = predict(image_path, model, top_k)
    
    flower_names = [class_names[str(idd+1)] for idd in indices[0]]
    
    print("The top predicted classes: ",flower_names)
    print("The top proabilities corresponding to the above classes: ",probs)
    
    print('The prediction is complete')