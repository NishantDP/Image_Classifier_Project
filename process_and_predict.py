import numpy as np
import tensorflow as tf
from PIL import Image

def process_image(image): 
   
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    processed_image= ((tf.image.resize(image,(224,224)))/255).numpy()
    
    return processed_image
    

def predict(image_path, model, top_k=5):
    
    image = Image.open(image_path)
    image = np.asarray(image)
    processedImage = process_image(image)
    expanded_image = np.expand_dims(processedImage,  axis=0)

    pred_probabilities = model.predict(expanded_image)
    
    probs, indices = tf.nn.top_k(pred_probabilities, k=top_k)
    probs = probs.numpy()
    indices = indices.numpy()
    
    return probs, indices