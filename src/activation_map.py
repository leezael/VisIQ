from tensorflow.keras.models import load_model

# Load your trained model
model = load_model('your_model_path.h5')
model.summary()

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import Model

def get_activation_map(model, img_path, layer_name):
    # Load and preprocess the image
    img = load_img(img_path, target_size=(224, 224))  # Adjust target_size to your model's expected input shape
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Model expects a batch of images

    # Create a model that maps the input image to the activations of the specified layer
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer(layer_name).output)

    # Get the activation of the specified layer
    intermediate_output = intermediate_layer_model.predict(img_array)

    # Postprocess and visualize the activation map
    # Assuming we visualize the first filter activation
    activation = intermediate_output[0, :, :, 0]

    plt.imshow(activation, cmap='viridis')
    plt.show()

# Example usage
get_activation_map(model, 'path_to_your_image.jpg', 'layer_name_here')
