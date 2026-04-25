import tensorflow as tf
import numpy as np
import cv2
import matplotlib.cm as cm
from PIL import Image
from utils.logger import logger

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Generates a Grad-CAM heatmap for a given image and model.
    """
    # Create a model that maps the input image to the activations of the last conv layer
    try:
        grad_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
        )
    except Exception as e:
        logger.error(f"Error creating Grad-CAM model: {e}")
        return None

    # Compute the gradient of the top predicted class for our input image
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        
        # Handle cases where preds might be wrapped in a list (e.g., if model output is a list)
        if isinstance(preds, list):
            preds = preds[0]
            
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img, heatmap, alpha=0.4):
    """
    Superimposes the heatmap on the original image.
    img: PIL Image or numpy array (0-255)
    heatmap: numpy array (normalized)
    """
    # Ensure img is a PIL Image and convert to RGB
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    
    # Always convert to RGB to ensure 3 channels
    if img.mode != "RGB":
        img = img.convert("RGB")
        
    # Now convert to numpy array (H, W, 3)
    img = np.array(img)
    
    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)

    return superimposed_img
'''
You have the core concept exactly right: **Grad-CAM works by looking at how much each feature in the last
convolutional layer contributes to the final prediction score.**

To achieve this, we don't just need the "values" of the last layer; we need the "gradients" (the derivatives) of
the output with respect to those values.

Here is the line-by-line explanation of your code.

---

### Part 1: `make_grad_cam_heatmap`
This function performs the heavy mathematical lifting.

#### 1. The Multi-Output Model
```python
grad_model = tf.keras.models.Model(
    [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
)
```
Normally, a model takes an image and returns a prediction. To calculate gradients, we need the model to return
**two** things simultaneously:
1.  The **activations** (the feature maps) of the last convolutional layer.
2.  The **predictions** (the final output probabilities).
By creating this `grad_model`, we can "intercept" the data mid-flow.

#### 2. The Gradient Tape
```python
with tf.GradientTape() as tape:
    last_conv_layer_output, preds = grad_model(img_array)

    if isinstance(preds, list):
        preds = preds[0]

    if pred_index is None:
        pred_index = tf.argmax(preds[0])
    class_channel = preds[:, pred_index]
```
*   **`tf.GradientTape()`**: This starts recording operations so we can compute derivatives later.
*   **`preds[:, pred_index]`**: This is crucial. We don't want the gradient of the *whole* prediction vector. We
only want the gradient of the **single scalar value** representing the class we are interested in (e.g., "is this
a dog?"). We isolate that specific score.

#### 3. Calculating the Gradients
```python
grads = tape.gradient(class_channel, last_conv1ayer_output)
```
This is the heart of Grad-CAM. We ask TensorFlow: *"If I change the pixels in the convolutional feature map
slightly, how much does the 'dog' prediction score change?"*
The result, `grads`, tells us which parts of the feature map are most important for that specific class.

#### 4. Global Average Pooling (Importance Weights)
```python
pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
```
The `grads` tensor has the same shape as the feature maps (Height, Width, Channels). To simplify this, we
calculate the **mean** of the gradients across the height and width. This gives us a single weight for every
**channel**.
*   *Example:* If Channel 5 has a high positive mean gradient, it means Channel 5 is very important for detecting
a "dog."

#### 5. The Weighted Sum (Generating the Heatmap)
```python
last_conv_layer_output = last_conv_layer_output[0]
heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
heatmap = tf.squeeze(heatmap)
```
*   **`@` (Matrix Multiplication)**: We take our original feature maps and multiply each channel by its
corresponding importance weight (`pooled_grads`) calculated in the previous step.
*   **`tf.squeeze`**: We clean up the dimensions to get a 2D image.

#### 6. Normalization (ReLU & Scaling)
```python
heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
```
*   **`tf.maximum(heatmap, 0)`**: This is the **ReLU** operation. We only care about features that have a
**positive** influence on the class. Features that decrease the probability (negative gradients) are discarded.
*   **`/ tf.math.reduce_max`**: This scales the values between 0 and 1 so we can use it as a colormap.

---

### Part 2: `save_and_display_gradcam`
This function is purely for visualization (turning numbers into a pretty picture).

#### 1. Image Preparation
```python
if not isinstance(img, Image.Image):
    img = Image.fromarray(img)
if img.mode != "RGB":
    img = img.convert("RGB")
img = np.array(img)
```
This ensures that regardless of whether you passed a NumPy array or a PIL image, it ends up as a standard RGB
NumPy array.

#### 2. Colorizing the Heatmap
```python
heatmap = np.uint8(255 * heatmap)
jet = cm.get_cmap("jet")
jet_colors = jet(np.arange(256))[:, :3]
jet_heatmap = jet_colors[heatmap]
```
*   **`255 * heatmap`**: Converts the 0.0–1.0 range to 0–255.
*   **`jet`**: This is a standard colormap where low values are blue and high values are red.
*   **`jet_colors[heatmap]`**: This uses the heatmap values as indices to look up the corresponding RGB color in
the "Jet" spectrum.

#### 3. Resizing and Superimposing
```python
jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

superimposed_img = jet_heatmap * alpha + img
```
*   **`resize`**: The heatmap is usually much smaller (e.g., 7x7 or 14x14) than the original image. We must
stretch it back to the original image size.
*   **`alpha` (Blending)**: This is the "transparency" step. We take a percentage of the colorful heatmap
(`alpha`) and add it to the original image. If `alpha=0.4`, the original image is 60% visible and the heatmap is
40% visible.

### Summary of the Logic Flow
1.  **Forward Pass:** Get features and predictions.
2.  **Backward Pass:** Calculate how much each feature affects the prediction score.
3.  **Pool:** Average those gradients to get "importance weights" per channel.
4.  **Weight:** Multiply the original features by these weights.
5.  **ReLU:** Remove negative influences.
6.  **Overlay:** Paint the result back onto the original image using a color map.
'''
