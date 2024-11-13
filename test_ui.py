import gradio as gr
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import cifar10

# Load CIFAR-10 data
(_, _), (x_test, y_test) = cifar10.load_data()
x_test = x_test.astype('float32') / 255.0

# Load model
model = load_model('cifar-10.h5')

# CIFAR-10 class names
class_names = [
    'Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
    'Dog', 'Frog', 'Horse', 'Ship', 'Truck'
]

# Function to generate random image, scale it, and make prediction
def classify_random_image():
    # Select random image from test set
    index = np.random.randint(0, x_test.shape[0])
    img = x_test[index]
    actual_label = class_names[y_test[index][0]]

    # Make prediction
    img_expanded = np.expand_dims(img, axis=0)
    predictions = model.predict(img_expanded)
    predicted_label = class_names[np.argmax(predictions)]

    # Upscale image for display (e.g., 128x128 instead of 32x32)
    img_display = cv2.resize(img, (128, 128), interpolation=cv2.INTER_NEAREST)
    img_display = (img_display * 255).astype(np.uint8)  # Convert to 8-bit format for display

    return img_display, f"Actual Label: {actual_label}", f"Predicted Label: {predicted_label}"

# Gradio interface
with gr.Blocks() as app:
    gr.Markdown("# CIFAR-10 Image Classification")
    gr.Markdown("Click the button to see a random image from CIFAR-10 and the model's prediction.")
    
    # Image display and labels for actual and predicted
    image_output = gr.Image(label="Random CIFAR-10 Image", type="numpy")
    actual_label_output = gr.Textbox(label="Actual Label")
    predicted_label_output = gr.Textbox(label="Predicted Label")

    # Button to generate random image and prediction
    classify_button = gr.Button("Generate Random Image")

    classify_button.click(
        classify_random_image,
        inputs=[],
        outputs=[image_output, actual_label_output, predicted_label_output]
    )

# Launch the app
app.launch()