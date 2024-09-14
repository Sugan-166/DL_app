import tensorflow as tf
import cv2
import numpy as np
import streamlit as st
from PIL import Image
import os
import time

# Load Pretrained Model from .h5 File
def load_pretrained_model(model_path):
    if os.path.isfile(model_path):
        try:
            model = tf.keras.models.load_model(model_path)
            print(f"Successfully loaded model from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            model = None
    else:
        print(f"File not found: {model_path}")
        model = None
    return model

# Enhance Region Resolution with Color Overlay
def enhance_region_resolution_with_fade(model, region, color=(255, 105, 180)):
    # Resize to model input size
    region_resized = cv2.resize(region, (256, 256))
    
    # Convert grayscale to RGB (3 channels)
    region_rgb = cv2.cvtColor(region_resized, cv2.COLOR_GRAY2RGB)
    
    # Normalize the region
    region_norm = np.expand_dims(region_rgb, axis=0).astype('float32') / 255.0

    # Apply the model
    region_tensor = tf.convert_to_tensor(region_norm, dtype=tf.float32)
    high_res_region = model.predict(region_tensor)[0]
    high_res_region = (high_res_region * 255).astype(np.uint8)
    
    # Resize back to original size
    high_res_region_resized = cv2.resize(high_res_region, (region.shape[1], region.shape[0]))

    # Create a colored overlay with some transparency
    overlay = np.full_like(high_res_region_resized, color, dtype=np.uint8)  # Create overlay with specified color
    
    # Blend the enhanced region with the color overlay
    blended_region = cv2.addWeighted(high_res_region_resized, 0.7, overlay, 0.3, 0)  # Adjust weights for desired fade
    
    return blended_region

# Process Image and Enhance Detected Regions
def process_image_and_enhance_regions(model, image, color=(255, 105, 180), detect_high=True):
    # Convert PIL image to NumPy array
    image_np = np.array(image.convert("RGB"))  # Convert to RGB if not already
    original_image_gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    
    if original_image_gray is None:
        st.error("Error converting image to grayscale.")
        return None, None

    # Detect regions based on intensity
    contours = detect_high_intensity_regions(original_image_gray) if detect_high else detect_low_intensity_regions(original_image_gray)
    
    if not contours:
        return original_image_gray, None
    
    # Create an enhanced image copy in RGB
    enhanced_image = cv2.cvtColor(original_image_gray, cv2.COLOR_GRAY2RGB)
    
    # Initialize the progress bar
    progress_bar = st.progress(0)
    
    num_regions = len(contours)
    if num_regions == 0:
        return original_image_gray, enhanced_image
    
    # Process each detected region
    for idx, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        x_end, y_end = x + w, y + h
        
        # Extract and enhance the detected region with a color overlay
        region = original_image_gray[y:y_end, x:x_end]
        enhanced_region = enhance_region_resolution_with_fade(model, region, color)
        
        # Ensure sizes match before placing the enhanced region back into the image
        if enhanced_region.shape[:2] == (y_end - y, x_end - x):
            enhanced_image[y:y_end, x:x_end] = enhanced_region
        else:
            st.warning(f"Size mismatch: {enhanced_region.shape} vs {(y_end - y, x_end - x)}")
        
        # Update the progress bar
        progress = (idx + 1) / num_regions
        progress_bar.progress(progress)
    
    progress_bar.empty()  # Clear the progress bar
    
    return original_image_gray, enhanced_image

# Detect Low-Intensity Regions Using Simple Thresholding
def detect_low_intensity_regions(image, threshold=50):
    _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# Detect High-Intensity Regions Using Simple Thresholding
def detect_high_intensity_regions(image, threshold=200):
    _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# Draw Detected Regions on Image with Rectangles
def draw_detected_regions(image, contours, color=(255, 105, 180), thickness=2):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x_end, y_end = x + w, y + h
        
        # Ensure coordinates are within image bounds
        x = max(x, 0)
        y = max(y, 0)
        x_end = min(x_end, image_rgb.shape[1])
        y_end = min(y_end, image_rgb.shape[0])
        
        # Draw rectangles
        cv2.rectangle(image_rgb, (x, y), (x_end, y_end), color, thickness)
    
    return image_rgb

# Estimate Processing Time
def estimate_processing_time_for_image(model, image):
    start_time = time.time()
    
    # Run the processing function to get an estimate
    _, _ = process_image_and_enhance_regions(model, image)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    return elapsed_time

# Streamlit UI
def main():
    st.title("Image Processing with Super-Resolution")
    
    # Load pre-trained model
    model_path = 'super_resolution_model.h5'  # Update with the actual .h5 file path
    model_sr = load_pretrained_model(model_path)
    
    if model_sr is None:
        st.error("Model not loaded. Please check the model file.")
        return
    
    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        
        # Estimate processing time
        estimated_time = estimate_processing_time_for_image(model_sr, image)
        st.write(f"Estimated processing time: {estimated_time:.2f} seconds")
        
        # Process the image
        original_image_gray, enhanced_image = process_image_and_enhance_regions(model_sr, image)
        
        if enhanced_image is not None:
            # Display images
            st.image(original_image_gray, caption='Original Image (Grayscale)', use_column_width=True)
            st.image(enhanced_image, caption='Enhanced Image', use_column_width=True)
            
            # Save and provide download links
            save_path_original = 'original_image_gray.png'
            save_path_enhanced = 'enhanced_image.png'
            cv2.imwrite(save_path_original, original_image_gray)
            cv2.imwrite(save_path_enhanced, enhanced_image)
            
            st.download_button(label="Download Original Image (Grayscale)", data=open(save_path_original, "rb").read(), file_name=save_path_original)
            st.download_button(label="Download Enhanced Image", data=open(save_path_enhanced, "rb").read(), file_name=save_path_enhanced)
        else:
            st.error("Error processing image.")

if __name__ == "__main__":
    main()
