import streamlit as st
import cv2
import pytesseract
import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from transformers import AutoModel, AutoTokenizer, AutoFeatureExtractor
import faiss

# Set Tesseract executable path if it's not in your PATH
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update the path as needed

# Initialize object detection model (replace with your preferred model)
model_name = "yolov5s"  # You can choose a pre-trained YOLOv5 model for object detection
model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
model.to('cpu')  # Assuming CPU inference for simplicity

# Initialize CLIP model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = AutoModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
clip_tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch16")
clip_processor = AutoFeatureExtractor.from_pretrained("openai/clip-vit-base-patch16")

st.set_page_config(page_title="Document Genie", layout="wide")

st.markdown("""
## Document Genie: Analyze and Understand Your Images

This AI tool empowers you to upload images, extract information and objects within them, and gain insights using powerful models.
""")

# Convert image bytes to a format suitable for OpenCV
def convert_image_bytes(image_bytes):
    np_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    return image

# Extract text from images using Tesseract OCR
def extract_text_from_image(image):
    try:
        # Convert the image to grayscale (optional, but can improve OCR accuracy)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Use Tesseract to extract text
        extracted_text = pytesseract.image_to_string(gray_image)
        return extracted_text
    except Exception as e:
        st.warning(f"Failed to extract text from image. Error: {str(e)}")
        return ""

# Use object detection model to identify objects in the image
def detect_objects(image):
    results = model(image)  # Run object detection
    objects = results.pandas().xyxy[0]  # Extract object information
    return objects["name"].tolist()  # Return a list of detected object names

# Create CLIP embeddings for images
def get_image_embeddings(image_bytes):
    image = convert_image_bytes(image_bytes)
    image = clip_processor(image, return_tensors="pt").to(device)
    with torch.no_grad():
        img_embeds = clip_model.get_image_features(**image)
    return img_embeds.cpu().numpy()

# Build a FAISS index for embeddings (optional for future improvements)
def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  # Use L2 distance
    index.add(embeddings)
    return index

# Generate comprehensive response based on extracted information and user query
def create_response(image_bytes, user_query):
    image = convert_image_bytes(image_bytes)

    # Extract text using OCR
    extracted_text = extract_text_from_image(image)

    # Detect objects using the object detection model
    detected_objects = detect_objects(image)

    # Generate a response based on the user query and extracted information
    response = f"This image likely contains: "
    if detected_objects:
        response += ", ".join(detected_objects)
    if extracted_text:
        response += f"\nText found in the image: {extracted_text}"

    # If the user query is specific, provide more detailed information
    if user_query:
        # Use CLIP to find the most relevant object or text to the query
        query_embedding = get_text_embeddings(user_query)
        image_embedding = get_image_embeddings(image_bytes)  # Get image embeddings within the function
        similarity_score, similar_index = find_most_similar_item(image_embedding, query_embedding)

        if similarity_score > 0.7:  # Adjust threshold as needed
            if similar_index == 0:  # Similar to image
                response += f"\nRegarding your query about '{user_query}', this image seems to be directly related."
            else:  # Similar to detected object or text
                response += f"\nRegarding your query about '{user_query}', this image contains '{detected_objects[similar_index]}' which might be relevant."

    return response

def get_text_embeddings(text):
    inputs = clip_tokenizer([text], return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        text_embeds = clip_model.get_text_features(**inputs)
    return text_embeds.cpu().numpy()

def find_most_similar_item(image_embedding, query_embedding):
    # Ensure the embeddings have compatible shapes (e.g., both (512,))
    image_embedding = image_embedding.squeeze(0)  # Remove leading dimension if necessary
    query_embedding = query_embedding.squeeze(0)  # Remove leading dimension if necessary

    # Implement your similarity measure here (e.g., cosine similarity)
    similarity_score = np.dot(image_embedding, query_embedding) / (np.linalg.norm(image_embedding) * np.linalg.norm(query_embedding))
    similar_index = 0  # Assuming the first item is the most similar
    return similarity_score, similar_index

def main():
    st.header("AI Image Analyzer ")

    # User image upload
    image_file = st.file_uploader("Upload Image (JPG, JPEG)", type="jpg", accept_multiple_files=False, key="image_uploader")

    if image_file:
        with st.spinner("Analyzing image..."):
            image_bytes = image_file.read()

            # Get image embeddings
            image_embedding = get_image_embeddings(image_bytes)

            # Generate response based on the image
            response = create_response(image_bytes, "")  # Initial response without query

            # Display the initial response
            st.success(response)

            # Allow user to ask questions
            user_query = st.text_input("Ask a question about the image:")
            if user_query:
                # Generate updated response with query
                response = create_response(image_bytes, user_query)
                st.success(response)

if __name__ == "__main__":
    main()