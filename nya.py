import streamlit as st 
from PIL import Image
import torch
from torchvision import models, transforms
import requests
import json
import faiss
import numpy as np
import google.generativeai as ai

# Set page config
st.set_page_config(page_title="Document J.A.R.V.I.S", layout="wide")

# Apply custom CSS for animations and styling
st.markdown("""
    <style>
    body {
        font-family: Aerial, sans-serif;
        background-color: #14151a;
    }
    
    h1, h2, h4, h5, h6 {
        color: #059862;
        animation: fade-in 2s;
    }
    
    p,h3 {
        font-size: 18px;
        line-height: 1.5;
        color: #14151a;
        animation: slide-in 3s;
    }

    @keyframes fade-in {
        from { opacity: 0; }
        to { opacity: 1; }
    }

    @keyframes slide-in {
        from { transform: translateX(-100px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    </style>

    <!-- Animating the title -->
    <h1 style="animation-delay: 0.5s;">J.A.R.V.I.S : Just a Rather Very Intelligent System</h1>

    <!-- Animating the subtitle -->
    <h3 style="animation-delay: 1s;">This chatbot is built using the Retrieval-Augmented Generation (RAG) framework, leveraging Google's Generative AI model Gemini-PRO.</h3>

   
""", unsafe_allow_html=True)

# Load a pre-trained image classification model (ResNet)
model = models.resnet50(pretrained=True)
model.eval()

# Define image preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load ImageNet labels
LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
labels = requests.get(LABELS_URL).json()

# Function to predict image class
def predict_image_class(image):
    try:
        img_tensor = preprocess(image).unsqueeze(0)
        with torch.no_grad():
            outputs = model(img_tensor)
        _, predicted_class = outputs.max(1)
        return labels[predicted_class.item()]
    except Exception as e:
        st.error(f"Error processing the image: {e}")
        return None

# Function to interact with Gemini (AI21)
def chat_with_gemini(prompt, api_key):
    try:
        endpoint = "https://api.ai21.com/studio/v1/j1-large/complete"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "prompt": prompt,
            "numResults": 1,
            "maxTokens": 256,
            "stopSequences": ["\n"],
            "temperature": 0.7
        }
        
        response = requests.post(endpoint, headers=headers, json=payload)
        response.raise_for_status()

        data = response.json()

        if 'completions' in data:
            return data["completions"][0]["data"]["text"].strip()
        else:
            st.error(f"Unexpected API response: {data}")
            return None

    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {e}")
        return None
    except Exception as e:
        st.error(f"Error generating a response: {e}")
        return None

# Function to create embedding using Faiss
def create_embedding(image, api_key):
    try:
        # Get image features
        img_tensor = preprocess(image).unsqueeze(0)
        with torch.no_grad():
            features = model(img_tensor)
        
        # Create Faiss index
        index = faiss.IndexFlatL2(features.shape[1])
        
        # Add features to index
        index.add(features.numpy())
        
        # Search for similar vectors
        D, I = index.search(features.numpy(), k=1)
        
        return D, I
    except Exception as e:
        st.error(f"Error creating embedding: {e}")
        return [], []

# Streamlit interface
st.title("Conversational Image Recognition Chatbot")
st.write("Upload an image, enter your Gemini API key, and ask questions about it using advanced AI capabilities.")

# Upload an image
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# Input Gemini API key
api_key = st.text_input("Enter your Gemini API key:", type="password")

# Process image button
if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button('Process Image'):
            prediction = predict_image_class(image)
            if prediction:
                st.write(f"Recognition Result: {prediction}")
                st.session_state['image_processed'] = True
                st.session_state['prediction'] = prediction
                
                # Create embedding
                D, I = create_embedding(image, api_key)
                if D == [] or I == []:
                    st.error("Failed to create embedding.")
                else:
                    st.write(f"Embedding created successfully!")
                    st.session_state['embedding'] = (D, I)
    
    # with col2:
    #     user_query = st.text_input("Ask a question about the image", key="query")
    #     if st.button('Get Answer') and 'image_processed' in st.session_state:
    #         prompt = f"The image shows {st.session_state['prediction']}. Based on this, answer the following question: {user_query}"
    #         response = chat_with_gemini(prompt, api_key)
    #         if response:
    #             st.write(f"Chatbot Response: {response}")

# Separate section for the Google Gemini AI chatbot
st.markdown("<hr/>", unsafe_allow_html=True)
st.write("## J.A.R.V.I.S AI Chatbox (Conversational Mode)")

# Configure the Gemini API key for Google Generative AI (Google Gemini)
API_KEY_GEMINI = 'AIzaSyAmpsTN7_EVj-6UuGt-g127d-5ANK_oN1c'
ai.configure(api_key=API_KEY_GEMINI)

# Create chat object using Gemini model
model = ai.GenerativeModel("gemini-pro")
chat = model.start_chat()

# Google Gemini AI chatbox interaction
user_input = st.text_input("Enter your query for J.A.R.V.I.S (type 'bye' to exit):")

if user_input.lower() != 'bye' and st.button("Send to J.A.R.V.I.S"):
    try:
        gemini_response = chat.send_message(user_input)
        if gemini_response:
            st.write(f"J.A.R.V.I.S: {gemini_response.text}")
    except Exception as e:
        st.error(f"Error with Google Gemini chat: {e}")
elif user_input.lower() == 'bye':
    st.write("J.A.R.V.I.S: Goodbye Sir!")