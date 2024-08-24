import os
import torch
import streamlit as st
from torchvision import transforms
from PIL import Image

from data_loader import get_loader
from model import DecoderRNN, EncoderCNN
from nlp_utils import clean_sentence

# Update the path accordingly
cocoapi_dir = r"C:\Users\pradeep dubey\Desktop\project\imgcaption-env\cocoapi"

# Defining a transform to pre-process the testing images.
transform_test = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
        ),
    ]
)

# Creating the data loader.
data_loader = get_loader(transform=transform_test, mode="test", cocoapi_loc=cocoapi_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Specify the saved models to load.
encoder_file = "encoder-3.pkl"
decoder_file = "decoder-3.pkl"

# Select appropriate values for the Python variables below.
embed_size = 256
hidden_size = 512

# The size of the vocabulary.
vocab_size = len(data_loader.dataset.vocab)

# Initialize the encoder and decoder, and set each to inference mode.
encoder = EncoderCNN(embed_size)
decoder = DecoderRNN(embed_size, hidden_size, vocab_size)
encoder.eval()
decoder.eval()

# Load the trained weights.
encoder.load_state_dict(torch.load(os.path.join("./models", encoder_file), map_location=device))
decoder.load_state_dict(torch.load(os.path.join("./models", decoder_file), map_location=device))

# Move models to GPU if CUDA is available.
encoder.to(device)
decoder.to(device)

# Streamlit interface
st.title("🖼️ Image Captioning with AI 🤖")
st.markdown("### Upload an image and let the AI describe it! 🌟")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        st.success("File uploaded successfully! ✅")
        image = Image.open(uploaded_file).convert("RGB")
        
        # Display image with a smaller size
        st.image(image, caption="📷 Uploaded Image", width=300)  # Adjust the width as needed

        if st.button("🔮 Generate Caption"):
            st.info("Generating caption... ⏳")
            image_tensor = transform_test(image).unsqueeze(0)

            with torch.no_grad():
                image_tensor = image_tensor.to(device)
                features = encoder(image_tensor).unsqueeze(1)
                output = decoder.sample(features)

            sentence = clean_sentence(output, data_loader.dataset.vocab.idx2word)
            
            # Highlighted and larger caption
            st.markdown(f"<h2 style='text-align: center; color: #4CAF50;'>📝 Predicted Caption: {sentence}</h2>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"🚨 An error occurred: {e}")
else:
    st.warning("Please upload an image to get started. 📂")
