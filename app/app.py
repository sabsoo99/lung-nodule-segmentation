import streamlit as st
import numpy as np
from model.unet3d import UNet3D
import torch

st.title("🫁 Lung Nodule Segmentation")

uploaded_file = st.file_uploader("Upload CT scan (NumPy .npy file)", type=["npy"])

if uploaded_file is not None:
    volume = np.load(uploaded_file)
    st.write("Shape:", volume.shape)

    model = UNet3D()
    model.eval()

    input_tensor = torch.tensor(volume).unsqueeze(0).unsqueeze(0).float()
    with torch.no_grad():
        output = model(input_tensor)

    prediction = output.squeeze().numpy()
    st.success("Segmentation complete!")

    st.image(prediction[prediction.shape[0] // 2], caption="Middle Slice Prediction", use_column_width=True)
