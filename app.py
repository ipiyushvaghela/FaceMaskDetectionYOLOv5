import streamlit as st
from PIL import Image
import torch
import numpy as np



@st.cache(allow_output_mutation=True)
def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=r'D:\DXProjects\FaceMaskDetectionYOLOv5\models\best.pt', force_reload=True)
    # model = torch.hub.load("D:\E\DXAssignment\YOLOv5Projects\yolov5", 'custom', source ='local', path='D:\E\DXAssignment\YOLOv5Projects\yolov5\\runs\\train\exp8\weights\\best.pt',force_reload=True)
    return model


with st.spinner('Model is being loaded..'):
  model=load_model()

if __name__ == '__main__':

    st.title('Mask Detection using Yolov5')

    source = ("Upload Image", "Upload Video", "Use Webcam")
    source_index = st.sidebar.selectbox("Select Data Source", range(
        len(source)), format_func=lambda x: source[x])

    if source_index == 0:
        uploaded_file = st.sidebar.file_uploader(
            "Upload an Image", type=['png', 'jpeg', 'jpg'])
        if uploaded_file is not None:
            is_valid = True
            with st.spinner(text='Loading Image...'):
                st.sidebar.image(uploaded_file)
                picture = Image.open(uploaded_file)
                picture = model(picture)
                st.image(np.squeeze(picture.render()))
