from pathlib import Path
import sys
import torch

import streamlit as st

from PIL import Image

from backend.producers import producer_mnist, consumer_mnist

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from backend.data import inference_transforms

def main():
    st.title('MNIST Image Classification with Kafka')

    producer = producer_mnist()
    consumer = consumer_mnist()

    uploaded_file = st.file_uploader("Upload an MNIST image", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('L')
        image = image.resize((28, 28))
        image_tensor = inference_transforms(image)

        producer.send_to_kafka('mnist_images', image_tensor)

        prediction_data = consumer.receive_from_kafka('mnist_images')
        print('-' * 80)
        print(prediction_data)
        if prediction_data:
            prediction_label = int(prediction_data)
            print('-' * 80)
            print(prediction_label)
            st.write(f'Predicted Label: {prediction_label}')

        st.image(image, caption='Uploaded Image', use_column_width=True)

if __name__ == "__main__":
    main()