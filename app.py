from pathlib import Path
import sys
import torch

import streamlit as st

from PIL import Image

from backend.producers_consumers import producer_mnist_inference, consumer_mnist_inference
from backend.config_file import kafka_topic

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from backend.data import inference_transforms

def main():
    st.title('MNIST Image Classification with Kafka')

    producer = producer_mnist_inference()
    consumer = consumer_mnist_inference()

    uploaded_file = st.file_uploader("Upload an MNIST image", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('L')
        image = image.resize((28, 28))
        image_tensor = inference_transforms(image)

        producer.send_to_kafka(kafka_topic, image_tensor)

        prediction_data = consumer.receive_from_kafka(kafka_topic)
        # if st.button('Predict'):
        if prediction_data is not None:
            if isinstance(prediction_data, int):
                st.write(f'Predicted Label: {prediction_data}')
            else:
                st.write(f'Received data: {prediction_data}')
        else:
            st.write("No prediction received from Kafka.")

        st.image(image, caption='Uploaded Image', use_column_width=True)

if __name__ == "__main__":
    main()
