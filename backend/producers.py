import sys
from pathlib import Path
import torch
from confluent_kafka import Producer, Consumer
import json

from backend.data import load_mnist_data, SimpleCNN, run_train
from backend.config_file import cfg_producer, cfg_consumer, kafka_topic

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

model_path = "/home/andrey/Projects/HSE/HSE_mag_Data_Mining/BD_HW_Kafka/mnist_cnn.pth"

class producer_mnist:
    def __init__(self):
        self.producer = Producer(cfg_producer)

    def send_to_kafka(self, topic, data):
        model = SimpleCNN()
        model.load_state_dict(torch.load('mnist_cnn.pth'))
        model.eval()
        
        with torch.no_grad():
            output = model(data)
            _, predicted = torch.max(output, 1)

            predicted_label = predicted.item()
            print(f'Predicted Label: {predicted_label}')

        message = json.dumps({"predicted_label": predicted_label}) #.encode('utf-8')
        print(message)
        self.producer.produce(topic, value=message)
        self.producer.flush()

class consumer_mnist:
    def __init__(self):
        self.consumer = Consumer(cfg_consumer)

    def receive_from_kafka(self, topic):
        self.consumer.subscribe([topic])
        msg = self.consumer.poll(timeout=10.0)
        if msg is None:
            return None
        try:
            message = json.loads(msg.value())
            print(message)
            return message["predicted_label"]
        except (json.JSONDecodeError, KeyError, UnicodeDecodeError) as e:
            print(f"Error decoding message: {e}")
            return None