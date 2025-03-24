from tensorflow.keras.models import load_model
import os

def load_my_model():
    model_path = os.path.join(os.path.dirname(__file__), 'model3.h5')
    model = load_model(model_path)
    return model

import pandas as pd

def load_csv():
    csv_path = os.path.join(os.path.dirname(__file__), 'Data-of-teeth.csv')
    data = pd.read_csv(csv_path)
    return data
