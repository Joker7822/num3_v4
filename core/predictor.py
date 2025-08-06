from core.utils import preprocess_data, set_global_seed, parse_number_string

class LotoPredictor:
    def __init__(self, input_size, hidden_size):
        print("[INFO] Predictor initialized")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.model = None  # ダミー

    def train_model(self, data):
        print("[INFO] Training model...")

    def predict(self, data):
        print("[INFO] Predicting...")
        return [[1, 2, 3], 0.91], [4, 5, 6], 0.88
