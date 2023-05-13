import torch
from model import ColaModel
from data import DataModule

class ColaPredictor:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = ColaModel.load_from_checkpoint(model_path)
        self.model.eval()
        self.model.freeze()
        self.preprocessor = DataModule()
        self.softmax = torch.nn.Softmax(dim=0)
        self.labels = ["unacceptable", "acceptable"]
        
    def predict(self, text):
        inference_sample = {"sentence": text}
        processed = self.preprocessor.tokenize_data(inference_sample)
        logits = self.model(
            torch.tensor([processed["input_ids"]]),
            torch.tensor([processed["attention_mask"]]),
        )
        scores = self.softmax(logits[0]).tolist()
        predictions = []
        for score, lable in zip(scores, self.labels):
            predictions.append({"lable": lable, "score": score})
        return predictions
    
    
if __name__ == "__main__":
    sentence = "The boy is sitting on a bench"
    predictor = ColaPredictor("./models/epoch=0-step267.ckpt")
    print(predictor.predict(sentence))