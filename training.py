import sys
from tqdm.notebook import tqdm
import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.nn.functional import softmax

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding=True, return_tensors="pt")


class Trainer:
    def __init__(self, model, device, learning_rate=1e-4, path_to_weights='/content/drive/MyDrive/model_compression/model.pt'):
        self.device = device
        self.model = model.to(self.device)
        self.optimizer = Adam(list(self.model.parameters()), lr=learning_rate)
        self.criterion = CrossEntropyLoss()
        self.path_to_weights = path_to_weights

    def train(self, train_loader, val_loader, num_epochs, save_model=False):
        loss_history = []
        train_history = []
        val_history = []
        for epoch in range(num_epochs):
            print(f'Epoch {epoch + 1}')
            self.model.train()
            loss_accum = 0
            correct = 0
            total = 0
            for batch_num, batch in enumerate(tqdm(train_loader)):
                prediction, y = self.get_prediction(batch)
                loss = self.compute_loss(prediction, y)
                loss_accum += loss
                accuracy, correct, total = self.compute_accuracy(prediction, y, correct, total)
                sys.stdout.write(f'\rLoss: {loss}. Train accuracy: {accuracy}')
                sys.stdout.flush()

            mean_loss = loss_accum / batch_num
            train_accuracy = float(correct) / total
            val_accuracy = self.compute_val_accuracy(val_loader)

            loss_history.append(float(mean_loss))
            train_history.append(train_accuracy)
            val_history.append(val_accuracy)

            print(
               f"\nAverage loss: {mean_loss}, Train accuracy: {train_accuracy}, Val accuracy: {val_accuracy}" )
            if save_model:
                self.save_model()
        val_accuracy = self.compute_val_accuracy(val_loader)
        return loss_history, train_history, val_history

    def get_prediction(self, batch):
        tokenized_batch = tokenize(batch)
        ids = tokenized_batch['input_ids'].to(self.device)
        mask = tokenized_batch['attention_mask'].to(self.device)
        y = batch['label'].to(self.device)
        prediction = softmax(self.model(input_ids=ids, attention_mask=mask), dim=1)
        return prediction, y

    def compute_loss(self, prediction, y):
        loss = self.criterion(prediction, y)
        self.model.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def compute_accuracy(self, prediction, y_true, correct, total):
        y_pred = torch.argmax(prediction, axis=1)
        correct += torch.sum(y_pred == y_true)
        total += y_true.shape[0]
        accuracy = correct / total
        return accuracy, correct, total

    def compute_val_accuracy(self, val_loader):
        self.model.eval()
        correct = 0
        total = 0
        for batch_num, batch in enumerate(tqdm(val_loader)):
            prediction, y = self.get_prediction(batch)
            accuracy, correct, total = self.compute_accuracy(prediction, y, correct, total)
            sys.stdout.write(f'\rVal accuracy: {accuracy}')
            sys.stdout.flush()
        accuracy = float(correct) / total
        return accuracy

    def save_model(self):
        torch.save(self.model, self.path_to_weights)
        print(f'Model saved to {self.path_to_weights}')