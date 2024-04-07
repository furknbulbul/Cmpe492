import torch
import numpy as np
import torch.nn.functional as F


class CustomTrainer:

    def __init__(self, train_len, val_len, test_len):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_len = train_len
        self.val_len = val_len
        self.test_len = test_len

    def train_model(self, model, data_loader, criterion, optimizer):
        correct = 0
        losses = []
        model.train()

        for i, (images, labels) in enumerate(data_loader):
            images, labels = images.to(self.device), labels.to(self.device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            _, predicted_labels = torch.max(outputs, dim=1)

            correct += int(torch.sum(predicted_labels == labels.data))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            losses.append(loss.item())

        return correct/self.train_len, np.mean(losses)

    def val_model(self, model, data_loader, criterion, validation = True):
        model.eval()
        losses = []
        correct = 0
        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                loss = criterion(outputs, labels)
                losses.append(loss.item())

            if validation:
                return correct/self.val_len, np.mean(losses)
            else:
                return correct/self.test_len, np.mean(losses)

    def get_predictions(self, model, data_loader):
        model.eval()
        predictions = []
        prediction_probs = []
        correct_values = []

        with torch.no_grad():
            for images, labels in data_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                probs = F.softmax(outputs, dim=1)
                predictions.extend(predicted)
                prediction_probs.extend(probs)
                correct_values.extend(labels)

        predictions = torch.stack(predictions).to(self.device)
        prediction_probs = torch.stack(prediction_probs).to(self.device)
        correct_values = torch.stack(correct_values).to(self.device)
        return predictions, prediction_probs, correct_values

