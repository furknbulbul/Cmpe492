import torch
import numpy as np
import torch.nn.functional as F


class ImageTrainer:

    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
   

    def train_model(self, model, data_loader, criterion, optimizer):
        correct = 0
        running_loss = 0.0
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

            running_loss += loss.item()

        return correct/len(data_loader.dataset), running_loss/len(data_loader) 


    def val_model(self, model, data_loader, criterion):
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

        return correct/len(data_loader.dataset), np.mean(losses)

    def test_model(self, model, data_loader):
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
        acc = (predictions == correct_values).sum().item() / len(correct_values)
        return predictions, prediction_probs, correct_values, acc
    


    def train_contrastive(self, model, data_loader, criterion, optimizer):
        running_loss = 0.0
        model.train()
        for (images1, images2), target in data_loader:
            images1, images2, target = images1.to(self.device), images2.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            outputs1, outputs2 = model(images1, images2)
            loss = criterion(outputs1, outputs2, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        return running_loss / len(data_loader)
    
    def val_contrastive(self, model, data_loader, criterion):
        running_loss = 0.0
        model.eval()
        with torch.no_grad():
            for (images1, images2), target in data_loader:
                images1, images2, target = images1.to(self.device), images2.to(self.device), target.to(self.device)
                outputs1, outputs2 = model(images1, images2)
                loss = criterion(outputs1, outputs2, target)
                running_loss += loss.item()

        return running_loss / len(data_loader)


    def train_classifier(self, model, data_loader, criterion, optimizer):
        running_loss = 0.0
        model.train()
        for (images, labels) in data_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        return running_loss / len(data_loader)
    
    def val_classifier(self, model, data_loader, criterion):
        running_loss = 0.0
        correct = 0
        model.eval()
        with torch.no_grad():
            for (images, labels) in data_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
        accuracy = correct / len(data_loader.dataset)
        return accuracy, running_loss / len(data_loader)