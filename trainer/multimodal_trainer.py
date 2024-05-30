import torch
import numpy as np
from trainer.image_trainer import ImageTrainer

class MultimodalTrainer(ImageTrainer):

    def __init__(self):
        super(MultimodalTrainer, self).__init__()
    

    def pretrain_model(self, model, data_loader, criterion_contrastive, optimizer):
        correct = 0
        running_loss = 0.0
        model.train()

        assert not model.use_classifier, "Model must be set to not use classifier"

        for i, (images, texts, labels) in enumerate(data_loader):

            images = images.to(self.device)
            texts = texts.to(self.device)
            labels = labels.to(self.device)

            optimizer.zero_grad()

            images_projected, texts_projected = model(images, texts)
            loss = criterion_contrastive(images_projected, texts_projected)
       
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        return running_loss / len(data_loader)
    

    def train_classifier(self, model, data_loader, criterion_classifier, optimizer):
        correct = 0
        running_loss = 0.0
        model.train()
        

        
        assert model.use_classifier, "Model must be set to use classifier"
        # freeze text

        for i, (images, texts, labels) in enumerate(data_loader):

            images = images.to(self.device)
            texts = texts.to(self.device)
            labels = labels.to(self.device)

            optimizer.zero_grad()

            logits = model(images, texts)
            loss = criterion_classifier(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            correct += (torch.argmax(logits, dim=1) == labels).sum().item()
        

        accuracy = correct / len(data_loader.dataset)
            
        
        return accuracy, running_loss / len(data_loader)

    
        
    def pre_val_model(self, model, data_loader, criterion):
        
        running_loss = 0.0
        model.eval()
        
        assert not model.use_classifier, "Model must be set to not use classifier in pretrained validation"

        with torch.no_grad():
            for i, (images, texts, labels) in enumerate(data_loader):
                images = images.to(self.device)
                texts = texts.to(self.device)

                
                images_projected, texts_projected = model(images, texts)
                loss = criterion(images_projected, texts_projected)


                running_loss += loss.item()

        return running_loss / len(data_loader)
    

    def val_model(self, model, data_loader, criterion):
        correct = 0
        running_loss = 0.0
        model.eval()

        assert model.use_classifier, "Model must be set to use classifier in validation"

        with torch.no_grad():
            for i, (images, texts, labels) in enumerate(data_loader):
                images = images.to(self.device)
                texts = texts.to(self.device)
                labels = labels.to(self.device)

                logits = model(images, texts)
                loss = criterion(logits, labels)

                running_loss += loss.item()
                correct += (torch.argmax(logits, dim=1) == labels).sum().item()

        accuracy = correct / len(data_loader.dataset)
        return accuracy, running_loss / len(data_loader)
    

    def test_model(self, model, data_loader):
        model.eval()
        predictions = []
        prediction_probs = []
        correct_values = []
        correct_predictions = 0

        with torch.no_grad():
            for images, texts, labels in data_loader:
                images = images.to(self.device)
                texts = texts.to(self.device)
                labels = labels.to(self.device)

                logits = model(images, texts)
                _, predicted = torch.max(logits, 1)
                probs = torch.nn.functional.softmax(logits, dim=1)

                correct_predictions += torch.sum(predicted == labels).item()

                predictions.extend(predicted)
                prediction_probs.extend(probs)
                correct_values.extend(labels)

        accuracy = correct_predictions / len(data_loader.dataset)
        return predictions, prediction_probs, correct_values, accuracy
    

    
    

    
