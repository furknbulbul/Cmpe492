import torch
import numpy as np
from trainer.training import ImageTrainer

class MultimodalTrainer(ImageTrainer):

    def __init__(self):
        super(MultimodalTrainer, self).__init__()
    

    def train_model(self, model, data_loader, criterion, optimizer):
        correct = 0
        running_loss = 0.0
        model.train()

        for i, (images, texts, labels) in enumerate(data_loader):
            images = images.to(self.device)
            texts = texts.to(self.device)
            labels = labels.to(self.device)

            # print(labels)
            # print(images.size())
            # print(texts.size())
            # print(labels.size())

            
            images_projected, texts_projected = model(images, texts)

            correct_text_projected = texts_projected[range(texts_projected.shape[0]), labels]
            loss = criterion(images_projected, correct_text_projected)
            
            
    
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            running_loss += loss.item()
            
            _, predicted_indices = torch.min(torch.norm(images_projected.unsqueeze(1) - texts_projected, dim=2, p=2), dim=1)
            correct += (predicted_indices == labels).sum().item()
        
        return correct / len(data_loader.dataset), running_loss / len(data_loader)

        
    def val_model(self, model, data_loader, criterion):
        correct = 0
        running_loss = 0.0
        model.eval()

        with torch.no_grad():
            for i, (images, texts, labels) in enumerate(data_loader):
                images = images.to(self.device)
                texts = texts.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                images_projected, texts_projected = model(images, texts)

                # Select the correct text embeddings based on the labels
                correct_texts_projected = texts_projected[range(texts_projected.shape[0]), labels]

                loss = criterion(images_projected, correct_texts_projected)

                # Predictions are the indices of the minimum distances
                _, predicted_indices = torch.min(torch.norm(images_projected.unsqueeze(1) - texts_projected, dim=2, p=2), dim=1)
                correct += (predicted_indices == labels).sum().item()

                running_loss += loss.item()

        return correct / len(data_loader.dataset), running_loss / len(data_loader)


    

    
