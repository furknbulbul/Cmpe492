import torch
import numpy as np
from training import ImageTrainer

class MultimodalTrainer(ImageTrainer):

    def __init__(self):
        super(MultimodalTrainer, self).__init__()
    

    def train_model(self, model, data_loader, criterion, optimizer):
        correct = 0
        running_loss = 0.0
        model.train()
        

        for i, (images, texts, labels) in enumerate(data_loader):
            images = images.to(self.device)
            
            labels = labels.to(self.device)



            #forward pass
            outputs = model(images, texts)
            loss = criterion(outputs, labels)

            #backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted_labels = torch.max(outputs, dim=1)
            correct += int(torch.sum(predicted_labels == labels.data))

            running_loss += loss.item()

            if i % 50 == 0:
                print(f"Batch {i}, Loss: {loss.item()}, Accuracy: {correct / ((i+1) * images.size(0))}")


        return correct/len(data_loader.dataset), running_loss/len(data_loader)
    
    def val_model(self, model, data_loader, criterion):
        correct = 0
        running_loss = 0.0
        model.eval()

        with torch.no_grad():
            for i, (images, texts, labels) in enumerate(data_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                # forward pass
                outputs = model(images, texts)
                loss = criterion(outputs, labels)

                _, predicted_labels = torch.max(outputs, dim=1)
                correct += int(torch.sum(predicted_labels == labels.data))

                running_loss += loss.item()

                if i % 100 == 0:
                    print(f"Batch {i}, Loss: {loss.item()}, Accuracy: {correct / ((i+1) * images.size(0))}")

        return correct/len(data_loader.dataset), running_loss/len(data_loader)



    
    
