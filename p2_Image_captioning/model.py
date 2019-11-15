#!/usr/bin/python3
import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet18(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        
        #embedding layer
        self.embd=nn.Embedding(num_embeddings= vocab_size ,embedding_dim=embed_size)
        
        #LSTM Layer
        self.lstm = nn.LSTM(embed_size,hidden_size,num_layers,batch_first=True)
        
        #Linear layer
        self.fc = nn.Linear(hidden_size,vocab_size)
        
        #batch norm
        self.norm1 = nn.BatchNorm1d(num_features=vocab_size)
        pass
    
    def forward(self, features, captions):
        
        captions = captions[:, :-1]
        captions = self.embd(captions)
        
        # Concatenate the features and caption inputs
        features = features.unsqueeze(1)
        inputs = torch.cat((features, captions), 1)
        outputs, _ = self.lstm(inputs)
        
        # Convert LSTM outputs to word predictions
        outputs = self.fc(outputs)
        
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        
        preds = []
        count = 0
        word_item = None
        
        while count < max_len and word_item != 1 :
            
            #Predict output
            output_lstm, states = self.lstm(inputs, states)
            output = self.fc(output_lstm)
            
            #Get max value
            prob, word = output.max(2)
            
            #append word
            word_item = word.item()
            preds.append(word_item)
            
            #next input is current prediction
            inputs = self.embd(word)
            
            count+=1
        
        return preds
