from torch.autograd import Variable
import torch
import torch.optim as optim
from datetime import datetime, timedelta
from torch.utils.data import DataLoader
import os
import math
from dataset import MyDataset
from decoder import *
from encoders import *

class Model(nn.Module):
    def __init__(self, emb_size, ntokens, nhead, nhid, nlayers):
        super(Model, self).__init__()
        self.aenc = AudioTEncoder(240, emb_size, nhead, nhid, nlayers)
        self.venc = VideoTEncoder(512, emb_size, nhead, nhid, nlayers)
        self.dec = TDecoder_Gen(emb_size, ntokens, nhead, nhid, nlayers)
        
    def forward(self, oa, ov, audio_dims, video_dims):
        opa = self.aenc(oa, audio_dims)
        opv = self.venc(ov, video_dims)
        output = self.dec(opa, opv)
        return output
        

class Trainer():
    def __init__(self):
        self.trainingdataset = MyDataset('./',mode='test',vid_out='resnet')
        self.trainingdataloader = DataLoader(
                                    self.trainingdataset,
                                    batch_size=5,
                                    shuffle=True,
                                    drop_last=True
                                )
        self.model = Model(512, 100, 4, 256, 4)
        
    def get_dataloader(self):
        return self.trainingdataloader
    
    def check_data(self):
        for i_batch, sample in enumerate(self.trainingdataloader):
            print(i_batch)
            audio, audio_dims = sample[0]
            video, _, video_dims = sample[1]
            # print(audio.shape, video.shape)
            output = self.model(torch.tensor(audio, dtype=torch.float32).permute(1, 0, 2), torch.tensor(video, dtype=torch.float32).permute(1, 0, 2), audio_dims[0], video_dims[0])
            print(output.shape)
            break
            
    # def epoch(self, model, epoch):
    #     #set up the loss function.
    #     criterion = model.loss()
    #     optimizer = optim.SGD(
    #                     model.parameters(),
    #                     lr = self.learningRate(epoch),
    #                     momentum = self.learningrate,
    #                     weight_decay = self.weightdecay)

    #     #transfer the model to the GPU.
    #     if(self.usecudnn):
    #         criterion = criterion.cuda(self.gpuid)

    #     startTime = datetime.now()
    #     print("Starting training...")
    #     for i_batch, sample_batched in enumerate(self.trainingdataloader):
    #         optimizer.zero_grad()
    #         input = Variable(sample_batched['temporalvolume'])
    #         labels = Variable(sample_batched['label'])

    #         if(self.usecudnn):
    #             input = input.cuda(self.gpuid)
    #             labels = labels.cuda(self.gpuid)

    #         outputs = model(input)
    #         loss = criterion(outputs, labels.squeeze(1))

    #         loss.backward()
    #         optimizer.step()
    #         sampleNumber = i_batch * self.batchsize

    #         if(sampleNumber % self.statsfrequency == 0):
    #             currentTime = datetime.now()
    #             output_iteration(sampleNumber, currentTime - startTime, len(self.trainingdataset))

    #     print("Epoch completed, saving state...")
    #     torch.save(model.state_dict(), "trainedmodel.pt")
if __name__ == "__main__":
    
    tr = Trainer()
    tr.check_data()