from torch.autograd import Variable
import torch
import torch.optim as optim
from datetime import datetime, timedelta
from torch.utils.data import DataLoader
import os
import math
from dataset import MyDataset

class Trainer():
    def __init__(self):
        self.trainingdataset = MyDataset('/home/pi242/xai/CSE291E_AVSR_project/gdrive',mode='test',vid_out='none')
        self.trainingdataloader = DataLoader(
                                    self.trainingdataset,
                                    batch_size=1,
                                    shuffle=True,
                                    drop_last=True
                                )
    def get_dataloader(self):
        return self.trainingdataloader
    def check_data(self):
        for i_batch, sample_batched in enumerate(self.trainingdataloader):
            print(i_batch)
            print(sample_batched[0][-1], sample_batched[1][-1], sample_batched[2][-1])
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