"""
BY Aashis Khanal
"""
import os
import sys

import numpy as np
import torch
from PIL import Image as IMG

import sklearn
from scipy import spatial
from sklearn.metrics.pairwise import cosine_similarity

import utils.img_utils as imgutils
from neuralnet.torchtrainer import NNTrainer
import torch.nn.functional as F
from neuralnet.utils.measurements import ScoreAccumulator

sep = os.sep


class TracknetTrainer(NNTrainer):
    def __init__(self, **kwargs):
        NNTrainer.__init__(self, **kwargs)
        self.patch_shape = self.run_conf.get('Params').get('patch_shape')
        self.patch_offset = self.run_conf.get('Params').get('patch_offset')
        self.counter = 1

    def train(self, optimizer=None, data_loader=None, validation_loader=None):

        if validation_loader is None:
            raise ValueError('Please provide validation loader.')
        logger = NNTrainer.get_logger(self.log_file)
        print('Training...')
        for epoch in range(0, self.epochs):
            self.model.train()
            running_loss = 0.0
            self.adjust_learning_rate(optimizer=optimizer, epoch=epoch + 1)
            for i, data in enumerate(data_loader, 0):
                inputs, labels = data['inputs'].to(self.device), data['labels'].to(self.device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                # loss1 = torch.dist(outputs, labels, p=2)
                # loss.backward()
                # optimizer.step()
                # current_loss = loss.item() / labels.numel()
                # running_loss += current_loss
                # print(loss1)
                print(outputs.shape)
                print(labels.shape)

                outputs_input_dis = outputs - data['POS'].float()
                label_input_dis = labels - data['POS'].float()
                print('outputs_input_dis', outputs_input_dis.shape)
                print('label_input_dis', label_input_dis.shape)

                loss = - F.cosine_similarity(outputs_input_dis, label_input_dis, dim=1).mean()
                print('loss', loss)
                loss.backward()
                optimizer.step()
                current_loss = loss
                # / float(len(loss))
                running_loss += current_loss

                if (i + 1) % self.log_frequency == 0:  # Inspect the loss of every log_frequency batches
                    current_loss = running_loss / self.log_frequency if (i + 1) % self.log_frequency == 0 \
                        else (i + 1) % self.log_frequency
                    running_loss = 0.0

                self.flush(logger, ','.join(str(x) for x in [0, 0, epoch + 1, i + 1, 0, 0, 0, 0, current_loss]))
                print('Epochs[%d/%d] Batch[%d/%d] loss:%.5f pre:%.3f rec:%.3f f1:%.3f acc:%.3f' %
                      (epoch + 1, self.epochs, i + 1, data_loader.__len__(), current_loss, 0, 0, 0, 0),
                      end='\r' if running_loss > 0 else '\n')

            self.checkpoint['epochs'] += 1
            if (epoch + 1) % self.validation_frequency == 0:
                self.evaluate(data_loaders=validation_loader, force_checkpoint=self.force_checkpoint, logger=logger,
                              mode='test')
        try:
            logger.close()
        except IOError:
            pass

    def evaluate(self, data_loaders=None, force_checkpoint=False, logger=None, mode=None):
        assert (logger is not None), 'Please Provide a logger'
        self.model.eval()

        print('\nEvaluating...')
        with torch.no_grad():
            print(data_loaders)
            all_loss = 0.0
            for total_images, loader in enumerate(data_loaders, 1):
                img_loss = 0.0
                img_obj = loader.dataset.image_objects[0]
                all_predicted, all_labels, all_pos = [], [], []
                for i, data in enumerate(loader, 0):
                    inputs, labels = data['inputs'].to(self.device), data['labels'].to(self.device)
                    positions = data['POS']
                    outputs = self.model(inputs)
                    outputs = outputs.float() + positions.float()
                    # loss = torch.dist(outputs, labels, p=1)

                    outputs_input_dis = outputs - data['POS'].float()
                    label_input_dis = labels - data['POS'].float()
                    loss = F.cosine_similarity(outputs_input_dis, label_input_dis, dim=1)
                    current_loss = sum(loss) / float(len(loss))
                    img_loss += current_loss
                    all_loss += current_loss

                    if mode == 'test':
                        all_predicted += outputs.clone().cpu().numpy().tolist()
                        all_labels += labels.clone().cpu().numpy().tolist()
                        all_pos += positions.clone().cpu().numpy().tolist()
                    # write to the Train.CSV file
                    self.flush(logger, ','.join(
                        str(x) for x in [img_obj.file_name, 1, self.checkpoint['epochs'], 0, current_loss]))

                if mode is 'test':
                    all_predicted = np.array(np.ceil(np.array(all_predicted)), dtype=int)
                    all_labels = np.array(all_labels, dtype=int)
                    all_pos = np.array(all_pos, dtype=int)
                    all_labels = all_labels + all_pos
                    # all_predicted = all_predicted + all_pos
                    estimated = np.zeros((img_obj.working_arr.shape[0], img_obj.working_arr.shape[1], 3))
                    try:
                        estimated[:, :, 0][all_predicted[:, 0], all_predicted[:, 1]] = 255
                    except:
                        pass
                    estimated[:, :, 1][all_labels[:, 0], all_labels[:, 1]] = 255
                    # estimated[:, :, 1][all_pos[:, 0], all_pos[:, 1]] = 255

                    IMG.fromarray(estimated.astype(np.uint8)).save(
                        os.path.join(self.log_dir, img_obj.file_name.split('.')[0] + str(self.counter) + '.png'))
                    print(img_obj.file_name + ' LOSS: ', all_loss / total_images)
                    self.counter  += 1
        if mode is 'train':
            self._save_if_better(force_checkpoint=force_checkpoint, score=all_loss / total_images)
