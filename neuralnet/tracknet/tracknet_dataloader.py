"""
BY Aashis Khanal
"""
import os
import random
from random import shuffle

import numpy as np
import torch

import utils.img_utils as imgutils
from neuralnet.datagen import Generator
import math
from commons.MAT import Mat
from commons.IMAGE import Image

sep = os.sep


class PatchesGenerator(Generator):
    def __init__(self, **kwargs):
        super(PatchesGenerator, self).__init__(**kwargs)
        self.patch_shape = self.run_conf.get('Params').get('patch_shape')
        self.patch_pad = self.run_conf.get('Params').get('patch_pad')
        self.patch_offset = self.run_conf.get('Params').get('patch_offset')
        self.k_half = int(math.floor(self.patch_shape[0] / 2))
        self._load_indices()
        print('Patches:', self.__len__())

    def _load_indices(self):
        for ID, img_file in enumerate(self.images):
            mat_file = Mat(mat_file=self.image_dir + sep + img_file)

            V = mat_file.get_graph('V')
            A = mat_file.get_graph('A')
            I = mat_file.get_image('I')

            # print('V', V)

            img_obj = Image()
            img_obj.file_name = img_file
            img_obj.image_arr = I
            img_obj.working_arr = I[:, :, 1]
            img_obj.load_mask(self.mask_dir, self.mask_getter)
            self.image_objects[ID] = img_obj

            path_index = mat_file.get_graph('pathNode')
            vessel_pathidx = np.where(path_index==1)[0]
            u_pos_input = V[vessel_pathidx, :]
            print('vessel_pathidx', vessel_pathidx)
            # b = np.where(A[:, vessel_pathidx])[0]
            b = vessel_pathidx.copy()
            for i, src in enumerate(vessel_pathidx):
                b[i] = np.where(A[src, :])[0][0]
            # b = np.where(A[vessel_pathidx, :])[0]
            # print('b', b)
            b_pos_output = V[b, :]

            u_pos_input = u_pos_input.astype(np.int)
            b_pos_output = b_pos_output.astype(np.int)
            # print('----------------<<<<<<>>>>>>>>---------------')
            # # print(u_pos_input - b_pos_output)
            # print('----------------????????????????---------------')
            # print(b_pos_output.shape)
            for (i, j), output in zip(u_pos_input, b_pos_output - u_pos_input):
                # print('>>>>>>>>>>>', i, j, output)
                # print('<<<<<<<<<<<', u_pos_input)
                # print('<<<>>>>', b_pos_output)
                row_from, row_to = i - self.k_half, i + self.k_half + 1
                col_from, col_to = j - self.k_half, j + self.k_half + 1
                if row_from < 0 or col_from < 0:
                    continue
                if row_to >= img_obj.working_arr.shape[0] or col_to >= img_obj.working_arr.shape[1]:
                    continue
                if np.isin(0, img_obj.mask[row_from:row_to, col_from:col_to]):
                    continue

                self.indices.append([ID, [i, j], output.tolist()])

    def __getitem__(self, index):
        ID, (i, j), out = self.indices[index]
        # print('i , j, out', i, j, out)

        row_from, row_to = i - self.k_half, i + self.k_half + 1
        col_from, col_to = j - self.k_half, j + self.k_half + 1

        row_from = int(row_from)
        row_to = int(row_to)
        col_from = int(col_from)
        col_to = int(col_to)
        img_tensor = self.image_objects[ID].working_arr[row_from:row_to, col_from:col_to][..., None]

        if self.transforms is not None:
            img_tensor = self.transforms(img_tensor)

        return {'IDs': ID, 'POS': np.array([i, j]), 'inputs': np.array(img_tensor), 'labels': torch.FloatTensor(out)}
