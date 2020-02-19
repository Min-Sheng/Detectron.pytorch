import os
import numpy as np
from PIL import Image, ImageFont, ImageDraw, ImageOps
import torch
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import utils.blob as blob_utils

class Logger:
    """The base class for all loggers.
    Args:
        log_dir (str): The saved directory.
    """
    def __init__(self, log_dir):
        
        self.writer = SummaryWriter(log_dir)

    def close(self):
        """Close the writer.
        """
        self.writer.close()

    def _add_images(self, step, input_data):
        """Plot the visualization results.
        Args:
            step (int): The number of the step.
            input_data (dict): The sample batch.
        """
        inv_normalize = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
            std=[1/0.229, 1/0.224, 1/0.255]
        )
        to_tensor = transforms.ToTensor()

        im_batched = input_data['data'][0].float()
        query_batched = input_data['query'][0][0].float()
        im_info_batched = input_data['im_info'][0].float()
        roidb_batched = list(map(lambda x: blob_utils.deserialize(x)[0], input_data['roidb'][0]))

        im_info = im_info_batched[0]
        im_scale = im_info.data.numpy()[2]

        gt_boxes = roidb_batched[0]['boxes'] * im_scale

        im = inv_normalize(im_batched[0]).permute(1, 2, 0).data.numpy()
        im = (im - im.max()) / (im.max() - im.min())
        im = (im *255).astype(np.uint8)
        im = Image.fromarray(im)

        query = inv_normalize(query_batched[0]).permute(1, 2, 0).data.numpy()
        query = (query - query.max()) / (query.max() - query.min())
        query = (query *255).astype(np.uint8)
        query = Image.fromarray(query)

        query_w, query_h = query.size
        query_bg = Image.new('RGB', (im.size), (0, 0, 0))
        bg_w, bg_h = query_bg.size
        offset = ((bg_w - query_w) // 2, (bg_h - query_h) // 2)
        query_bg.paste(query, offset)

        im_gt_bbox = im.copy()
        for bbox in gt_boxes:
            if bbox.sum().item()==0:
                break
            bbox = tuple(list(map(int,bbox.tolist())))
            draw = ImageDraw.Draw(im_gt_bbox)
            draw.rectangle(bbox, fill=None, outline=(0, 110, 255), width=2)
        
        train_grid = [to_tensor(im), to_tensor(query_bg), to_tensor(im_gt_bbox)]
        train_grid = make_grid(train_grid, nrow=2, normalize=True, scale_each=True, pad_value=1)
        self.writer.add_image("logs/train", train_grid, step)
        