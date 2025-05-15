from torch.utils.data import Dataset
from torchvision import transforms
import transformers
import os
import json
from PIL import Image
import torch
import torch.distributed as dist
from enum import Enum
import numpy as np

# def resize(f):
#     w, h = f.size
#     if w>h:
#         p = (w-h)//2
#         f = f.crop([p, 0, p+h, h])
#     elif h>w:
#         p = (h-w)//2
#         f = f.crop([0, p, w, p+w])
#     f = f.resize([512, 512])
#     return f

def image_transform(image, resolution=512, normalize=True):
    image = transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC)(image)
    image = transforms.CenterCrop((resolution, resolution))(image)
    image = transforms.ToTensor()(image)
    if normalize:
        image = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)(image)
    return image


class EditingDataset(Dataset):
    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer) -> None:
        self.data_path = data_path
        self.tokenizer = tokenizer
        current_path = os.getcwd()
        print(f"Current Working Directory: {current_path}")
        with open(os.path.join(data_path, 'editing_instruction_dict.json')) as f:
            edit_instruction_dict = json.load(f)
            self.edit_instruction_dict = edit_instruction_dict
        self.images = os.listdir(os.path.join(data_path, 'imgs'))
        print(f"There are {len(self.images)} images!")
        print(f"There are {len(self.edit_instruction_dict)} instructions!")    
    def __len__(self):
        return len(self.edit_instruction_dict)

    def __getitem__(self, i):
        img_name = self.images[i]
        img_id = img_name.split('_')[-1].split('.')[0].lstrip('0') #datatype: str
        instruction = self.edit_instruction_dict[img_id]["instruction"]
        image = image_transform(image=Image.open(os.path.join(self.data_path, 'imgs', img_name)))
        target = image_transform(image=Image.open(os.path.join(self.data_path, 'gt', img_name)))

        return (img_name,
                img_id,
                instruction,
                image,
                target)
    
    def collate_fn(self, batch):
        img_names, img_ids, instructions, images, targets = zip(*batch)
        return (
            list(img_names),
            list(img_ids),
            list(instructions),
            torch.stack(images),
            torch.stack(targets)
        )

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if isinstance(self.sum, np.ndarray):
            total = torch.tensor(
                self.sum.tolist()
                + [
                    self.count,
                ],
                dtype=torch.float32,
                device=device,
            )
        else:
            total = torch.tensor(
                [self.sum, self.count], dtype=torch.float32, device=device
            )

        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        if total.shape[0] > 2:
            self.sum, self.count = total[:-1].cpu().numpy(), total[-1].cpu().item()
        else:
            self.sum, self.count = total.tolist()
        self.avg = self.sum / (self.count + 1e-5)

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ""
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.3f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.3f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.3f}"
        else:
            raise ValueError("invalid summary type %r" % self.summary_type)

        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(" ".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def dict_to_cuda(input_dict):
    for k, v in input_dict.items():
        if isinstance(input_dict[k], torch.Tensor):
            input_dict[k] = v.cuda(non_blocking=True)
        elif (
            isinstance(input_dict[k], list)
            and len(input_dict[k]) > 0
            and isinstance(input_dict[k][0], torch.Tensor)
        ):
            input_dict[k] = [ele.cuda(non_blocking=True) for ele in v]
    return input_dict