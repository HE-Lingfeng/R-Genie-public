# R-Genie: Reasoning-Guided Generative Image Editing
[![](https://img.shields.io/badge/Project-Page-blue)](https://dongzhang89.github.io/RGenie.github.io/)

## ⭐Introduction
R-Genie addresses the challenges that current image editing methods remain constrained by explicit textual instructions and limited editing operations, lacking deep comprehension of implicit user intentions and contextual reasoning. R-Genie is a reasoning-guided generative image editor, which synergizes the generation power of diffusion models with advanced reasoning capabilities of multimodal large language models. R-Genie incorporates a reasoning-attention mechanism to bridge linguistic understanding with visual synthesis, enabling it to handle intricate editing requests involving abstract user intentions and contextual reasoning relations.
![Fig3](https://github.com/user-attachments/assets/2d23f58a-cdc7-4257-ad16-84eae99ed0d8)

## ⭐Dataset Samples

![Figure2](https://github.com/user-attachments/assets/f252ea4c-6870-45a3-8b6d-1cfe9157fff4)

## ⭐Results

![Figure4](https://github.com/user-attachments/assets/e7aba913-0ff7-4fec-bcde-b311d6f92369)

## 🔧Requirements
**Download datasets:**

You can download our datasets on [BaiduCloud](https://pan.baidu.com/s/1eN0YdcnxqmxgcrQjzyyw-Q?pwd=pzpz).

-Note that you have to rewrite the dataloader file to adapt the rearranged dataset file structure.

Other components of our model can also be download from huggingface, as follows:
- [Show-o](https://huggingface.co/showlab/show-o)
- [MAGVIT-V2](https://huggingface.co/showlab/magvitv2)
- [Phi-1.5](https://huggingface.co/microsoft/phi-1_5)

## 🔨How to train
```
deepspeed --master_port=24999 training/train_edit.py config=your_train_config_path
```

