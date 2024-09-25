## Spiking Wavelet Transformer (ECCV-2024)

Spiking Wavelet Transformer, ECCV'24: [[Paper]](https://arxiv.org/pdf/2403.11138). 
![eccv_poster-1](https://github.com/user-attachments/assets/480336c7-7b50-47c2-bd89-ffd4ce54aa92)

<div align="center"> <img src="https://github.com/bic-L/Spiking-Wavelet-Transformer/blob/main/figures/fig1.png"  width="810" height="270" alt="acc"/> </div>
  
### Key contributions

The "***Spiking Wavelet Transformer***" introduces an innovative approach to enhancing spiking neural networks (SNNs) by integrating wavelet transforms with transformer architectures in an attention-free fashion. This method addresses the challenge of effectively capturing high-frequency patterns crucial for event-driven vision tasks, unlike self-attention, which prioritizes low-frequency elements. Key features include:

- **Frequency-Aware Token Mixer (FATM):** Learns spatial-frequency features without relying on self-attention.
- **Spiking Frequency Representation:** Getting robust frequency representation efficiently in a spike-driven, multiplication-free manner.
- **Enhanced Performance:** Offers improved accuracy and reduced parameter count on datasets like ImageNet.

This approach provides a practical solution for advancing energy-efficient, event-driven computing.

### Implementation
[Checkpoints for ImageNet](https://github.com/bic-L/Spiking-Wavelet-Transformer/releases/tag/checkpoints)

For more details on our training, please check out our paper and supplementary material. (Note: for Imagenet, we used 8×A800 GPU cards for training, total batch size = 512 )

#### Requirement:

Make sure your PyTorch version is 2.0.0 or higher. For more information, please visit [link](https://pytorch.org/get-started/previous-versions/) for details

```bash
  pip install timm==0.6.12 spikingjelly==0.0.0.0.12 opencv-python==4.8.1.78 wandb einops PyYAML Pillow six torch
```

#### Running the code

Please check the bash file in each folder (cifar10-100, event, imagenet).
