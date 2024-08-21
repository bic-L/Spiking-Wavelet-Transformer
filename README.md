## Spiking Wavelet Transformer (ECCV-2024)

Spiking Wavelet Transformer, ECCV'24: [[Paper]](https://arxiv.org/pdf/2403.11138). 

<div align="center"> <img src="https://github.com/bic-L/Spiking-Wavelet-Transformer/blob/main/figures/fig1.png"  width="810" height="270" alt="acc"/> </div>
  
### Key contributions

The "***Spiking Wavelet Transformer***" introduces an innovative approach to enhancing spiking neural networks (SNNs) by integrating wavelet transforms with transformer architectures in an attention-free fashion. This method addresses the challenge of effectively capturing high-frequency patterns crucial for event-driven vision tasks, unlike self-attention, which prioritizes low-frequency elements. Key features include:

- **Frequency-Aware Token Mixer (FATM):** Learns spatial-frequency features without relying on self-attention.
- **Spiking Frequency Representation:** Getting robust frequency representation efficiently in a spike-driven, multiplication-free manner.
- **Enhanced Performance:** Offers improved accuracy and reduced parameter count on datasets like ImageNet.

This approach provides a practical solution for advancing energy-efficient, event-driven computing.

### Running the Code
