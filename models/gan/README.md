# GAN Model Checkpoints

This directory contains saved checkpoints for the GAN models used in synthetic data generation.

## Checkpoint Format
Each checkpoint file contains:
- Generator state dict
- Discriminator state dict
- Generator optimizer state
- Discriminator optimizer state
- Training epoch information

## Usage
Checkpoints are automatically saved every 10 epochs during training.
The naming format is: `gan_checkpoint_epoch_X.pt` where X is the epoch number.

## Loading Checkpoints
```python
checkpoint = torch.load('gan_checkpoint_epoch_100.pt')
generator.load_state_dict(checkpoint['generator_state_dict'])
discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
``` 