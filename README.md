# mario_pytorch
Implementation of Super-Mario-Bros RL through PyTorch

## About
With inspiration from pytorch's own tutorial of gym-super-mario-bros, their tutorial on DQNs, and the video "Deep Q Learning is Simple with PyTorch" by @philtabor, I am currently working on making a more elligible version of Super Mario Bros RL using PyTorch. Reading the PyTorch tutorial was useful but I'm kinda biased towards @philtabor's more concise DQN algorithm. Thus, I wanted to combine the two and try to self-teach RL.

## Limitations
I'm currently on my MacBook Air with the M1 processor, so there are code adjustments specifically for that. I have not tested this code on other OS's. It's kinda ironic using PyTorch w/o a GPU but I can't use TensorFlow so here I am.

## TODO
- Create the DQN, Mario agent, Replay Memory, Learn methods
- Implement image transformations, so grayscale & image rescale are probs most important
- Maybe test out PPO instead of DQN
- Make sure to graph it out, see if there is real "learning"
- Somehow save the model for future use (i know there is someway to do this idr how)

## Installation
First:
```shell
git clone https://github.com/ei5uke/mario_pytorch.git
cd mario_pytorch
```

You don't have to follow these next commands it's a modification for my computer, explanation to why is in those repositories. You probably still want to download PyTorch but follow that on the PyTorch installation page depending on if you want to use a GPU or don't have one.
```shell
pip install git+https://github.com/ei5uke/nes-py-macm1
pip install git+https://github.com/ei5uke/gym-super-mario-bros-macm1
pip install --user --upgrade git+http://github.com/pyglet/pyglet@pyglet-1.5-maintenance
pip install torch torchvision torchaudio
```

The remaining part is tbd but I'll probably make some kind of main.py file which you just run 
```shell
python3 main.py
```
