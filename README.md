# mario_pytorch
Implementation of Super-Mario-Bros RL through PyTorch

## About
With inspiration from pytorch's own tutorial of gym-super-mario-bros, their tutorial on DQNs, and the video "Deep Q Learning is Simple with PyTorch" by @philtabor, I am currently working on making a more elligible version of Super Mario Bros RL using PyTorch. Reading the PyTorch tutorial was useful but I'm kinda biased towards @philtabor's more concise DQN algorithm. Thus, I wanted to combine the two and try to self-teach RL. Adding to this, I used various other resources, and realized that brthor's youtube video (link in the sources) which follows PyTorch's tutorial was a clear way in understanding how RL works, and decided to follow that.

## Limitations
I'm currently on my MacBook Air with the M1 processor, so there are code adjustments specifically for that. I have not tested this code on other OS's. It's kinda ironic using PyTorch w/o a GPU but I can't use TensorFlow so here I am.

## TODO
*Dates are in MM/DD*
- Create the ~~DQN~~ **(01/03, 11:07pm)**, ~~Mario agent~~ **(Not doing this, instead added functions to the main.py), ~~Replay Memory~~ **(01/04, 4:42pm)**, ~~Learn methods~~ **(01/04, 10:21pm)**
- Implement image transformations, so grayscale & image rescale are probs most important **(Copied PyTorch's so probably want to create my own version)**
- Maybe test out PPO instead of DQN
- Make sure to graph it out, see if there is real "learning" **(Watching the render to see if my Mario is achieving superhuman gameplay)**
- Somehow save the model for future use (i know there is someway to do this idr how)

## Comments
So far, this needs a lot of work. I ran it for probably 3 hours **(01/05, from 12pm ish to around 2:40pm)** to achieve a final average score of 2000, and when I rendered it Mario decided to just get stuck infront of a pipe and didn't move. Reinforcement Learning! The state transformation techniques used follows PyTorch's tutorial on mario which follows DeepMind's paper, however, these techniques are used on Atari games, not nes, so I probably need to find some way to tinker with it. I think all the ideas used are definitely helping, though, so I guess I need to either add more techniques or change the hyperparameters or even the neural network nodes.

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

After you get those installations, run this:
```shell
python3 main.py
```
It will probably take like a good time for the code to run and get an average reward that is good.


When running, some errors such as this might pop up but don't worry about them.
```shell
warn(f"Failed to load image Python extension: {e}")
```

## Sources
Just general info:
- IBM's Introduction to CNN's: https://www.ibm.com/cloud/learn/convolutional-neural-networks
- DQN paper: https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
- Explanation on what parameters to use though I don't really understand it either: https://stats.stackexchange.com/questions/196646/what-is-the-significance-of-the-number-of-convolution-filters-in-a-convolutional

Tutorials that I followed:
- brthor's tutorial (the one that helped me out the most to understand how to create the online vs target functions): https://www.youtube.com/watch?v=NP8pXZdU-5U&t=1842s
- PyTorch Mario Tutorial (straight up copied their environment transformation functions): https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html
- PyTorch DQN Tutorial: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
- DeepLizard's Tutorial: https://deeplizard.com/learn/video/FU-sNVew9ZA