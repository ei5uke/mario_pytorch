# mario_pytorch
My Journey (or I guess Diary) in learning RL through Mario and PyTorch

## About
*v0.2*, 01/08/2021 - current
Following implementation of Youtuber "Jack of Some"'s implementation of RL through his 'DQN in PyTorch Stream _ of N' series, Deep RL Hands-on book by Maxim Lapan, and following the DQN paper. It's kinda repetitive, redoing the project again but while the first time I was easily able to understand the replay buffer, I didn't really understand anything else. So far, I'm more clear on the DQN algorithm itself, how PyTorch CNN's work, etc.

*v0.1*
With inspiration from pytorch's own tutorial of gym-super-mario-bros, their tutorial on DQNs, and the video "Deep Q Learning is Simple with PyTorch" by @philtabor, I am currently working on making a more elligible version of Super Mario Bros RL using PyTorch. Reading the PyTorch tutorial was useful but I'm kinda biased towards @philtabor's more concise DQN algorithm. Thus, I wanted to combine the two and try to self-teach RL. Adding to this, I used various other resources, and realized that brthor's youtube video (link in the sources) which follows PyTorch's tutorial was a clear way in understanding how RL works, and decided to follow that.

## Limitations
I'm currently on my MacBook Air with the M1 processor, so there are code adjustments specifically for that. I have not tested this code on other OS's. It's kinda ironic using PyTorch w/o a GPU but I can't use TensorFlow so here I am.

## TODO
*Dates are in MM/DD*
*v0.2*
- ~~Create the Model~~ (1/8)
- ~~Create the Memory~~ (1/8)
- ~~Create the Train step~~ (1/9 technically)
    - I've completed it for the most part because I'm not sure how to implement the clipping of the Squared Error term between [-1, 1] bc it's squared so when will it ever be negative?
- Create the frame transitions
    - grayscale
    - framestacking
    - frame skipping
    - image scaling
- Create main run file
    - Main run loop
    - Save model

*v0.1*
- Create the ~~DQN~~ **(01/03, 11:07pm)**, ~~Mario agent~~ **(Not doing this, instead added functions to the main.py), ~~Replay Memory~~ **(01/04, 4:42pm)**, ~~Learn methods~~ **(01/04, 10:21pm)**
- Implement image transformations, so grayscale & image rescale are probs most important **(Copied PyTorch's so probably want to create my own version)**
- Maybe test out PPO instead of DQN
- Make sure to graph it out, see if there is real "learning" **(Watching the render to see if my Mario is achieving superhuman gameplay, also using wandb)**
- ~~Somehow save the model for future use (i know there is someway to do this idr how)~~ **(01/08)**

## Comments
*01/08/2021*
This repo feels more like a blog series or diary of my journey learning RL... Maybe I change it to that sometime later to seriously showcase my progress. Anyway, yea so I've noticed a few things: 1) definitely read the published journals 2) maybe find if there are multiple versions of journals. It looks like DeepMind has published two papers: "Playing Atari with Deep RL" and "Human-level control through deep rl" and it seems that most tutorials and guides follow the second one. So yea, probably look if there are multiple papers because that almost caught me off guard. Also, reading PyTorch's documentation on nn.Conv2d was helpful in learning what the output of the CNN is because we need that to connect it to the FC layers which (I may be wrong about this) isn't written in the paper. 

*01/06/2021*
So far, this needs a lot of work. I ran it for probably 3 hours **(01/05, from 12pm ish to around 2:40pm)** to achieve a final average score of 2000, and when I rendered it Mario decided to just get stuck infront of a pipe and didn't move. Reinforcement Learning! The state transformation techniques used follows PyTorch's tutorial on mario which follows DeepMind's paper, however, these techniques are used on Atari games, not nes, so I probably need to find some way to tinker with it. I think all the ideas used are definitely helping, though, so I guess I need to either add more techniques or change the hyperparameters or even the neural network nodes.

## Installation and Running
First:
```shell
git clone https://github.com/ei5uke/mario_pytorch.git
cd mario_pytorch
cd v0.1 # v0.1 works albeit being very inefficient.
```

You don't have to follow these next commands it's a modification for my computer, explanation to why is in those repositories. You still want to download PyTorch but follow that on the PyTorch installation page depending on if you want to use a GPU or don't have one. If you don't have the same issues as me, then make sure to download nes-py, gym-super-mario-bros, and pyglet normally through pip.
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

## Notes on PyTorch things that took me an extra second to understand
*torch.unsqueeze(x, n)* 
spaces the given tensor out by the given **n**. If n >= 0, then 0 spaces it out by nothing, while 1 spaces it out the items once, 2 spaces it out twice, so on and so forth.
e.g. x = torch.Tensor([1, 2], [3, 4])
- torch.unsqueeze(x, 0) -> tensor([1, 2], [3, 4]) # aka nothing changed
- torch.unsqueeze(x, 1) -> tensor([[1, 2]], [], [[3, 4]]) # aka a dimension is added between to space things out
- torch.unsqueeze(x, 2) -> tensor([[[1], [2]], [], [[3], [4]]]) # everything is more spaced

*torch.squeeze(x)* and *torch.unsqueeze(x, n)*
squeezes depending on input if there is none just squeeze
e.g. x = torch.zeros(2, 1, 2, 1, 2)
y = torch.squeeze(x)
y.size() # torch.size([2, 2, 2]), so it removed all dimensions of 1
y = torch.squeeze(x, 0)
y.size() # torch.size([2, 1, 2, 1, 2]) nothing changed bc dimension 0 isn't size 1
y = torch.squeeze(x, 1)
y.size() # torch.size([2, 2, 1, 2]) it got removed because its dimension is size 1

## Sources
Just general info:
- Sutton and Barto's RL book; probably want to go to at least the MDP part to even barely understand DQN's
- Deep RL Hands-on by Maxim Lapan; the github with the source code (though I believe it's outdated so I prefer the code of the textbook): https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On
- IBM's Introduction to CNN's: https://www.ibm.com/cloud/learn/convolutional-neural-networks
- **Better** DQN paper: https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf
- DQN paper: https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
- Explanation of zerograd: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch

Tutorials that I followed:
- Jack of Some's tutorial: https://www.youtube.com/watch?v=WVBp4Cj2lXo&list=PLd_Oyt6lAQ8Q0MaTG41iwPdy9GQmoz8dG
- brthor's tutorial: https://www.youtube.com/watch?v=NP8pXZdU-5U&t=1842s
- PyTorch Mario Tutorial: https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html
- PyTorch DQN Tutorial: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
- DeepLizard's Tutorial: https://deeplizard.com/learn/video/FU-sNVew9ZA