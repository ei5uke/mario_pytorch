# mario_pytorch
My Journey (or I guess Diary) in learning RL through Mario and PyTorch

## About
*v0.2*, 01/08/2021 - current\
Following implementation of Youtuber "Jack of Some"'s implementation of RL through his 'DQN in PyTorch Stream _ of N' series, Deep RL Hands-on book by Maxim Lapan, and following the DQN paper. It's kinda repetitive, redoing the project again but while the first time I was easily able to understand the replay buffer, I didn't really understand anything else. So far, I'm more clear on the DQN algorithm itself, how PyTorch CNN's work, etc. 
\
While I'm more confident in understanding essentially everything, I still have really low scores in breakout so I'm trying out DDQN and other methods.

*v0.1*\
With inspiration from pytorch's own tutorial of gym-super-mario-bros, their tutorial on DQNs, and the video "Deep Q Learning is Simple with PyTorch" by @philtabor, I am currently working on making a more elligible version of Super Mario Bros RL using PyTorch. Reading the PyTorch tutorial was useful but I'm kinda biased towards @philtabor's more concise DQN algorithm. Thus, I wanted to combine the two and try to self-teach RL. Adding to this, I used various other resources, and realized that brthor's youtube video (link in the sources) which follows PyTorch's tutorial was a clear way in understanding how RL works, and decided to follow that.

## Limitations
I'm currently on my MacBook Air with the M1 processor, so there are code adjustments specifically for that. I have not tested this code on other OS's. It's kinda ironic using PyTorch w/o a GPU but I can't use TensorFlow so here I am.

## TODO
*Dates are in mm/dd*\
*v0.2*
- ~~Create the Model~~ (1/8)
- ~~Create the Memory~~ (1/8)
    - ~~Swap to DDQN~~ (1/14)
    - Implement A2C
    - Implement PPO
    - Implement Rainbow
- ~~Create the Train step~~ (1/9 technically)
    - ~~Calculate loss~~ (1/10), might switch to Huber or Smoothl1loss depending on how bad MSE is 
    - Clipping the error term or the gradient I have no idea which one
- Create the frame transitions
    - ~~grayscale~~ (1/11)
    - ~~framestacking~~ (1/11)
    - ~~frame skipping~~ (1/11)
    - ~~image scaling~~ (1/11)
    - need to think of more transitions though
- Create main run file
    - ~~Main run loop~~ (1/12)
    - ~~Save model~~ (1/12)

*v0.1*
- Create the ~~DQN~~ **(01/03, 11:07pm)**, ~~Mario agent~~ **(Not doing this, instead added functions to the main.py), ~~Replay Memory~~ **(01/04, 4:42pm)**, ~~Learn methods~~ **(01/04, 10:21pm)**
- Implement image transformations, so grayscale & image rescale are probs most important **(Copied PyTorch's so probably want to create my own version)**
- Maybe test out PPO instead of DQN
- Make sure to graph it out, see if there is real "learning" **(Watching the render to see if my Mario is achieving superhuman gameplay, also using wandb)**
- ~~Somehow save the model for future use (i know there is someway to do this idr how)~~ **(01/08)**

## Diary(?) or Comments I guess
*01/14/2022*\
I followed the DDQN algorithm and within like 5 hrs, or like around 20 epochs (which each epoch consists of 50000 parameter updates), it got an average score of 17, which although is much better than what I was doing before, it plateaus from there because epsilon is 0.1. Still need much tinkering but I think so far it's good. The DDQN algorithm is clearly working but now I need to figure out how often to consider an epoch and maybe somehow tinker more with the loss function possibly. I've uploaded updated code but its for breakout; when I get good scores on breakout, I'll try it on mario. 

![ddqn_1](https://user-images.githubusercontent.com/55261758/149607611-408420a3-46ac-4862-acae-798ffe52119c.png)

*01/13/2022*\
I'm trying out DDQN following the paper published by DeepMind but I'm trying out my own hyperparameters to speed up training. I'm trying it out with breakout first for now and it really sucks. I've ran breakout to train overnight and it reached an average score of 20-ish when the average score should be around 300+ I believe. Definitely there has to be a way to make this better, I mean the OpenAI Retro Contest featuring Sonic had solutions that learned within 2 hrs, so there has to be a way to make this better not using 8 hrs. For now, it seems I'll try to tinker with all this with breakout, and then whatever new techniques/tricks I implement I'll apply them to mario.

*01/10/2022*\
Ok so I'm like a good 80% there in understanding the train step of DQN. I understand how we unpack the transition tuple, why we unsqueeze and gather, and how to calculate the loss. The remaining 20% is to understand where to place the zero_grad() and figure out how to clip the error. I've heard from tutorials that there is a huge debate on the confusion of what to clip, so I guess I'll try to learn where to do so.

*01/08/2022*\
This repo feels more like a blog series or diary of my journey learning RL... Maybe I change it to that sometime later to seriously showcase my progress. Anyway, yea so I've noticed a few things: 1) definitely read the published journals 2) maybe find if there are multiple versions of journals. It looks like DeepMind has published two papers: "Playing Atari with Deep RL" and "Human-level control through deep rl" and it seems that most tutorials and guides follow the second one. So yea, probably look if there are multiple papers because that almost caught me off guard. Also, reading PyTorch's documentation on nn.Conv2d was helpful in learning what the output of the CNN is because we need that to connect it to the FC layers which (I may be wrong about this) isn't written in the paper. 

*01/06/2022*\
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
*torch.unsqueeze(x, n)*\
spaces the given tensor out by the given **n**. If n >= 0, then 0 spaces it out by nothing, while 1 spaces it out the items once, 2 spaces it out twice, so on and so forth.\
e.g. x = torch.Tensor([1, 2], [3, 4])
- torch.unsqueeze(x, 0) -> tensor([1, 2], [3, 4]) # aka nothing changed
- torch.unsqueeze(x, 1) -> tensor([[1, 2]], [], [[3, 4]]) # aka a dimension is added between to space things out
- torch.unsqueeze(x, 2) -> tensor([[[1], [2]], [], [[3], [4]]]) # everything is more spaced

*torch.squeeze(x)* and *torch.unsqueeze(x, n)*\
squeezes depending on input if there is none just squeeze\
e.g. x = torch.zeros(2, 1, 2, 1, 2)
- y = torch.squeeze(x)\
y.size() # torch.size([2, 2, 2]), so it removed all dimensions of 1\
- y = torch.squeeze(x, 0)\
y.size() # torch.size([2, 1, 2, 1, 2]) nothing changed bc dimension 0 isn't size 1\
- y = torch.squeeze(x, 1)\
y.size() # torch.size([2, 2, 1, 2]) it got removed because its dimension is size 1\

*torch.gather(input, dim, index)*\
honestly just read this: https://stackoverflow.com/questions/50999977/what-does-the-gather-function-do-in-pytorch-in-layman-terms
\
\
*Hadamard Product*
- When using * to find the product of 2+ tensors, it outputs the hadamard product, aka inplace element multiplication. Thus, the tensors must have the same exact dimensions.

## Sources
Just general info:
- Sutton and Barto's RL book; probably want to go to at least the MDP part to even barely understand DQN's
- Deep RL Hands-on by Maxim Lapan; the github with the source code (though I believe it's outdated so I prefer the code of the textbook): https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On
- IBM's Introduction to CNN's: https://www.ibm.com/cloud/learn/convolutional-neural-networks
- DDQN paper: https://arxiv.org/pdf/1509.06461.pdf
- **Better** DQN paper: https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf
- DQN paper: https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
- Explanation of zerograd: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch
- Epoch vs Episode: https://stats.stackexchange.com/questions/250943/what-is-the-difference-between-episode-and-epoch-in-deep-q-learning

Tutorials that I followed:
- Jack of Some's tutorial: https://www.youtube.com/watch?v=WVBp4Cj2lXo&list=PLd_Oyt6lAQ8Q0MaTG41iwPdy9GQmoz8dG
- brthor's tutorial: https://www.youtube.com/watch?v=NP8pXZdU-5U&t=1842s
- Fabio M. Graetz tutorial (though it's in Tensorflow): https://towardsdatascience.com/tutorial-double-deep-q-learning-with-dueling-network-architectures-4c1b3fb7f756
- PyTorch Mario Tutorial: https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html
- PyTorch DQN Tutorial: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
- DeepLizard's Tutorial: https://deeplizard.com/learn/video/FU-sNVew9ZA