# DRLND_Collaboration_and_Competition

## Project Description
For this project, you will work with the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.

![Unity ML-Agents Tennis Environment](tennis.png)

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores. This yields a single score for each episode.

The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.


## Installation
The project was in a linux server with unityagent=0.4.0 and python 3.6 installed.

1. You may need to install Anaconda and create a python 3.6 environment.
```bash
conda create -n drnd python=3.6
conda activate drnd
```
2. Clone the repository below, navigate to the python folder and install dependencies. Pay attention that the torch=0.4.0 in the requirements.txt is no longer listed in pypi.org, you may leave your current torch and remove the line torch=0.4.0
 ```
git clone https://github.com/udacity/deep-reinforcement-learning.git
cd deep-reinforcement-learning/python
pip install .
```
3. Download unity environment file  [tennis](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip). This is the "headless" version of the environment. You will not be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.
(To watch the agent, you may follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the [environment for the Linux operating system](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip).)

5. Unzip the compressed file
6. Create the Ipython kernel:
```bash
python -m ipykernel install --user --name=drnd
```

   
## Executing 
In the notebook, be sure to change the kernel to match "drnd" by using the drop down in "Kernel" menu. Be sure the adjust the Tennis file location locally.

Executing Tennis.ipynb
  
