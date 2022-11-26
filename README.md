# Learned Motion Matching (Work in progress)

![LMM2](https://user-images.githubusercontent.com/55563608/150857265-d90b95f7-32fa-4613-a86c-dc3d4c73b397.gif)

A neural-network-based generative model for character animation.

The system takes user controls as input to automatically produce high quality motions that achieves the desired target. Implemented using Pytorch.

Following Ubisoft La-Forge [paper](https://dl.acm.org/doi/abs/10.1145/3386569.3392440).

<!-- ## Table of contents
- [How it works](#how-it-works)
- [Installation](#installation)
- [Usage](#usage)
- [Important notes](#important-notes) -->

## How it works

<details>
	<summary>---------</summary>
       
Currently, this project can be separated in two parts: 
* Unity: Extract all character animations information and store in three files: XData.txt, YData.txt and HierarchyData.txt;
* Pytorch: Using above generated datas, neural network models are trained.

After training, .onnx files are generated and exported to Unity, where the neural nets inference can be run using [Barracuda](https://medium.com/@a.abelhopereira/how-to-use-pytorch-models-in-unity-aa1e964d3374).

### XData.txt
This file consist of C blocks, F lines and M columns. C is the number of clips; Fi is the number of frames of clip i; M is the number of features (Described [here](https://dl.acm.org/doi/pdf/10.1145/3386569.3392440?casa_token=vfgWm5NZnE0AAAAA:LpyNyvcno0zSmbZETgY_q2jM3oeBGvC2QLTc-1383m4V2pnxkxR39P3XUllljGk4-91rB84Nn9fA), section 3:  BASIC MOTION MATCHING).
Each block C is separated by a empty line. 

Let's consider the following animation database: 
```
C = 2, F[0] = 3, F[1] = 4, M = 24
```

XData.txt should be in this format (illustrative values):
```
-8.170939E-08 0 0 -1.634188E-07 0 0 -2.451281E-07 0 0 0 -3.773226E-05 0 1.117587E-10 4.470348E-11 -0.001392171 0 0 0 -6.705522E-11 3.352761E-11 -0.001392171 0 0 0
-8.579486E-08 0 0 -1.675043E-07 0 0 -2.492136E-07 0 0 0 -3.773226E-05 0 1.117587E-10 4.470348E-11 -0.001392171 0 0 0 -6.705522E-11 3.352761E-11 -0.001392171 0 0 0
-8.988033E-08 0 0 -1.715897E-07 0 0 -2.532991E-07 0 0 0 -4.085493E-05 0 1.117587E-10 4.470348E-11 -0.001392171 0 0 0 -6.705522E-11 3.352761E-11 -0.001392171 0 0 0

-8.170939E-08 0 0 -1.634188E-07 0 0 -2.451281E-07 0 0 0 -3.773226E-05 0 1.117587E-10 4.470348E-11 -0.001392171 0 0 0 -6.705522E-11 3.352761E-11 -0.001392171 0 0 0
-8.579486E-08 0 0 -1.675043E-07 0 0 -2.492136E-07 0 0 0 -3.773226E-05 0 1.117587E-10 4.470348E-11 -0.001392171 0 0 0 -6.705522E-11 3.352761E-11 -0.001392171 0 0 0
-8.988033E-08 0 0 -1.715897E-07 0 0 -2.532991E-07 0 0 0 -4.085493E-05 0 1.117587E-10 4.470348E-11 -0.001392171 0 0 0 -6.705522E-11 3.352761E-11 -0.001392171 0 0 0
-8.988033E-08 0 0 -1.715897E-07 0 0 -2.532991E-07 0 0 0 -4.085493E-05 0 1.117587E-10 4.470348E-11 -0.001392171 0 0 0 -6.705522E-11 3.352761E-11 -0.001392171 0 0 0

```

### YData.txt
Similar to XData, but M is the number of pose information

### HierarchyData.txt
This file stores the character hierarchy to generate Forward Kinematcs for Pytorch usage. Consists of N lines, the number of joints of our character. Each line refers to it parent joint, except the root, which is 0.

Let's consider the following rig hierarchy:
```
       root
         |
      spine_01
        / \ 
  leg_l    leg_r
```

HierarchyData.txt should be:
```
0
0
1
1

```
</details>

## Installation
<details>
	<summary>---------</summary>

- Download the Source code from the latest tag <a href="https://github.com/pau1o-hs/Learned-Motion-Matching/tags">here</a>
- Download the Unity sample project from the latest tag <a href="https://github.com/pau1o-hs/Learned-Motion-Matching/tags">here</a>.

\* Install the **Barracuda** package inside of Unity’s Package Manager (Window->Package Manager)
</details>

## Usage
<details>
	<summary>---------</summary>
       
Currently, for use this system, the user needs to do the following steps:
       
#### Unity
1. Add the desired animation clips in the character Animator tab;
2. Add and setup the Gameplay script to the desired character;
3. Hit the "Extract data from animator" button, located the Inspector of Gameplay script;
4. Export "XData", "YData" and "HierarchyData" previously generated to Pytorch "/database" folder;

#### Pytorch
5. Run decompressor.py, followed by stepper.py and projector.py (this last two can be run in parallel);
6. Export the ONNX files generated in Pytorch environment to Unity's "/Assets/Motion Matching/ONNX " folder;
7. Export the "QData.txt" and "ZData.txt" file generated in Pytorch environment to Unity's "/Assets/Motion Matching/Database" folder;

#### Unity
8. Hit "Play" button and play.

![githubimg1](https://user-images.githubusercontent.com/55563608/139554182-2e4c9f23-ff1c-4ea4-971b-402f1fd7c197.png)
  
</details>

## Important notes

<details>
	<summary>---------</summary>
If you try to use it with your own character and animations, there are some details:
<br>
       
-  All your character's bones scales must be (1, 1, 1) to ForwardKinematics method works properly;
-  Every animation clip must have at least 60 frames;
-  The last 60 frames of every animation clip must have the same trajectory directions, because as input to the neural networks, are passed the **future** 60 frames.
