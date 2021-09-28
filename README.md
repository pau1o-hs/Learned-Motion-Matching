# Learned Motion Matching (Work in progress)
A neural-network-based generative model for character animation

The system takes user controls as input to automatically produce high quality motions that achieves the desired target. Implemented using Pytorch.

Following Daniel Holden's paper: http://theorangeduck.com/page/learned-motion-matching#

## How it works
Currently, this project can be separated in two parts: 
* Unity: Extract all character animations information and store in three files: XData.txt, YData.txt and HierarchyData.txt (I should extract it using .bvh file direcly, but I'm leaving it for later);
* Pytorch: Using above generated datas, I train the neural network models.

After training, .onnx files are generated, which are exported to Unity, where I can run the neural net inference using [Barracuda](https://medium.com/@a.abelhopereira/how-to-use-pytorch-models-in-unity-aa1e964d3374)

### XData.txt
This file consist of C blocks, N lines and M columns. C is the number of clips; Ni is the number of frames of clip i; M is the number of features (Described in this [paper](https://theorangeduck.com/media/uploads/other_stuff/Learned_Motion_Matching.pdf), section 3:  BASIC MOTION MATCHING).
Each block C is separated by a empty line. 

Example: C = 2, N[0] = 3, N[1] = 4, M = 24
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

## Roadmap
* Improve this readme
* Fix the the neural network losses (currently);
* ... (don't know if it's just the above missing);
* Extract character animations information using .bvh, without Unity.

## Note
I've already managed to make the system work, it's just not perfect yet. Warning that I'm still just an undergraduate, I'm not sure about anything hahahaha.
