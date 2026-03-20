# PINN Solution of 2D Lid Driven Cavity Flow

## Overview

This project solves the 2D incompressible Navier–Stokes equations using a Physics Informed Neural Network (PINN).

The neural network predicts:

• Velocity components (u,v)  
• Pressure field (p)

The governing equations enforced are:

Continuity:
du/dx + dv/dy = 0

Momentum equations:

u du/dx + v du/dy = − dp/dx + (1/Re)∇²u

u dv/dx + v dv/dy = − dp/dy + (1/Re)∇²v

Reynolds number used:
Re = 100

## Boundary conditions

Top lid:
u = 1
v = 0

Other walls:
u = 0
v = 0

## Neural Network

Architecture:

Input: (x,y)

Hidden layers:
3 layers × 50 neurons

Activation:
Tanh

Output:
(u,v,p)

## Training details

Physics collocation points:
10000

Boundary points:
2000 per wall

Optimizer:
Adam

Learning rate:
0.001

Epochs:
5000

## Results

The model produces:

• Velocity vector field  
• Streamlines  
• Pressure contours  

## How to run

Install dependencies:

pip install torch numpy matplotlib

Run:

python cavity_flow.py

## Author

Manya Johari  
Aerospace Engineering Student  
Interested in CFD, PINNs and computational physics