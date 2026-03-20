import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class PINN(nn.Module):

    def __init__(self):

        super(PINN,self).__init__()

        self.layers = nn.Sequential(

            nn.Linear(2,50),
            nn.Tanh(),

            nn.Linear(50,50),
            nn.Tanh(),

            nn.Linear(50,50),
            nn.Tanh(),

            nn.Linear(50,3)

        )

    def forward(self,x):

        return self.layers(x)


# Physics residual (OUTSIDE class)

def residual(model,x,y):

    xy = torch.cat([x,y],1)

    xy.requires_grad_(True)

    pred = model(xy)

    u = pred[:,0:1]
    v = pred[:,1:2]
    p = pred[:,2:3]

    grads = torch.autograd.grad(
        u,xy,
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]

    u_x = grads[:,0:1]
    u_y = grads[:,1:2]
    
    grads_v = torch.autograd.grad(
        v,xy,
        grad_outputs=torch.ones_like(v),
        create_graph=True
    )[0]

    v_x = grads_v[:,0:1]
    v_y = grads_v[:,1:2]

    continuity = u_x + v_y
    continuity_loss = torch.mean(continuity**2)

    u_xx = torch.autograd.grad(
        u_x,xy,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True
    )[0][:,0:1]

    u_yy = torch.autograd.grad(
        u_y,xy,
        grad_outputs=torch.ones_like(u_y),
        create_graph=True
    )[0][:,1:2]

    grads_p = torch.autograd.grad(
        p,xy,
        grad_outputs=torch.ones_like(p),
        create_graph=True
    )[0]

    p_x = grads_p[:,0:1]
    p_y = grads_p[:,1:2]

    Re = 100.0
    mom_x = u*u_x + v*u_y + p_x - (1/Re)*(u_xx + u_yy)
    mom_x_loss = torch.mean(mom_x**2)

    v_xx = torch.autograd.grad(
        v_x,xy,
        grad_outputs=torch.ones_like(v_x),
        create_graph=True
    )[0][:,0:1]

    v_yy = torch.autograd.grad(
        v_y,xy,
        grad_outputs=torch.ones_like(v_y),
        create_graph=True
    )[0][:,1:2]

    mom_y = u*v_x + v*v_y + p_y - (1/Re)*(v_xx + v_yy)
    mom_y_loss = torch.mean(mom_y**2)

    physics_loss = continuity_loss + mom_x_loss + mom_y_loss
    return physics_loss
# Interior collocation points (physics points)

N_f = 10000

x_f = torch.rand(N_f,1)*2 - 1
y_f = torch.rand(N_f,1)*2 - 1


# Boundary points

N_b = 2000

x_top = torch.rand(N_b,1)
y_top = torch.ones(N_b,1)

x_bottom = torch.rand(N_b,1)
y_bottom = torch.zeros(N_b,1)

y_left = torch.rand(N_b,1)
x_left = torch.zeros(N_b,1)

y_right = torch.rand(N_b,1)
x_right = torch.ones(N_b,1)


model = PINN()

optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)

for epoch in range(5000):

    optimizer.zero_grad()

    physics_loss = residual(model,x_f,y_f)

    # --- boundary recompute ---

    xy_top = torch.cat([x_top,y_top],1)
    pred_top = model(xy_top)

    u_top = pred_top[:,0:1]
    v_top = pred_top[:,1:2]

    loss_top = torch.mean((u_top-1)**2) + torch.mean(v_top**2)


    xy_bottom = torch.cat([x_bottom,y_bottom],1)
    pred_bottom = model(xy_bottom)

    u_bottom = pred_bottom[:,0:1]
    v_bottom = pred_bottom[:,1:2]

    loss_bottom = torch.mean(u_bottom**2) + torch.mean(v_bottom**2)


    xy_left = torch.cat([x_left,y_left],1)
    pred_left = model(xy_left)

    u_left = pred_left[:,0:1]
    v_left = pred_left[:,1:2]

    loss_left = torch.mean(u_left**2) + torch.mean(v_left**2)


    xy_right = torch.cat([x_right,y_right],1)
    pred_right = model(xy_right)

    u_right = pred_right[:,0:1]
    v_right = pred_right[:,1:2]

    loss_right = torch.mean(u_right**2) + torch.mean(v_right**2)


    boundary_loss = loss_top + loss_bottom + loss_left + loss_right


    loss = physics_loss + 20*boundary_loss

    loss.backward()

    optimizer.step()


    if epoch % 500 == 0:
        print(epoch, loss.item())


x = np.linspace(0,1,100)
y = np.linspace(0,1,100)

X,Y = np.meshgrid(x,y)

XY = torch.tensor(
    np.vstack([X.flatten(),Y.flatten()]).T,
    dtype=torch.float32
).detach()

pred = model(XY)

u = pred[:,0].detach().numpy()
v = pred[:,1].detach().numpy()

plt.quiver(X,Y,
           u.reshape(100,100),
           v.reshape(100,100))

plt.streamplot(X,Y,
               u.reshape(100,100),
               v.reshape(100,100),
               density=1.5)

plt.show()

p = pred[:,2].detach().numpy()

plt.contourf(X,Y,p.reshape(100,100),50)
plt.colorbar()
plt.title("Pressure field")

plt.show()

