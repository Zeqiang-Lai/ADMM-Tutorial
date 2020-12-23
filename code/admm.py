import torch.nn as nn
import torch
import torch.optim as optim


def total_variation(images):
    # c, h, w
    pixel_dif1 = images[:, 1:, :] - images[:, -1:, :]
    pixel_dif2 = images[:, :, 1:] - images[:, :, :-1]
    return torch.sum(pixel_dif1) + torch.sum(pixel_dif2)


class ObjectiveX(nn.Module):
    def __init__(self, lmbda=100.0):
        super().__init__()
        self.lmbda = lmbda
        self.rho = 1.0 / self.lmbda

    def forward(self, x, pv, pu, img):
        ubar = pu / self.rho
        xbar = pv - ubar
        return 0.5 * torch.sum((x - img) ** 2) + self.rho / 2 * torch.sum((x - xbar) ** 2)


class ObjectiveV(nn.Module):
    def __init__(self, lmbda=100.0):
        super().__init__()
        self.lmbda = lmbda
        self.rho = 1.0 / self.lmbda

    def forward(self, v, x, pu):
        ubar = pu / self.rho
        vbar = x + ubar
        return self.lmbda * total_variation(v) + self.rho / 2 * torch.sum((v - vbar) ** 2)


def admm(img, max_iter=20):
    f_obj_x = ObjectiveX()
    f_obj_v = ObjectiveV()

    x = torch.rand_like(img, requires_grad=True)
    v = torch.rand_like(img, requires_grad=True)
    u = torch.rand_like(img)

    x_optimizer = optim.SGD([x], lr=1e-3)
    v_optimizer = optim.SGD([v], lr=1e-3)
    for i in range(max_iter):
        x_optimizer.zero_grad()
        obj_x = f_obj_x(x, v, u, img)
        obj_x.backward()
        x_optimizer.step()

        v_optimizer.zero_grad()
        obj_v = f_obj_v(v, x, u)
        obj_v.backward()
        v_optimizer.step()

        u = u + x - v

        print('{}, {}'.format(obj_x.item(), obj_v.item()))

    return x, v, u


if __name__ == '__main__':
    from PIL import Image
    from torchvision.transforms import ToTensor

    img = Image.open('test.png')
    img = img.convert("RGB")
    img = ToTensor()(img)
    print(img.shape)

    admm(img)
