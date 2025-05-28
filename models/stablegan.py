from torch.nn import Parameter
from torch import nn
import torch

import torch.nn.functional as F


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SSN(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1, alpha=0.95):
        super(SSN, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        self.alpha = alpha
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height, -1).data, v.data))

        sigma = u.dot(w.view(height, -1).mv(v))

        w_sn = w / (sigma + 1e-12)
        w_soft = self.alpha * w_sn + (1 - self.alpha) * w
        setattr(self.module, self.name, w_soft)

    def _made_params(self):
        try:
            getattr(self.module, self.name + "_u")
            getattr(self.module, self.name + "_v")
            getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


class Generator(nn.Module):

    def __init__(self, z_dim=100, image_size=128):
        super(Generator, self).__init__()

        self.layer1 = nn.Sequential(
            SSN(nn.ConvTranspose2d(z_dim, image_size * 8,
                                   kernel_size=4, stride=1)),
            nn.BatchNorm2d(image_size * 8),
            nn.ReLU(inplace=True))

        self.layer2 = nn.Sequential(
            SSN(nn.ConvTranspose2d(image_size * 8, image_size * 4,
                                   kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm2d(image_size * 4),
            nn.ReLU(inplace=True))

        self.layer3 = nn.Sequential(
            SSN(nn.ConvTranspose2d(image_size * 4, image_size * 2,
                                   kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm2d(image_size * 2),
            nn.ReLU(inplace=True))

        self.layer4 = nn.Sequential(
            SSN(nn.ConvTranspose2d(image_size * 2, image_size,
                                   kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm2d(image_size),
            nn.ReLU(inplace=True))

        self.last = nn.Sequential(
            nn.ConvTranspose2d(image_size, 1, kernel_size=4,
                               stride=2, padding=1),
            nn.Tanh())

    def forward(self, z):
        out = self.layer1(z)

        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.last(out)

        return out


class Discriminator(nn.Module):

    def __init__(self, image_size=64):
        super(Discriminator, self).__init__()

        self.layer1 = nn.Sequential(
            SSN(nn.Conv2d(1, image_size, kernel_size=4,
                          stride=2, padding=1)),
            nn.LeakyReLU(0.1, inplace=True))

        self.layer2 = nn.Sequential(
            SSN(nn.Conv2d(image_size, image_size * 2, kernel_size=4,
                          stride=2, padding=1)),
            nn.LeakyReLU(0.1, inplace=True))

        self.layer3 = nn.Sequential(
            SSN(nn.Conv2d(image_size * 2, image_size * 4, kernel_size=4,
                          stride=2, padding=1)),
            nn.LeakyReLU(0.1, inplace=True))

        self.layer4 = nn.Sequential(
            SSN(nn.Conv2d(image_size * 4, image_size * 8, kernel_size=4,
                          stride=2, padding=1)),
            nn.LeakyReLU(0.1, inplace=True))

        self.last = nn.Conv2d(image_size * 8, 1, kernel_size=4, stride=1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.last(out)

        return out


if __name__ == "__main__":
    G = Generator(z_dim=100, image_size=64)
    D = Discriminator(image_size=64)

    # Generate fake image
    input_z = torch.randn(1, 100)
    input_z = input_z.view(input_z.size(0), input_z.size(1), 1, 1)
    print(input_z.shape)

    fake_images = G(input_z)
    print(fake_images.shape)