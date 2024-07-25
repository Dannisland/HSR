import torch
import torch.nn as nn
import torchvision


class VGGLoss(nn.Module):
    """
    Part of pre-trained VGG16. This is used in case we want perceptual loss instead of Mean Square Error loss.
    See for instance https://arxiv.org/abs/1603.08155
    """
    def __init__(self, block_no: int, layer_within_block: int, use_batch_norm_vgg: bool):
        super(VGGLoss, self).__init__()
        if use_batch_norm_vgg:
            vgg16 = torchvision.models.vgg16_bn(pretrained=True)
        else:
            vgg16 = torchvision.models.vgg16(pretrained=True)
        curr_block = 1
        curr_layer = 1
        layers = []
        for layer in vgg16.features.children():
            layers.append(layer)
            if curr_block == block_no and curr_layer == layer_within_block:
                break
            if isinstance(layer, nn.MaxPool2d):
                curr_block += 1
                curr_layer = 1
            else:
                curr_layer += 1

        self.vgg_loss = nn.Sequential(*layers)
        # print(self.vgg_loss)

    def forward(self, img):
        return self.vgg_loss(img)

if __name__ == "__main__":
    criterion_L2 = torch.nn.MSELoss()

    a = torch.rand(3, 3, 32, 32)
    b = torch.rand(3, 3, 32, 32)
    vgg_loss = VGGLoss(3, 1, False)

    vgg_on_cov = vgg_loss(a)
    vgg_on_cov2 = vgg_loss(b)
    g_vgg_loss = criterion_L2(vgg_on_cov, vgg_on_cov2)
    print(g_vgg_loss)