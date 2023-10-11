from __future__ import print_function
import os
import argparse
import numpy as np
import random


import torch
import torch.nn as nn
import torch.nn.functional as F
# import torchvision
import torch.optim as optim
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torch.fft as fft

from torchattacks import PGD #, AutoAttack, CW
from torchattacks.attack import Attack

from models.resnet import ResNet18
# from models.wideresnet import WideResNet
# from models.preactresnet import PreActResNet18
# from models.vggnet import VGGNet19
from models import net_style

from utils.standard_loss import standard_loss

from torch.autograd import Variable
from utils.data import data_dataset
from utils.function import adaptive_instance_normalization, coral


parser = argparse.ArgumentParser(description='PyTorch PAD')

parser.add_argument('--nat-img-train', type=str, help='natural training data', default='./data/train_images.npy')
parser.add_argument('--nat-label-train', type=str, help='natural training label', default='./data/train_labels.npy')
parser.add_argument('--nat-img-test', type=str, help='natural test data', default='./data/test_images.npy')
parser.add_argument('--nat-label-test', type=str, help='natural test label', default='./data/test_labels.npy')

parser.add_argument('--batch-size', type=int, default=512, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

parser.add_argument('--epsilon', default=8/255,
                    help='perturbation')
parser.add_argument('--num-steps', default=10,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=2/255,
                    help='perturb step size')

parser.add_argument('--alpha', type=float, default=0.85, help='The weight that controls the degree of stylization. '
                                                             'Should be between 0 and 1')
parser.add_argument('--vgg', type=str, default='checkpoint/style/vgg_normalised.pth')
parser.add_argument('--decoder', type=str, default='checkpoint/style/decoder.pth')

parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model-dir', default='./checkpoint/ResNet_18/PAD',
                    help='directory of model for saving checkpoint')
parser.add_argument('--save-freq', '-s', default=1, type=int, metavar='N',
                    help='save frequency')

args = parser.parse_args()


class PGD_adaptive(Attack):

    def __init__(self, model, eps=0.3,
                 alpha=2/255, steps=40, random_start=True):
        super().__init__("PGD", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self._supported_mode = ['default', 'targeted']

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self._targeted:
            target_labels = self._get_target_label(images, labels)

        loss = nn.CrossEntropyLoss()

        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for _ in range(self.steps):

            adv_images_pha = amplitude_replacement(adv_images, images)
            adv_images_pha.requires_grad = True
            outputs = self.model(adv_images_pha)

            # Calculate loss
            if self._targeted:
                cost = -loss(outputs, target_labels)
            else:
                cost = loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images_pha,
                                       retain_graph=False, create_graph=False)[0]

            adv_images = adv_images.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images


class PGD_adaptive_style(Attack):

    def __init__(self, model, eps=0.3,
                 alpha=2/255, steps=40, random_start=True):
        super().__init__("PGD", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self._supported_mode = ['default', 'targeted']

    def forward(self, images, refs, labels, vgg, decoder):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        refs = refs.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self._targeted:
            target_labels = self._get_target_label(images, labels)

        loss = nn.CrossEntropyLoss()

        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for _ in range(self.steps):

            adv_images_pha = amplitude_replacement_style(adv_images, refs, vgg, decoder)
            adv_images_pha.requires_grad = True
            outputs = self.model(adv_images_pha)

            # Calculate loss
            if self._targeted:
                cost = -loss(outputs, target_labels)
            else:
                cost = loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images_pha,
                                       retain_graph=False, create_graph=False)[0]

            adv_images = adv_images.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 100:
        lr = args.lr * 0.001
    elif epoch >= 90:
        lr = args.lr * 0.01
    elif epoch >= 75:
        lr = args.lr * 0.1

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def style_transfer(vgg, decoder, content, style, alpha=1.0,
                   interpolation_weights=None):
    assert (0.0 <= alpha <= 1.0)
    content_f = vgg(content)
    style_f = vgg(style)
    if interpolation_weights:
        _, C, H, W = content_f.size()
        feat = torch.FloatTensor(1, C, H, W).zero_().cuda()
        base_feat = adaptive_instance_normalization(content_f, style_f)
        for i, w in enumerate(interpolation_weights):
            feat = feat + w * base_feat[i:i + 1]
        content_f = content_f[0:1]
    else:
        feat = adaptive_instance_normalization(content_f, style_f)
    feat = feat * alpha + content_f * (1 - alpha)
    return decoder(feat)


def amplitude_replacement(data_inp, data_ref):

    data_inp = data_inp.clone().detach()
    data_ref = data_ref.clone().detach()

    amp_ref = torch.abs(fft.fftn(data_ref, dim=(1, 2, 3)))
    phs_inp = torch.angle(fft.fftn(data_inp, dim=(1, 2, 3)))
    data_inp = torch.abs(fft.ifftn(amp_ref*torch.exp((1j) * phs_inp), dim=(1, 2, 3)))

    return data_inp


def amplitude_replacement_style(data_inp, data_ref, vgg, decoder):

    data_inp = data_inp.clone().detach()
    data_ref = data_ref.clone().detach()
    with torch.no_grad():
        data_ref = style_transfer(vgg, decoder, transforms.Resize((300, 300))(data_inp.detach()),
                                     transforms.Resize((300, 300))(data_ref), args.alpha)

    data_ref = transforms.Resize((32, 32))(data_ref.detach())
    amp_ref = torch.abs(fft.fftn(data_ref, dim=(1, 2, 3)))
    phs_inp = torch.angle(fft.fftn(data_inp, dim=(1, 2, 3)))
    data_inp = torch.abs(fft.ifftn(amp_ref*torch.exp((1j) * phs_inp), dim=(1, 2, 3)))

    return data_inp


def craft_adversarial_example(model, x_natural, y, step_size=2/255, epsilon=8/255, perturb_steps=10):

    attack = PGD(model, eps=epsilon, alpha=step_size, steps=perturb_steps, random_start=True)
    x_adv = attack(x_natural, y)

    '''
    attack  = AutoAttack(model, norm='Linf', eps=8/255, version='stand')
    x_adv = attack(x_natural, y)
    '''

    '''
    adversary = DDNL2Attack(model, nb_iter=40, gamma=0.05, init_norm=1.0, quantize=True, levels=256, clip_min=0.0,
                            clip_max=1.0, targeted=False, loss_fn=None)
    x_adv = attack(x_natural, y)
    '''

    '''
    attack = CW(model, c=1, kappa=0, steps=10, lr=0.01)
    x_adv = attack(x_natural, y)
    '''


    '''
    adversary = LinfFWA(predict=model, loss_fn=nn.CrossEntropyLoss(reduction="sum"),
                      eps=8/255, kernel_size=4, lr=0.007, nb_iter=40, dual_max_iter=15, grad_tol=1e-4,
                    int_tol=1e-4, device="cuda", postprocess=False, verbose=True)
    x_adv = adversary(x_natural, y)
    # x_adv = adversary.perturb(x_natural, y)
    '''

    '''
    adversary = SpatialTransformAttack(model, 10, clip_min=0.0, clip_max=1.0, max_iterations=10, search_steps=5, targeted=False)
    x_adv = adversary(x_natural, y)
    '''

    '''
    adversary = TIDIM_Attack(model,
                       decay_factor=1, prob=0.5,
                       epsilon=8/255, steps=40, step_size=0.01,
                       image_resize=33,
                       random_start=False)
    '''

    '''
    adversary = TIDIM_Attack(eps=8/255, steps=40, step_size=0.007, momentum=0.1, prob=0.5, clip_min=0.0, clip_max=1.0,
                 device=torch.device('cuda'), low=32, high=32)
    x_adv = adversary.perturb(model, x_natural, y)
    '''

    return x_adv


def craft_adaptive_adversarial_example(model, x_natural, y, step_size=2/255, epsilon=8/255, perturb_steps=10):
    attack = PGD_adaptive(model, eps=epsilon, alpha=step_size, steps=perturb_steps, random_start=True)
    x_adv = attack(x_natural, y)
    return x_adv


def craft_adaptive_adversarial_example_style(model, x_natural, x_ref, y, vgg, decoder, step_size=2/255, epsilon=8/255,
                                       perturb_steps=10):
    attack = PGD_adaptive_style(model, eps=epsilon, alpha=step_size, steps=perturb_steps, random_start=True)
    x_adv = attack(x_natural, x_ref, y, vgg, decoder)
    return x_adv


def eval_test(model, vgg, decoder, device, test_loader):
    model.eval()
    correct = 0
    correct_adv = 0
    correct_adv_adp = 0
    correct_adv_adp_sty = 0
    #with torch.no_grad():
    for data, label in test_loader:
        data, label = data.to(device), label.to(device)

        batch_size = data.size()[0]
        index = torch.randperm(batch_size).cuda()
        data_ref = data[index, :]

        data_out = amplitude_replacement_style(data, data_ref, vgg, decoder)
        logits_out = model(data_out)
        pred = logits_out.max(1, keepdim=True)[1]
        correct += pred.eq(label.view_as(pred)).sum().item()

        data_adv = craft_adversarial_example(model=model, x_natural=data, y=label,
                                             step_size=2/255, epsilon=8/255, perturb_steps=20)

        data_out = amplitude_replacement_style(data_adv, data_ref, vgg, decoder)
        logits_out = model(data_out)
        pred = logits_out.max(1, keepdim=True)[1]
        correct_adv += pred.eq(label.view_as(pred)).sum().item()

        data_adv = craft_adaptive_adversarial_example_style(model=model, x_natural=data, x_ref=data_ref, y=label, vgg=vgg,
                                                      decoder=decoder, step_size=2/255, epsilon=8/255, perturb_steps=20)

        data_out = amplitude_replacement_style(data_adv, data_ref, vgg, decoder)
        logits_out = model(data_out)
        pred = logits_out.max(1, keepdim=True)[1]
        correct_adv_adp_sty += pred.eq(label.view_as(pred)).sum().item()

        data_adv = craft_adaptive_adversarial_example(model=model, x_natural=data, y=label, step_size=2/255, epsilon=8/255, perturb_steps=20)

        data_out = amplitude_replacement_style(data_adv, data_ref, vgg, decoder)
        logits_out = model(data_out)
        pred = logits_out.max(1, keepdim=True)[1]
        correct_adv_adp += pred.eq(label.view_as(pred)).sum().item()

    print('Test: Accuracy: {}/{} ({:.0f}%), Robust Accuracy: {}/{} ({:.0f}%)'.format(
        correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset), correct_adv,
        len(test_loader.dataset), 100. * correct_adv / len(test_loader.dataset)))


def main():
    # settings
    setup_seed(args.seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    model_dir = args.model_dir
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    # setup data loader
    trans_test = transforms.Compose([
        transforms.ToTensor()
    ])

    testset = data_dataset(img_path=args.nat_img_test, clean_label_path=args.nat_label_test, transform=trans_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, drop_last=False, shuffle=False,
                                              num_workers=4, pin_memory=True)


    model = ResNet18(10).to(device)
    # model = WideResNet(34, 10, 10).to(device)

    model.load_state_dict(torch.load(args.model_dir))
    model = torch.nn.DataParallel(model)

    decoder_style = net_style.decoder
    vgg_style = net_style.vgg

    vgg_style.load_state_dict(torch.load(args.vgg))
    vgg_style = nn.Sequential(*list(vgg_style.children())[:31])
    decoder_style.load_state_dict(torch.load(args.decoder))

    vgg_style = vgg_style.to(device)
    decoder_style = decoder_style.to(device)

    vgg_style = torch.nn.DataParallel(vgg_style)
    decoder_style = torch.nn.DataParallel(decoder_style)

    model.eval()
    vgg_style.eval()
    decoder_style.eval()
    cudnn.benchmark = True

    for epoch in range(1, args.epochs + 1):
        # adjust learning rate for SGD

        eval_test(model, vgg_style, decoder_style, device, test_loader)

        # save checkpoint
        if epoch % args.save_freq == 0:
            torch.save(model.module.state_dict(),
                       os.path.join(model_dir, 'best_model.pth'))
            print('save the model')

        print('================================================================')


if __name__ == '__main__':
    main()
