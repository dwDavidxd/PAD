# from __future__ import print_function
import os
import argparse
import numpy as np
import random


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torch.fft as fft

from torchattacks import PGD, AutoAttack

from models.resnet import ResNet18
from models import net_style

from utils.data import data_dataset, data_dataset_with_ref
from utils.function import adaptive_instance_normalization


parser = argparse.ArgumentParser(description='PyTorch CIFAR-10')

parser.add_argument('--nat-img-train', type=str, help='natural training data', default='./data/train_images.npy')
parser.add_argument('--nat-label-train', type=str, help='natural training label', default='./data/train_labels.npy')
parser.add_argument('--nat-img-test', type=str, help='natural test data', default='./data/test_images.npy')
parser.add_argument('--nat-label-test', type=str, help='natural test label', default='./data/test_labels.npy')

parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=91, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--weight-decay', '--wd', default=2e-4,
                    type=float, metavar='W')
parser.add_argument('--lr', type=float, default=2e-1, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon', default=8/255,
                    help='perturbation')
parser.add_argument('--num-steps', default=10,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=2/255,
                    help='perturb step size')
parser.add_argument('--alpha', type=float, default=0.5, help='The weight that controls the degree of stylization. '
                                                             'Should be between 0 and 1')
parser.add_argument('--vgg', type=str, default='checkpoint/style/vgg_normalised.pth')
parser.add_argument('--decoder', type=str, default='checkpoint/style/decoder.pth')
# parser.add_argument('--decoder', type=str, default='checkpoint/style/decoder.pth')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model-dir', default='./checkpoint/ResNet_18/PAD',
                    help='directory of model for saving checkpoint')
parser.add_argument('--save-freq', '-s', default=1, type=int, metavar='N',
                    help='save frequency')

args = parser.parse_args()


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
  
  
def style_transfer_fun(vgg, decoder, data_inp, data_ref):
    data_ref = transforms.Resize((32, 32))(
        style_transfer(vgg, decoder, transforms.Resize((512, 512))(data_inp),
                       transforms.Resize((512, 512))(data_ref), args.alpha).detach())
    # data_ref = transforms.Resize((32, 32))(style_transfer(vgg, decoder, transforms.Resize((128, 128))(data_inp),
    #                             transforms.Resize((128, 128))(data_ref), args.alpha).detach()
    
    return data_ref


def amplitude_replacement_style(data_inp, data_ref, vgg, decoder):

    data_inp = data_inp.clone().detach()
    data_ref = data_ref.clone().detach()
    with torch.no_grad():
        data_ref = style_transfer_fun(vgg, decoder, data_inp.detach(), data_ref.detach())

    torch.cuda.empty_cache()

    amp_ref = torch.abs(fft.fftn(data_ref, dim=(1, 2, 3)))
    phs_inp = torch.angle(fft.fftn(data_inp, dim=(1, 2, 3)))
    data_inp = torch.abs(fft.ifftn(amp_ref*torch.exp((1j) * phs_inp), dim=(1, 2, 3)))

    return data_inp


def craft_adversarial_sample(model, x_natural, y, step_size=2/255, epsilon=8/255, perturb_steps=20):

    attack = PGD(model, eps=epsilon, alpha=step_size, steps=perturb_steps, random_start=True)
    x_adv = attack(x_natural, y)

    '''
    attack  = AutoAttack(model, norm='Linf', eps=8/255, version='stand')
    x_adv = attack(x_natural, y)
    '''
    return x_adv
  
  
def craft_white_box_adversarial_sample(model, x_natural, x_ref, y, vgg, decoder, step_size=2/255, epsilon=8/255, perturb_steps=20):

    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()

    for _ in range(perturb_steps):

        x_adv_pha = amplitude_replacement_style(x_adv, x_ref, vgg, decoder)

        x_adv.requires_grad_()
        x_adv_pha.requires_grad_()
        with torch.enable_grad():
            logits = model(x_adv_pha)
            loss_ce = Fn.cross_entropy(logits, y)

        grad = torch.autograd.grad(loss_ce, [x_adv_pha])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    return x_adv


def train(args, model, vgg, decoder, device, train_loader, optimizer, epoch):

    criterion_kl = nn.KLDivLoss(reduction='batchmean')

    for batch_idx, (data, data_ref, label) in enumerate(train_loader):
        data, data_ref, label = data.to(device), data_ref.to(device), label.to(device)

        batch_size = data.size()[0]
        index = torch.randperm(batch_size).cuda()
        data_ref = data_ref[index, :]

        # generate adversarial sample
        model.eval()

        attack = PGD(model, eps=args.epsilon, alpha=args.step_size, steps=args.num_steps, random_start=True)
        data_adv = attack(data, label)

        data_adv = amplitude_replacement_style(data_adv, data_ref, vgg, decoder)

        model.train()
        optimizer.zero_grad()

        logit_nat_ori = model(data)
        logit_adv_rec = model(data_adv)

        loss = F.cross_entropy(logit_adv_rec, label) + \
               6.0 * criterion_kl(F.log_softmax(logit_adv_rec, dim=1), F.softmax(logit_nat_ori, dim=1))

        loss.backward()
        optimizer.step()

        # print progress
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx+1) * len(data), len(train_loader.dataset),
                       100. * (batch_idx+1) / len(train_loader), loss.item()))


def eval_test(model, vgg, decoder, device, test_loader):
    model.eval()
    correct = 0
    correct_adv = 0

    for data, label in test_loader:
        data, label = data.to(device), label.to(device)

        batch_size = data.size()[0]
        index = torch.randperm(batch_size).cuda()
        data_ref = data[index, :]

        data_out = amplitude_replacement_style(data, data_ref, vgg, decoder)
        logits_out = model(data_out)
        pred = logits_out.max(1, keepdim=True)[1]
        correct += pred.eq(label.view_as(pred)).sum().item()

        data_adv = craft_adversarial_sample(model=model, x_natural=data, y=label,
                                             step_size=2/255, epsilon=8/255, perturb_steps=20) # general attack
        
       # data_adv = craft_white_box_adversarial_sample(model=model, x_natural=data, x_ref=data_ref, y=label, vgg=vgg, decoder=decoder, step_size=2/255, epsilon=8/255, perturb_steps=20) # white-box attack

        data_out = amplitude_replacement_style(data_adv, data_ref, vgg, decoder)
        logits_out = model(data_out)
        pred = logits_out.max(1, keepdim=True)[1]
        correct_adv += pred.eq(label.view_as(pred)).sum().item()

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
    trans_train_1 = transforms.Compose([
        transforms.RandomHorizontalFlip(),
    ])

    trans_train_2 = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
    ])

    trans_train_3 = transforms.Compose([
        transforms.ToTensor()
    ])

    trans_test = transforms.Compose([
        transforms.ToTensor()
    ])

    trainset = data_dataset_with_ref(img_path=args.nat_img_train, clean_label_path=args.nat_label_train,
                                     transform_1=trans_train_1, transform_2=trans_train_2, transform_3=trans_train_3)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, drop_last=False,
                                               shuffle=True, num_workers=4, pin_memory=True)
    testset = data_dataset(img_path=args.nat_img_test, clean_label_path=args.nat_label_test, transform=trans_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, drop_last=False, shuffle=False,
                                              num_workers=4, pin_memory=True)

    # init model
    model = ResNet18(10).to(device)
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

    vgg_style.eval()
    decoder_style.eval()
    cudnn.benchmark = True

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    for epoch in range(1, args.epochs + 1):
        adjust_learning_rate(optimizer, epoch)

        # adversarial training
        train(args, model, vgg_style, decoder_style, device, train_loader, optimizer, epoch)

        # evaluation on natural examples
        print('================================================================')
        torch.cuda.empty_cache()
        eval_test(model, vgg_style, decoder_style, device, test_loader)
        torch.cuda.empty_cache()

        # save checkpoint
        if epoch % args.save_freq == 0:
            torch.save(model.module.state_dict(),
                       os.path.join(model_dir, 'model.pth'))
            print('save the model')
        print('================================================================')


if __name__ == '__main__':
    main()
