# Authors: Charly Lamothe <charlylmth_AT_gmail_DOT_com>; Paul Best <paulobest25_AT_@gmail_DOT_com>

# License: BSD (3-clause)

from torch import nn
import torchvision
import utils as u
from filterbank import STFT, MelFilter, Log1p
from PCEN_pytorch import PCENLayer

pDropout = .25

def get_resnet18(num_classes=11):
  resnet = torchvision.models.resnet18()
  resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
  resnet = nn.Sequential(*[k for k in resnet.children()]) # turn resnet into a sequential instead of named modules
  resnet[-2] = nn.Conv2d(512, num_classes, (4, 1)) # Size 4 works for 128 frequency bins
  resnet[-1] = nn.AdaptiveMaxPool2d(output_size=(1,1))
  return resnet

def get_resnet50(num_classes=11):
  resnet = torchvision.models.resnet50()
  resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
  resnet.fc = nn.Linear(2048, num_classes)
  return resnet

get = {
 'frontend' : lambda sr, nfft, sampleDur, n_mel : nn.Sequential(
  STFT(nfft, int((sampleDur*sr - nfft)/128)),
  MelFilter(sr, nfft, n_mel, 1000, sr//2),
  Log1p(7, trainable=False),
  nn.InstanceNorm2d(1),
  u.Croper2D(n_mel, 128)
 ),
'frontend_pcen_96': nn.Sequential(
  STFT(1024, 368),
  MelFilter(96000, 1024, 128, 1000, 48000),
  PCENLayer(128, s=0.01, r=.5, trainable=False)
 ),
'frontend_logMel_96': nn.Sequential(
  STFT(1024, 368),
  MelFilter(96000, 1024, 128, 1000, 48000),
  Log1p(7)
 ),
 'frontend_pcen_44_1': nn.Sequential(
  STFT(1024, 163),
  MelFilter(44100, 1024, 128, 1000, 22050),
  PCENLayer(128, s=0.01, r=.5, trainable=False)
 ),
'frontend_logMel_44_1': nn.Sequential(
  STFT(1024, 163),
  MelFilter(44100, 1024, 128, 1000, 22050),
  Log1p(7)
 ),
'sparrow_encoder' : lambda nfeat, shape : nn.Sequential(
  nn.Conv2d(1, 32, 3, stride=2, bias=False, padding=(1)),
  nn.BatchNorm2d(32),
  nn.LeakyReLU(0.01),
  nn.Conv2d(32, 64, 3, stride=2, bias=False, padding=1),
  nn.BatchNorm2d(64),
  nn.ReLU(True),
  nn.Conv2d(64, 128, 3, stride=2, bias=False, padding=1),
  nn.BatchNorm2d(128),
  nn.ReLU(True),
  nn.Conv2d(128, 256, 3, stride=2, bias=False, padding=1),
  nn.BatchNorm2d(256),
  nn.ReLU(True),
  nn.Conv2d(256, nfeat, (3, 5), stride=2, padding=(1, 2)),
  u.Reshape(nfeat * shape[0] * shape[1])
 ),
'sparrow_decoder' : lambda nfeat, shape : nn.Sequential(
  u.Reshape(nfeat//(shape[0]*shape[1]), *shape),
  nn.ReLU(True),

  nn.Upsample(scale_factor=2),
  nn.Conv2d(nfeat//(shape[0]*shape[1]), 256, (3, 3), bias=False, padding=1),
  nn.BatchNorm2d(256),
  nn.ReLU(True),
  nn.Conv2d(256, 256, (3, 3), bias=False, padding=1),
  nn.BatchNorm2d(256),
  nn.ReLU(True),

  nn.Upsample(scale_factor=2),
  nn.Conv2d(256, 128, (3, 3), bias=False, padding=1),
  nn.BatchNorm2d(128),
  nn.ReLU(True),
  nn.Conv2d(128, 128, (3, 3), bias=False, padding=1),
  nn.BatchNorm2d(128),
  nn.ReLU(True),

  nn.Upsample(scale_factor=2),
  nn.Conv2d(128, 64, (3, 3), bias=False, padding=1),
  nn.BatchNorm2d(64),
  nn.ReLU(True),
  nn.Conv2d(64, 64, (3, 3), bias=False, padding=1),
  nn.BatchNorm2d(64),
  nn.ReLU(True),

  nn.Upsample(scale_factor=2),
  nn.Conv2d(64, 32, (3, 3), bias=False, padding=1),
  nn.BatchNorm2d(32),
  nn.ReLU(True),
  nn.Conv2d(32, 32, (3, 3), bias=False, padding=1),
  nn.BatchNorm2d(32),
  nn.ReLU(True),

  nn.Upsample(scale_factor=2),
  nn.Conv2d(32, 32, (3, 3), bias=False, padding=1),
  nn.BatchNorm2d(32),
  nn.ReLU(True),
  nn.Conv2d(32, 1, (3, 3), bias=False, padding=1),
  nn.ReLU(True)),

'sparrow_encoder_old' : lambda nfeat : nn.Sequential(
  nn.Conv2d(1, 32, 3, stride=2, bias=False, padding=(1)),
  nn.BatchNorm2d(32),
  nn.LeakyReLU(0.01),
  nn.Conv2d(32, 64, 3, stride=2, bias=False, padding=1),
  nn.BatchNorm2d(64),
  nn.MaxPool2d((1, 2)),
  nn.ReLU(True),
  nn.Conv2d(64, 128, 3, stride=2, bias=False, padding=1),
  nn.BatchNorm2d(128),
  nn.ReLU(True),
  nn.Conv2d(128, 256, 3, stride=2, bias=False, padding=1),
  nn.BatchNorm2d(256),
  nn.ReLU(True),
  nn.Conv2d(256, nfeat, (3, 5), stride=2, padding=(1, 2)),
  nn.AdaptiveMaxPool2d((1,1)),
  u.Reshape(nfeat)),

'sparrow_decoder_old' : lambda nfeat, shape : nn.Sequential(
  nn.Linear(nfeat, nfeat*shape[0]*shape[1]),
  u.Reshape(nfeat, *shape),
  nn.ReLU(True),

  nn.Upsample(scale_factor=2),
  nn.Conv2d(nfeat, 256, (3, 3), bias=False, padding=1),
  nn.BatchNorm2d(256),
  nn.ReLU(True),
  nn.Conv2d(256, 256, (3, 3), bias=False, padding=1),
  nn.BatchNorm2d(256),
  nn.ReLU(True),

  nn.Upsample(scale_factor=2),
  nn.Conv2d(256, 128, (3, 3), bias=False, padding=1),
  nn.BatchNorm2d(128),
  nn.ReLU(True),
  nn.Conv2d(128, 128, (3, 3), bias=False, padding=1),
  nn.BatchNorm2d(128),
  nn.ReLU(True),

  nn.Upsample(scale_factor=2),
  nn.Conv2d(128, 64, (3, 3), bias=False, padding=1),
  nn.BatchNorm2d(64),
  nn.ReLU(True),
  nn.Conv2d(64, 64, (3, 3), bias=False, padding=1),
  nn.BatchNorm2d(64),
  nn.ReLU(True),

  nn.Upsample(scale_factor=(1,2)),
  nn.Conv2d(64, 32, (3, 3), bias=False, padding=1),
  nn.BatchNorm2d(32),
  nn.ReLU(True),
  nn.Conv2d(32, 32, (3, 3), bias=False, padding=1),
  nn.BatchNorm2d(32),
  nn.ReLU(True),

  nn.Upsample(scale_factor=2),
  nn.Conv2d(32, 32, (3, 3), bias=False, padding=1),
  nn.BatchNorm2d(32),
  nn.ReLU(True),
  nn.Conv2d(32, 32, (3, 3), bias=False, padding=1),
  nn.BatchNorm2d(32),
  nn.ReLU(True),

  nn.Upsample(scale_factor=2),
  nn.Conv2d(32, 32, (3, 3), bias=False, padding=1),
  nn.BatchNorm2d(32),
  nn.ReLU(True),
  nn.Conv2d(32, 1, (3, 3), bias=False, padding=1),
  nn.ReLU(True)),

'sparrow_encoder_nomaxpool' : lambda nfeat : nn.Sequential(
  nn.Conv2d(1, 32, 3, stride=2, bias=False, padding=(1)),
  nn.BatchNorm2d(32),
  nn.LeakyReLU(0.01),
  nn.Conv2d(32, 64, 3, stride=2, bias=False, padding=1),
  nn.BatchNorm2d(64),
  nn.MaxPool2d((1, 5), stride=(1, 2), padding=(0, 2)),
  nn.ReLU(True),
  nn.Conv2d(64, 128, 3, stride=2, bias=False, padding=1),
  nn.BatchNorm2d(128),
  nn.ReLU(True),
  nn.Conv2d(128, 256, 3, stride=2, bias=False, padding=1),
  nn.BatchNorm2d(256),
  nn.ReLU(True),
  nn.Conv2d(256, nfeat//16, (3, 5), stride=2, padding=(1, 2)),
  u.Reshape(nfeat)),

'sparrow_decoder_nomaxpool' : lambda nfeat, shape : nn.Sequential(
  u.Reshape(nfeat//16, *shape),
  nn.ReLU(True),

  nn.Upsample(scale_factor=2),
  nn.Conv2d(nfeat//16, 256, (3, 3), bias=False, padding=1),
  nn.BatchNorm2d(256),
  nn.ReLU(True),
  nn.Conv2d(256, 256, (3, 3), bias=False, padding=1),
  nn.BatchNorm2d(256),
  nn.ReLU(True),

  nn.Upsample(scale_factor=2),
  nn.Conv2d(256, 128, (3, 3), bias=False, padding=1),
  nn.BatchNorm2d(128),
  nn.ReLU(True),
  nn.Conv2d(128, 128, (3, 3), bias=False, padding=1),
  nn.BatchNorm2d(128),
  nn.ReLU(True),

  nn.Upsample(scale_factor=2),
  nn.Conv2d(128, 64, (3, 3), bias=False, padding=1),
  nn.BatchNorm2d(64),
  nn.ReLU(True),
  nn.Conv2d(64, 64, (3, 3), bias=False, padding=1),
  nn.BatchNorm2d(64),
  nn.ReLU(True),

  nn.Upsample(scale_factor=(1,2)),
  nn.Conv2d(64, 32, (3, 3), bias=False, padding=1),
  nn.BatchNorm2d(32),
  nn.ReLU(True),
  nn.Conv2d(32, 32, (3, 3), bias=False, padding=1),
  nn.BatchNorm2d(32),
  nn.ReLU(True),

  nn.Upsample(scale_factor=2),
  nn.Conv2d(32, 32, (3, 3), bias=False, padding=1),
  nn.BatchNorm2d(32),
  nn.ReLU(True),
  nn.Conv2d(32, 32, (3, 3), bias=False, padding=1),
  nn.BatchNorm2d(32),
  nn.ReLU(True),

  nn.Upsample(scale_factor=2),
  nn.Conv2d(32, 32, (3, 3), bias=False, padding=1),
  nn.BatchNorm2d(32),
  nn.ReLU(True),
  nn.Conv2d(32, 1, (3, 3), bias=False, padding=1),
  nn.ReLU(True))
}
