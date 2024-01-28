""" Full assembly of the parts to form the complete network """

from .loam_parts import *
import torch.utils.checkpoint as Chkpt

class LOAM(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(LOAM, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))

        self.sa = (sa_layer(1024))

        # 256 = 4 (for category) * 2 (for column/ row side) * 32 (for entry per side)
        # 128 = 4 (for category) * 2 (for column/ row side) * 16 (for entry per side)
        '''
        self.fc1 = (Dense(128, 256))
        self.fc2 = (Dense(256, 512))
        self.fc3 = (Dense(512, 128))
        '''
        self.fc1 = (Dense(256, 512))
        self.fc2 = (Dense(512, 256))
        self.fc3 = (Dense(256, 128))

        # 4096 = 4 (for category) * 32 (for column per side) * 32 (for row per side)
        # 1024 = 4 (for category) * 16 (for column per side) * 16 (for row per side)
        '''
        self.fm1 = (OutConv(4, 64))
        self.fm2 = (OutConv(64, 256))
        self.fm3 = (OutConv(256, 512))
        '''
        self.fm1 = (UpConv(4, 64))
        self.fm2 = (OutConv(64, 256))
        self.fm3 = (OutConv(256, 512))

        self.mf = (mf_layer(1024))

        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

        self.up1y = (Up(1024, 512 // factor, bilinear))
        self.up2y = (Up(512, 256 // factor, bilinear))
        self.up3y = (Up(256, 128 // factor, bilinear))
        self.up4y = (Up(128, 64, bilinear))
        self.outcy = (OutConv(64, n_classes))

        

    def forward(self, x, zc, zm):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x6 = self.sa(x5)

        zc1 = self.fc1(zc)
        zc2 = self.fc2(zc1)
        zc3 = self.fc3(zc2)

        zm1 = self.fm1(zm)
        zm2 = self.fm2(zm1)
        zm3 = self.fm3(zm2)

        z2 = self.mf(x6, zc3, zm3)

        x = self.up1(z2, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        y = self.up1y(z2, x4)
        y = self.up2y(y, x3)
        y = self.up3y(y, x2)
        y = self.up4y(y, x1)

        logits = self.outc(x)
        logitsy = self.outcy(y)
        return logits, logitsy

    def use_checkpointing(self):
        self.inc = Chkpt(self.inc)
        self.down1 = Chkpt(self.down1)
        self.down2 = Chkpt(self.down2)
        self.down3 = Chkpt(self.down3)
        self.down4 = Chkpt(self.down4)
        self.sa = Chkpt(self.sa)
        self.fc1 = Chkpt(self.fc1)
        self.fc2 = Chkpt(self.fc2)
        self.fc3 = Chkpt(self.fc3)
        self.fm1 = Chkpt(self.fm1)
        self.fm2 = Chkpt(self.fm2)
        self.fm3 = Chkpt(self.fm3)
        self.mf = Chkpt(self.mf)
        self.up1 = Chkpt(self.up1)
        self.up2 = Chkpt(self.up2)
        self.up3 = Chkpt(self.up3)
        self.up4 = Chkpt(self.up4)
        self.outc = Chkpt(self.outc)
        self.up1y = Chkpt(self.up1y)
        self.up2y = Chkpt(self.up2y)
        self.up3y = Chkpt(self.up3y)
        self.up4y = Chkpt(self.up4y)
        self.outcy = Chkpt(self.outcy)
        '''
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.sa = torch.utils.checkpoint(self.sa)
        self.fc1 = torch.utils.checkpoint(self.fc1)
        self.fc2 = torch.utils.checkpoint(self.fc2)
        self.fc3 = torch.utils.checkpoint(self.fc3)
        self.fm1 = torch.utils.checkpoint(self.fm1)
        self.fm2 = torch.utils.checkpoint(self.fm2)
        self.fm3 = torch.utils.checkpoint(self.fm3)
        self.mf = torch.utils.checkpoint(self.mf)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)
        self.up1y = torch.utils.checkpoint(self.up1y)
        self.up2y = torch.utils.checkpoint(self.up2y)
        self.up3y = torch.utils.checkpoint(self.up3y)
        self.up4y = torch.utils.checkpoint(self.up4y)
        self.outcy = torch.utils.checkpoint(self.outcy)
        '''