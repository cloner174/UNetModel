import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


class MultitaskAttentionUNet(nn.Module):
    
    def __init__(self, input_channels=1, num_classes=1, bbox_size=4):
        super(MultitaskAttentionUNet, self).__init__()
        
        self.enc1 = self.conv_block(input_channels, 64)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.enc2 = self.conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.enc3 = self.conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.enc4 = self.conv_block(256, 512)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.bottleneck = self.conv_block(512, 1024)
        
        self.up5 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.att5 = AttentionGate(F_g=512, F_l=512, F_int=256)
        self.dec5 = self.conv_block(1024, 512)
        
        self.up6 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.att6 = AttentionGate(F_g=256, F_l=256, F_int=128)
        self.dec6 = self.conv_block(512, 256)
        
        self.up7 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.att7 = AttentionGate(F_g=128, F_l=128, F_int=64)
        self.dec7 = self.conv_block(256, 128)
        
        self.up8 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.att8 = AttentionGate(F_g=64, F_l=64, F_int=32)
        self.dec8 = self.conv_block(128, 64)
        
        self.seg_out = nn.Conv2d(64, num_classes, kernel_size=1)
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1024 * 1 * 1, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )
        
        self.localization = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1024 * 1 * 1, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, bbox_size)
        )
        
    def conv_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        return block
    
    def forward(self, x):
        enc1 = self.enc1(x)       # [B, 64, H, W]
        enc2 = self.enc2(self.pool1(enc1))  # [B, 128, H/2, W/2]
        enc3 = self.enc3(self.pool2(enc2))  # [B, 256, H/4, W/4]
        enc4 = self.enc4(self.pool3(enc3))  # [B, 512, H/8, W/8]
        
        bottleneck = self.bottleneck(self.pool4(enc4))  # [B, 1024, H/16, W/16]
        
        up5 = self.up5(bottleneck)  # [B, 512, H/8, W/8]
        att5 = self.att5(g=up5, x=enc4)  # [B, 512, H/8, W/8]
        merge5 = torch.cat([up5, att5], dim=1)  # [B, 1024, H/8, W/8]
        dec5 = self.dec5(merge5)  # [B, 512, H/8, W/8]
        
        up6 = self.up6(dec5)  # [B, 256, H/4, W/4]
        att6 = self.att6(g=up6, x=enc3)  # [B, 256, H/4, W/4]
        merge6 = torch.cat([up6, att6], dim=1)  # [B, 512, H/4, W/4]
        dec6 = self.dec6(merge6)  # [B, 256, H/4, W/4]
        
        up7 = self.up7(dec6)  # [B, 128, H/2, W/2]
        att7 = self.att7(g=up7, x=enc2)  # [B, 128, H/2, W/2]
        merge7 = torch.cat([up7, att7], dim=1)  # [B, 256, H/2, W/2]
        dec7 = self.dec7(merge7)  # [B, 128, H/2, W/2]
        
        up8 = self.up8(dec7)  # [B, 64, H, W]
        att8 = self.att8(g=up8, x=enc1)  # [B, 64, H, W]
        merge8 = torch.cat([up8, att8], dim=1)  # [B, 128, H, W]
        dec8 = self.dec8(merge8)  # [B, 64, H, W]
        
        seg_out = self.seg_out(dec8)  # [B, num_classes, H, W]
        
        cls_out = self.classifier(bottleneck)  # [B, 1]
        loc_out = self.localization(bottleneck)  # [B, bbox_size]
        
        return seg_out, cls_out, loc_out
    

class AttentionGate(nn.Module):
    
    def __init__(self, F_g, F_l, F_int):
        """
        F_g: Number of channels in gating signal
        F_l: Number of channels in the skip connection
        F_int: Number of intermediate channels
        """
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g, x):
        """
        g: gating signal from the deeper layer
        x: feature map from the skip connection
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi
    


class MultitaskAttentionUNet_Pretrained(nn.Module):
    
    def __init__(self, input_channels=1, num_classes=2, bbox_size=4 , weight_path = None):
        super(MultitaskAttentionUNet_Pretrained, self).__init__()
        # Encoder: Pretrained ResNet34
        if weight_path is None:
            weight_path = './weights/resnet34.pth'
        
        try:
            self.encoder = models.resnet34(weights=None)
            state_dict = torch.load(weight_path, weights_only=True)
            self.encoder.load_state_dict(state_dict)
        except:
            from torchvision.models import ResNet34_Weights
            self.encoder = models.resnet34(weights= ResNet34_Weights.IMAGENET1K_V1)
        
        if input_channels != 3:
            self.encoder.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        self.enc0 = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )  # [B, 16, H, W]
        
        self.enc1 = nn.Sequential(self.encoder.conv1, self.encoder.bn1, self.encoder.relu)  # [B,64,H/2,W/2]
        self.pool1 = self.encoder.maxpool  # [B,64,H/4,W/4]
        self.enc2 = self.encoder.layer1  # [B,64,H/4,W/4]
        self.enc3 = self.encoder.layer2  # [B,128,H/8,W/8]
        self.enc4 = self.encoder.layer3  # [B,256,H/16,W/16]
        self.enc5 = self.encoder.layer4  # [B,512,H/32,W/32]
        
        self.up5 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)  # [B,256,H/16,W/16]
        self.att5 = AttentionGate(F_g=256, F_l=256, F_int=128)
        self.dec5 = self.conv_block(512, 256)  # [B,256,H/16,W/16]
        
        self.up6 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)  # [B,128,H/8,W/8]
        self.att6 = AttentionGate(F_g=128, F_l=128, F_int=64)
        self.dec6 = self.conv_block(256, 128)  # [B,128,H/8,W/8]
        
        self.up7 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)  # [B,64,H/4,W/4]
        self.att7 = AttentionGate(F_g=64, F_l=64, F_int=32)
        self.dec7 = self.conv_block(128, 64)  # [B,64,H/4,W/4]
        
        self.up8 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)  # [B,32,H/2,W/2]
        self.att8 = AttentionGate(F_g=32, F_l=64, F_int=16)  # F_l=64 from enc1
        self.dec8 = self.conv_block(96, 32)  # [B,32,H/2,W/2] (32 from up8 + 64 from att8)
        
        # decoder layers for full resolution
        self.up9 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)  # [B,16,H,W]
        self.att9 = AttentionGate(F_g=16, F_l=16, F_int=8)  # F_l=16 from enc0
        self.dec9 = self.conv_block(32, 16)  # [B,16,H,W] (16 from up9 + 16 from att9)
        
        self.seg_out = nn.Conv2d(16, num_classes, kernel_size=1)  # [B,num_classes,H,W]
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )
        
        self.localization = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, bbox_size)
        )
    
    def conv_block(self, in_channels, out_channels):
        """
        Convolutional block consisting of two Conv-BatchNorm-ReLU layers.
        """
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        return block
    
    def forward(self, x):
        # encoder
        enc0 = self.enc0(x)  # [B,16,H,W]
        enc1 = self.enc1(x)  # [B,64,H/2,W/2]
        enc2 = self.enc2(self.pool1(enc1))  # [B,64,H/4,W/4]
        enc3 = self.enc3(enc2)    # [B,128,H/8,W/8]
        enc4 = self.enc4(enc3)    # [B,256,H/16,W/16]
        enc5 = self.enc5(enc4)    # [B,512,H/32,W/32]
        
        # decoder1
        up5 = self.up5(enc5)      # [B,256,H/16,W/16]
        att5 = self.att5(g=up5, x=enc4)  # [B,256,H/16,W/16]
        merge5 = torch.cat([up5, att5], dim=1)  # [B,512,H/16,W/16]
        dec5 = self.dec5(merge5)  # [B,256,H/16,W/16]
        
        # decoder2
        up6 = self.up6(dec5)      # [B,128,H/8,W/8]
        att6 = self.att6(g=up6, x=enc3)  # [B,128,H/8,W/8]
        merge6 = torch.cat([up6, att6], dim=1)  # [B,256,H/8,W/8]
        dec6 = self.dec6(merge6)  # [B,128,H/8,W/8]
        
        # decoder 3
        up7 = self.up7(dec6)      # [B,64,H/4,W/4]
        att7 = self.att7(g=up7, x=enc2)  # [B,64,H/4,W/4]
        merge7 = torch.cat([up7, att7], dim=1)  # [B,128,H/4,W/4]
        dec7 = self.dec7(merge7)  # [B,64,H/4,W/4]
        
        # decoder4
        up8 = self.up8(dec7)      # [B,32,H/2,W/2]
        att8 = self.att8(g=up8, x=enc1)  # [B,64,H/2,W/2]
        merge8 = torch.cat([up8, att8], dim=1)  # [B,96,H/2,W/2]
        dec8 = self.dec8(merge8)  # [B,32,H/2,W/2]
        
        up9 = self.up9(dec8)      # [B,16,H,W]
        att9 = self.att9(g=up9, x=enc0)  # [B,16,H,W]
        merge9 = torch.cat([up9, att9], dim=1)  # [B,32,H,W]
        dec9 = self.dec9(merge9)  # [B,16,H,W]
        
        seg_out = self.seg_out(dec9)  # [B,num_classes,H,W]
        
        cls_out = self.classifier(enc5)  # [B, 1]
        loc_out = self.localization(enc5)  # [B, bbox_size]
        
        return seg_out, cls_out, loc_out
    


class MultitaskNet(nn.Module):
    
    def __init__(self, input_channels=1, num_classes=1, bbox_size=4):
        super(MultitaskNet, self).__init__()
        
        self.enc1 = self.conv_block(input_channels, 64)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.enc2 = self.conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.enc3 = self.conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.enc4 = self.conv_block(256, 512)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.bottleneck = self.conv_block(512, 1024)
        
        self.up5 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.att5 = Attention(F_g=512, F_l=512, F_int=256)
        self.dec5 = self.conv_block(1024, 512)
        
        self.up6 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.att6 = Attention(F_g=256, F_l=256, F_int=128)
        self.dec6 = self.conv_block(512, 256)
        
        self.up7 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.att7 = Attention(F_g=128, F_l=128, F_int=64)
        self.dec7 = self.conv_block(256, 128)
        
        self.up8 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.att8 = Attention(F_g=64, F_l=64, F_int=32)
        self.dec8 = self.conv_block(128, 64)
        
        self.seg_out = nn.Conv2d(64, num_classes, kernel_size=1)
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )
        
        self.localization = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, bbox_size)
        )
    
    def conv_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        return block
    
    def forward(self, x, return_features=False):
        enc1 = self.enc1(x)       # [B, 64, H, W]
        enc2 = self.enc2(self.pool1(enc1))  # [B, 128, H/2, W/2]
        enc3 = self.enc3(self.pool2(enc2))  # [B, 256, H/4, W/4]
        enc4 = self.enc4(self.pool3(enc3))  # [B, 512, H/8, W/8]
        
        bottleneck = self.bottleneck(self.pool4(enc4))  # [B, 1024, H/16, W/16]
        
        if return_features:
            features = bottleneck.clone()
        
        up5 = self.up5(bottleneck)  # [B, 512, H/8, W/8]
        att5 = self.att5(g=up5, x=enc4)  # [B, 512, H/8, W/8]
        merge5 = torch.cat([up5, att5], dim=1)  # [B, 1024, H/8, W/8]
        dec5 = self.dec5(merge5)  # [B, 512, H/8, W/8]
        
        up6 = self.up6(dec5)  # [B, 256, H/4, W/4]
        att6 = self.att6(g=up6, x=enc3)  # [B, 256, H/4, W/4]
        merge6 = torch.cat([up6, att6], dim=1)  # [B, 512, H/4, W/4]
        dec6 = self.dec6(merge6)  # [B, 256, H/4, W/4]
        
        up7 = self.up7(dec6)  # [B, 128, H/2, W/2]
        att7 = self.att7(g=up7, x=enc2)  # [B, 128, H/2, W/2]
        merge7 = torch.cat([up7, att7], dim=1)  # [B, 256, H/2, W/2]
        dec7 = self.dec7(merge7)  # [B, 128, H/2, W/2]
        
        up8 = self.up8(dec7)  # [B, 64, H, W]
        att8 = self.att8(g=up8, x=enc1)  # [B, 64, H, W]
        merge8 = torch.cat([up8, att8], dim=1)  # [B, 128, H, W]
        dec8 = self.dec8(merge8)  # [B, 64, H, W]
        
        seg_out = self.seg_out(dec8)  # [B, num_classes, H, W]
        
        cls_out = self.classifier(bottleneck)  # [B, 1]
        loc_out = self.localization(bottleneck)  # [B, bbox_size]
        
        if return_features:
            return seg_out, cls_out, loc_out, features
        else:
            return seg_out, cls_out, loc_out



class Attention(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi
    
#cloner174