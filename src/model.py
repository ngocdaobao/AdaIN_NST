import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import vgg19

# Encoder
class VGG19Encoder(nn.Module):
    def __init__(self):
        super(VGG19Encoder, self).__init__()
        vgg = vgg19(pretrained=True).features
        # Extract layers up to relu4_1
        self.enc_layers = nn.Sequential(*list(vgg.children())[:21])
        for params in self.enc_layers.parameters():
            params.requires_grad = False
        self._adain_output = None
    
    def forward(self, x):
        h_final = self.enc_layers(x)
        
        #Extract features at relu1_1, relu2_1, relu3_1, relu4_1 if needed
        h1 = self.enc_layers[:2](x)   # relu1_1
        h2 = self.enc_layers[2:7](h1) # relu2_1
        h3 = self.enc_layers[7:12](h2) # relu3_1
        h4 = self.enc_layers[12:21](h3) # relu4_1
        h_middle = [h1, h2, h3, h4]
        return h_final, h_middle

    @property
    def adain_output(self):
        """Get the last computed AdaIN output"""
        return self._adain_output
    
    def compute_adain(self, content_feat, style_feat):
        """Compute AdaIN and store the result"""
        # compute mean and std per channel
        content_batch_size, content_c = content_feat.size()[:2]
        content_mean  = content_feat.view(content_batch_size, content_c, -1).mean(dim=2).view(content_batch_size, content_c, 1, 1)
        content_std   = content_feat.view(content_batch_size, content_c, -1).std(dim=2).view(content_batch_size, content_c, 1, 1) + 1e-5

        style_batch_size, style_c = style_feat.size()[:2]
        style_mean = style_feat.view(style_batch_size, style_c, -1).mean(dim=2).view(style_batch_size, style_c, 1, 1)
        style_std  = style_feat.view(style_batch_size, style_c, -1).std(dim=2).view(style_batch_size, style_c, 1, 1) + 1e-5

        self._adain_output = style_std*((content_feat - content_mean)/content_std) + style_mean
        return self._adain_output

# Decoder
class Decoder(nn.Module):
    # Decoder as a mirror of VGG19 up to relu4_1, replacing maxpool with nearest upsampling
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder_layers = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),

            nn.Conv2d(128,128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),

            nn.Conv2d(64,64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,3, kernel_size=3, padding=1),)
    
    def forward(self, x):
        h=self.decoder_layers(x)
        return h
    
# Style Transfer Model
class StyleTransferModel(nn.Module):
    def __init__(self, encoder_model = VGG19Encoder(), decoder_model = Decoder()):
        super(StyleTransferModel, self).__init__()
        self.encoder = encoder_model
        self.decoder = decoder_model
    
    def extract_features(self, content, style):
        content_features, _ = self.encoder(content)
        style_features, style_middle = self.encoder(style)
        adain = self.encoder.compute_adain(content_features, style_features)
        return content_features, style_features, style_middle, adain
    
    def generate(self, adain_output):
        generated_image = self.decoder(adain_output)
        generated_features, generated_middle = self.encoder(generated_image)
        return generated_image, generated_features, generated_middle
    
    def content_loss(self, content_features, generated_features):
        return F.mse_loss(generated_features, content_features)
    
    def style_loss(self, style_middle, generated_middle):
        loss_style = 0.0
        for sf, gf in zip(style_middle, generated_middle):
            sf_batch, sf_channels = sf.size()[:2]
            sf_mean = sf.view(sf_batch, sf_channels, -1).mean(dim=2).view(sf_batch, sf_channels, 1, 1)
            sf_std = sf.view(sf_batch, sf_channels, -1).std(dim=2).view(sf_batch, sf_channels, 1, 1)

            gf_batch, gf_channels = gf.size()[:2]
            gf_mean = gf.view(gf_batch, gf_channels, -1).mean(dim=2).view(gf_batch, gf_channels, 1, 1)
            gf_std = gf.view(gf_batch, gf_channels, -1).std(dim=2).view(gf_batch, gf_channels, 1, 1)

            loss_style += F.mse_loss(gf_mean, sf_mean) + F.mse_loss(gf_std, sf_std)
        return loss_style
    
    def forward(self, content, style, lambda_style = 0.5, training=True):
        content_features, style_features, style_middle, adain = self.extract_features(content, style)
        generated_image, generate_features, generated_middle = self.generate(adain)

        if training:
        #Compute losses
            loss_content = self.content_loss(content_features, generate_features)
            loss_style = self.style_loss(style_middle, generated_middle)
            total_loss = loss_content + lambda_style * loss_style
            return total_loss, loss_content, loss_style
        else:
            #Inference mode: training=False
            return generated_image