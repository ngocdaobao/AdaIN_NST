import torch
import torch.optim as optim
import argparse

from utils import data_loader
from model import StyleTransferModel, VGG19Encoder, Decoder
from train import trainer, inference

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Neural Style Transfer Training')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--encoder', type=str, default='vgg19')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lambda_style', type=float, default=0.5, help='Weight for style loss, adjust for temperature of style transfer')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--shuffle', type=bool, default=True)

    args = parser.parse_args()
    epoch = args.epoch
    pretrained_encoder = args.encoder
    batch_size = args.batch_size
    lr = args.lr
    lambda_style = args.lambda_style
    num_workers = args.num_workers
    shuffle = args.shuffle
    seed = args.seed

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    style_loader, content_loader = data_loader(batch_size=batch_size, num_workers=num_workers, seed=seed, shuffle=shuffle)

    if pretrained_encoder == 'vgg19':
        encoder_model = VGG19Encoder()
        
    ckpt = torch.load('/kaggle/input/nst-25epoch/pytorch/default/1/style_transfer_model.pth')
    model = StyleTransferModel(encoder_model=encoder_model)
    model.load_state_dict(ckpt)
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_item = trainer(model, style_loader, content_loader, optimizer, device, num_epochs=epoch)
    print("Training finished.")

    # Inference
    # inference(model=model, content_path='AdaIN_NST/example/content.jpg', style_path='AdaIN_NST/example/style.jpg', device=device)
    # print("Inference finished.")






