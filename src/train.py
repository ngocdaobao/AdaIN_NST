import torch
from torchvision.transforms import ToPILImage
import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from utils import preprocess_image

def lr_scheduler(optimizer, epoch, init_lr=1e-4, lr_decay_epoch=10):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.1 ** (epoch // lr_decay_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def trainer(model, style_loader, content_loader, optimizer, device, num_epochs=10):
    all_loss = []
    loss_item = []
    model.train()
    style_iteration = iter(style_loader)
    # keep the original learning rate so scheduler computes decay from this base
    base_lr = optimizer.param_groups[0]['lr']
    for epoch in range(num_epochs):
        # update learning rate once per epoch (avoid repeated per-batch decay)
        optimizer = lr_scheduler(optimizer, epoch, init_lr=base_lr, lr_decay_epoch=5)
        print(f"Epoch {epoch+1}/{num_epochs}, Learning Rate: {optimizer.param_groups[0]['lr']}")
        loss_list = [] # Store losses for the epoch
        for content in tqdm.tqdm(content_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Content"):
            content = content.to(device)
            #Random style batch
            style_iter = iter(style_loader)
            try:
                style = next(style_iter)[0] #Get images
            except StopIteration:
                # Restart the style iterator if exhausted
                style_iter = iter(style_loader)
                style = next(style_iter)[0] #Get images
            style = style.to(device)

            loss, loss_content, loss_style = model(content, style, training=True)
            loss_list.append(loss.item())
            all_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # (learning rate updated once at start of epoch)

        avg_loss = sum(loss_list) / len(loss_list)
        loss_item.append(avg_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
    print("Training complete. Saving model...")
    torch.save(model.state_dict(), 'style_transfer_model.pth')
    #Visualize loss curve
    plt.figure(figsize=(10,5))
    iters = range(1, len(all_loss)+1)
    plt.plot(iters, all_loss, label='Training Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.show()
    plt.savefig('training_loss_curve.png')

    return loss_item, all_loss

def inference(model, content_path, style_path, device):
    to_pil_image = ToPILImage()

    # Open image and preprocess
    content_image = Image.open(content_path).convert('RGB')
    style_image = Image.open(style_path).convert('RGB')
    content = preprocess_image(content_image).to(device)
    style = preprocess_image(style_image).to(device)

    model.eval()
    with torch.no_grad():
        generated_image = model(content, style, training=False)
        generated_image = generated_image.clamp(0, 1).cpu()
        generated_image = to_pil_image(generated_image)
        # Save or display the generated image
        print("Inference complete. Saving generated image as 'generated_image.png'.")
        generated_image.save('generated_image.png')


