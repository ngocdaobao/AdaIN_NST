import torch
from torchvision.transforms import ToPILImage
import tqdm
from PIL import Image
from utils import preprocess_image


def trainer(model, style_loader, content_loader, optimizer, device, num_epochs=10):
    loss_item = []
    model.train()

    for epoch in range(num_epochs):
        loss_list = [] # Store losses for the epoch
        for content in tqdm.tqdm(content_loader, des=f"Epoch {epoch+1}/{num_epochs} - Content"):
            #Random style batch
            try:
                style = next(style_loader)

                content = content.to(device)
                style = style.to(device)

                loss, loss_content, loss_style = model(content, style, training=True)
                loss_list.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            #Reinitialize style loader if exhausted
            except StopIteration:
                style_loader = iter(style_loader)
                style = next(style_loader)
        avg_loss = sum(loss_list) / len(loss_list)
        loss_item.append(avg_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
    print("Training complete. Saving model...")
    torch.save(model.state_dict(), 'output/style_transfer_model.pth')

    return loss_item

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
        generated_image.save('output/generated_image.png')
