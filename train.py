import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
from tqdm import tqdm
from model import UNet


# Custom Dataset for 3D CT volumes (slices as individual samples)
class CTDataset(Dataset):
    def __init__(self, img_dir, label_dir):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith('.npy')]
        # Verify matching label files
        self.img_files = [f for f in self.img_files if os.path.exists(os.path.join(label_dir, f))]
        if not self.img_files:
            raise ValueError("No matching .npy files found in img and label directories.")

        # Create a list of (file, slice_idx) pairs
        self.slice_list = []
        for f in self.img_files:
            img_path = os.path.join(self.img_dir, f)
            img = np.load(img_path)
            num_slices = img.shape[0]  # Assuming shape [slices, height, width]
            for slice_idx in range(num_slices):
                self.slice_list.append((f, slice_idx))

    def __len__(self):
        return len(self.slice_list)

    def __getitem__(self, idx):
        file_name, slice_idx = self.slice_list[idx]
        img_path = os.path.join(self.img_dir, file_name)
        label_path = os.path.join(self.label_dir, file_name)
        try:
            img = np.load(img_path)[slice_idx]  # Extract one 2D slice
            label = np.load(label_path)[slice_idx]  # Extract corresponding label slice
            # Normalize to [0, 1]
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            label = (label - label.min()) / (label.max() - label.min() + 1e-8)
            img = torch.from_numpy(img).float().unsqueeze(0)  # Shape: [1, height, width]
            label = torch.from_numpy(label).float().unsqueeze(0)  # Shape: [1, height, width]
        except Exception as e:
            raise RuntimeError(f"Error loading slice {slice_idx} from {file_name}: {e}")
        return img, label


# Calculate PSNR and SSIM
def calculate_metrics(pred, target):
    pred = pred.cpu().detach().numpy().squeeze()
    target = target.cpu().detach().numpy().squeeze()
    psnr_value = psnr(target, pred, data_range=1.0)
    ssim_value = ssim(target, pred, data_range=1.0)
    return psnr_value, ssim_value


# Save comparison images
def save_comparison_images(epoch, test_loader, model, device):
    model.eval()
    with torch.no_grad():
        for i, (img, label) in enumerate(test_loader):
            if i == 10:  # 不再保存后续图像
                break
            img = img.to(device)
            pred = model(img)
            pred = pred.cpu().numpy().squeeze()
            img = img.cpu().numpy().squeeze()
            label = label.numpy().squeeze()
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            plt.imshow(img, cmap='gray')
            plt.title('Input')
            plt.axis('off')
            plt.subplot(1, 3, 2)
            plt.imshow(pred, cmap='gray')
            plt.title('Prediction')
            plt.axis('off')
            plt.subplot(1, 3, 3)
            plt.imshow(label, cmap='gray')
            plt.title('Ground Truth')
            plt.axis('off')
            plt.savefig(f'./result/epoch{epoch}/comparison_slice_{i}.png', bbox_inches='tight')
            plt.close()


# Training function
def train_model(model, train_loader, val_loader, test_loader, criterion, optimizer, num_epochs, device):
    psnr_history = []
    ssim_history = []
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for img, label in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
            img, label = img.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(img)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * img.size(0)
        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for img, label in val_loader:
                img, label = img.to(device), label.to(device)
                output = model(img)
                loss = criterion(output, label)
                val_loss += loss.item() * img.size(0)
        val_loss /= len(val_loader.dataset)

        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        # Evaluate on test set
        psnr_list = []
        ssim_list = []
        with torch.no_grad():
            for img, label in test_loader:
                img, label = img.to(device), label.to(device)
                pred = model(img)
                for p, t in zip(pred, label):
                    p_psnr, p_ssim = calculate_metrics(p, t)
                    psnr_list.append(p_psnr)
                    ssim_list.append(p_ssim)

        avg_psnr = np.mean(psnr_list)
        avg_ssim = np.mean(ssim_list)
        psnr_history.append(avg_psnr)
        ssim_history.append(avg_ssim)
        print(f'Test PSNR: {avg_psnr:.4f}, Test SSIM: {avg_ssim:.4f}')

        # Save results
        os.makedirs(f'./result/epoch{epoch + 1}', exist_ok=True)
        plt.figure()
        plt.plot(range(1, epoch + 2), psnr_history, label='PSNR', color='#1f77b4')
        plt.plot(range(1, epoch + 2), ssim_history, label='SSIM', color='#ff7f0e')
        plt.xlabel('Epoch')
        plt.ylabel('Metric Value')
        plt.legend()
        plt.savefig(f'./result/epoch{epoch + 1}/metrics.png')
        plt.close()

        save_comparison_images(epoch + 1, test_loader, model, device)

        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f'./result/epoch{epoch + 1}/model_weights.pth')


# Main function
def main():
    # Hyperparameters
    batch_size = 16
    num_epochs = 50
    learning_rate = 0.001

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset and DataLoader
    try:
        dataset = CTDataset('./data/img', './data/label')
        print(f"Found {len(dataset)} slices from {len(dataset.img_files)} volumes")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)  # Batch size 1 for evaluation

    # Model, criterion, optimizer
    model = UNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train_model(model, train_loader, val_loader, test_loader, criterion, optimizer, num_epochs, device)


if __name__ == '__main__':
    main()