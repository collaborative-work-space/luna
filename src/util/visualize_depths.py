import matplotlib.pyplot as plt
from torchvision.utils import save_image

def visualize_depths(predicted_depth, ground_truth_depth, step, output_dir="visualizations"):
    """
    Visualize and save depth images during training.

    Args:
        predicted_depth (torch.Tensor): Predicted depth tensor, shape (H, W).
        ground_truth_depth (torch.Tensor): Ground truth depth tensor, shape (H, W).
        step (int): Current training step (or epoch).
        output_dir (str): Directory to save visualizations.
    """
    predicted_depth = predicted_depth.cpu().numpy()
    ground_truth_depth = ground_truth_depth.cpu().numpy()

    plt.figure(figsize=(10, 5))

    # Predicted depth
    plt.subplot(1, 2, 1)
    plt.title(f"Predicted Depth - Step {step}")
    plt.imshow(predicted_depth, cmap='viridis')
    plt.colorbar()

    # Ground truth depth
    plt.subplot(1, 2, 2)
    plt.title(f"Ground Truth Depth - Step {step}")
    plt.imshow(ground_truth_depth, cmap='viridis')
    plt.colorbar()

    # Save the figure
    plt.tight_layout()
    plt.savefig(f"{output_dir}/depth_step_{step}.png")
    plt.close()
