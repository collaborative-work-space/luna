import argparse
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


def add_gaussian_noise(image, mean=0, sigma=120):
    """
    Adds Gaussian noise to an image, ensuring proper scaling.
    :param image: Input image (uint8)
    :param mean: Mean of Gaussian noise
    :param sigma: Standard deviation of Gaussian noise
    :return: Noisy image
    """
    noise = np.random.normal(mean, sigma, image.shape).astype(np.float32)
    noisy_image = image.astype(np.float32) + noise  # Float operation to prevent overflow
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)  # Clip to valid range
    return noisy_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Synthesize additive Gaussian sensor noise for a scene's color frames "
                     "(paper Sec. IV-A: sigma=40/80/120 for Noise-Easy/Moderate/Strong)."
    )
    parser.add_argument("--input-dir", type=str, required=True,
                         help="Folder of clean color frames, e.g. data/ourreplica/office_3/original/color")
    parser.add_argument("--output-dir", type=str, required=True,
                         help="Folder to write noisy color frames to, e.g. data/ourreplica/office_3/noise-120/color")
    parser.add_argument("--sigma", type=float, default=120, help="Gaussian noise standard deviation")
    parser.add_argument("--pattern", type=str, default="0_*.png", help="Glob pattern for input frames")
    args = parser.parse_args()

    input_folder = Path(args.input_dir)
    output_folder = Path(args.output_dir)
    output_folder.mkdir(parents=True, exist_ok=True)

    for image_path in tqdm(sorted(input_folder.glob(args.pattern)), desc="Adding Gaussian noise"):
        img = cv2.imread(str(image_path))
        noisy_img = add_gaussian_noise(img, sigma=args.sigma)
        cv2.imwrite(str(output_folder / image_path.name), noisy_img)

    print(f"All images processed & saved in {output_folder}")
