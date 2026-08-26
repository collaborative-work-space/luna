import argparse
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Synthesize horizontal motion blur for a scene's color frames "
                     "(paper Sec. IV-A: kernel=10/20/30px for Blur-Easy/Moderate/Strong)."
    )
    parser.add_argument("--input-dir", type=str, required=True,
                         help="Folder of clean color frames, e.g. data/ourreplica/office_3/original/color")
    parser.add_argument("--output-dir", type=str, required=True,
                         help="Folder to write blurred color frames to, e.g. data/ourreplica/office_3/blur-10/color")
    parser.add_argument("--kernel-size", type=int, default=10, help="Motion-blur kernel size in pixels")
    parser.add_argument("--pattern", type=str, default="0_*.png", help="Glob pattern for input frames")
    args = parser.parse_args()

    input_folder = Path(args.input_dir)
    output_folder = Path(args.output_dir)
    output_folder.mkdir(parents=True, exist_ok=True)

    # Horizontal motion blur kernel
    kernel = np.zeros((1, args.kernel_size))
    kernel[0, :] = 1.0 / args.kernel_size

    for image_path in tqdm(sorted(input_folder.glob(args.pattern)), desc="Blurring images"):
        img = cv2.imread(str(image_path))
        blurred = cv2.filter2D(img, -1, kernel)
        cv2.imwrite(str(output_folder / image_path.name), blurred)

    print(f"All images processed & saved in {output_folder}")
