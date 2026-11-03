import argparse
import os
import uuid
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
from scipy.optimize import linear_sum_assignment

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff"}


def get_image_files(directory: Path) -> list[Path]:
    """Return a sorted list of valid image paths in a directory."""
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
    )


def load_preprocessed(path: Path, thumb_size: tuple[int, int] = (256, 256)) -> tuple[np.ndarray, float]:
    """
    Load an image, normalize EXIF orientation, record its aspect ratio,
    and resize to a fixed thumbnail array in RGB float32 format.
    """
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        width, height = img.size
        aspect_ratio = width / height
        # BILINEAR downsampling is fast and smooths minor compression artifacts
        img_resized = img.resize(thumb_size, Image.Resampling.BILINEAR)
        arr = np.asarray(img_resized, dtype=np.float32)
    return arr, aspect_ratio


def build_unredacted_mask(pix_arr: np.ndarray) -> np.ndarray:
    """
    Build a boolean mask (True = keep, False = ignore) that excludes pure white
    interiors and pure black borders caused by redaction rectangles in `pix`.
    """
    # Redacted interior: near-pure white across all RGB channels
    is_white = np.all(pix_arr > 250.0, axis=-1)
    # Redacted border: near-pure black across all RGB channels
    is_black = np.all(pix_arr < 5.0, axis=-1)
    return ~(is_white | is_black)


def compute_cost_matrix(
    pix_data: list[tuple[Path, np.ndarray, float]],
    qq_data: list[tuple[Path, np.ndarray, float]],
    ar_tolerance: float = 0.03,
) -> np.ndarray:
    """
    Compute pairwise dissimilarity between every image in `pix` and `qq`.
    Lower cost means higher similarity.
    """
    n_pix = len(pix_data)
    n_qq = len(qq_data)
    cost_matrix = np.full((n_pix, n_qq), fill_value=1e6, dtype=np.float32)

    # Precompute masks for all pix images
    pix_masks = [build_unredacted_mask(arr) for _, arr, _ in pix_data]

    for i, (_, pix_arr, pix_ar) in enumerate(pix_data):
        mask = pix_masks[i]
        valid_pixel_count = int(np.sum(mask))

        for j, (_, qq_arr, qq_ar) in enumerate(qq_data):
            # Skip expensive pixel math if aspect ratios don't match
            if abs(pix_ar - qq_ar) / max(pix_ar, qq_ar) > ar_tolerance:
                continue

            # Fallback to full-image comparison if the image is almost entirely white/black
            if valid_pixel_count < 100:
                diff = np.abs(pix_arr - qq_arr).mean()
            else:
                diff = np.abs(pix_arr[mask] - qq_arr[mask]).mean()

            cost_matrix[i, j] = diff

    return cost_matrix


def safe_rename_pairs(pairs: list[tuple[Path, Path]], dry_run: bool = False) -> None:
    """
    Rename each qq_path to match pix_path's filename (preserving qq's extension
    if formats differ, or matching pix's exact filename if extensions match).
    Uses a two-step temporary rename to avoid filename collisions.
    """
    temp_moves: list[tuple[Path, Path]] = []

    for pix_path, qq_path in pairs:
        # Keep qq's actual file extension so we don't corrupt file headers
        # (e.g., renaming a .jpg to .png without re-encoding).
        # Change `qq_path.suffix` to `pix_path.suffix` if exact extension matching is required.
        target_name = pix_path.stem + qq_path.suffix
        target_path = qq_path.with_name(target_name)

        if qq_path == target_path:
            print(f"[SKIP] {qq_path.name} already has the target name.")
            continue

        print(f"[MATCH] {qq_path.name}  -->  {target_path.name}")

        if not dry_run:
            temp_path = qq_path.with_name(f".tmp_{uuid.uuid4().hex}{qq_path.suffix}")
            os.rename(qq_path, temp_path)
            temp_moves.append((temp_path, target_path))

    if not dry_run:
        for temp_path, target_path in temp_moves:
            if target_path.exists():
                raise FileExistsError(
                    f"Collision detected: {target_path} already exists and was not part of the rename batch."
                )
            os.rename(temp_path, target_path)
        print(f"\nSuccessfully renamed {len(temp_moves)} file(s).")
    else:
        print("\nDry run complete. No files were renamed.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Match redacted images in `pix` with originals in `qq` and rename `qq` files."
    )
    parser.add_argument("pix", type=Path, help="Path to the `pix` directory (redacted images)")
    parser.add_argument("qq", type=Path, help="Path to the `qq` directory (original images to rename)")
    parser.add_argument(
        "--exact-ext",
        action="store_true",
        help="Force `qq` files to adopt the exact extension of `pix` (re-encodes if format differs).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview matches without renaming any files.",
    )
    args = parser.parse_args()

    if not args.pix.is_dir() or not args.qq.is_dir():
        raise ValueError("Both `pix` and `qq` must be valid directories.")

    pix_files = get_image_files(args.pix)
    qq_files = get_image_files(args.qq)

    if not pix_files or not qq_files:
        raise ValueError("One or both directories contain no supported image files.")

    if len(pix_files) != len(qq_files):
        print(
            f"Warning: Directory counts differ (pix={len(pix_files)}, qq={len(qq_files)}). "
            f"Will match the closest {min(len(pix_files), len(qq_files))} pairs."
        )

    print(f"Loading {len(pix_files)} images from '{args.pix}' and {len(qq_files)} from '{args.qq}'...")
    pix_data = [(p, *load_preprocessed(p)) for p in pix_files]
    qq_data = [(q, *load_preprocessed(q)) for q in qq_files]

    print("Computing similarity matrix...")
    cost_matrix = compute_cost_matrix(pix_data, qq_data)

    # Solve optimal 1-to-1 assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    pairs: list[tuple[Path, Path]] = []
    for r, c in zip(row_ind, col_ind):
        pix_path = pix_data[r][0]
        qq_path = qq_data[c][0]
        score = cost_matrix[r, c]
        if score >= 1e5:
            print(f"[WARN] Weak match (aspect ratio mismatch): {qq_path.name} -> {pix_path.name} (cost={score:.1f})")
        pairs.append((pix_path, qq_path))

    if args.exact_ext and not args.dry_run:
        # Handle cases where pix is .png and qq is .jpg and exact filename match is required
        temp_items: list[tuple[Path, Path, Path]] = []
        for pix_path, qq_path in pairs:
            target_path = qq_path.with_name(pix_path.name)
            if qq_path == target_path:
                continue
            print(f"[MATCH] {qq_path.name}  -->  {target_path.name}")
            temp_path = qq_path.with_name(f".tmp_{uuid.uuid4().hex}{qq_path.suffix}")
            os.rename(qq_path, temp_path)
            temp_items.append((temp_path, target_path, pix_path))

        for temp_path, target_path, pix_path in temp_items:
            if temp_path.suffix.lower() != pix_path.suffix.lower():
                with Image.open(temp_path) as img:
                    if pix_path.suffix.lower() in {".jpg", ".jpeg"} and img.mode != "RGB":
                        img = img.convert("RGB")
                    img.save(target_path)
                temp_path.unlink()
            else:
                os.rename(temp_path, target_path)
        print(f"\nSuccessfully processed {len(temp_items)} file(s).")
    else:
        safe_rename_pairs(pairs, dry_run=args.dry_run)


if __name__ == "__main__":
    main()