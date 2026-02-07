import os
from pathlib import Path

import requests
from requests.exceptions import RequestException


def ensure_directories(base_data_dir: Path) -> None:
    """
    Create the required data directory hierarchy if it does not exist.
    """
    subdirs = [
        base_data_dir / "01_fundamentals",
        base_data_dir / "02_papers",
        base_data_dir / "03_implementation",
    ]
    for d in subdirs:
        d.mkdir(parents=True, exist_ok=True)


def download_file(url: str, target_path: Path) -> None:
    """
    Download a single file with basic error handling.

    - Skips download if the file already exists.
    - Prints a status line for each file.
    """
    file_name = target_path.name

    if target_path.exists():
        print(f"Skipping {file_name}, already exists.")
        return

    print(f"Downloading {file_name} from {url} ...")

    try:
        # A simple User-Agent header helps with some hosts (e.g., arXiv) that
        # may be more strict about generic clients.
        response = requests.get(
            url,
            stream=True,
            timeout=60,
            headers={"User-Agent": "Mozilla/5.0 (RAG-Data-Downloader)"},
        )
        response.raise_for_status()

        # Write to disk in chunks
        with target_path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        print(f"Finished downloading {file_name}.")
    except RequestException as exc:
        print(f"Error downloading {file_name} from {url}: {exc}")


def main() -> None:
    """
    Entry point for setting up the RAG data directory.

    This script is intended to be run from the project root:

        python scripts/download_data.py
    """
    # Resolve project root as parent of this file's directory
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"

    ensure_directories(data_dir)

    # Define file lists
    papers_dir = data_dir / "02_papers"
    impl_dir = data_dir / "03_implementation"
    fundamentals_dir = data_dir / "01_fundamentals"

    downloads = [
        # 02_papers (ArXiv / JMLR)
        (
            "Attention Is All You Need",
            "https://arxiv.org/pdf/1706.03762.pdf",
            papers_dir / "attention_is_all_you_need.pdf",
        ),
        (
            "ResNet",
            "https://arxiv.org/pdf/1512.03385.pdf",
            papers_dir / "resnet.pdf",
        ),
        (
            "Adam Optimizer",
            "https://arxiv.org/pdf/1412.6980.pdf",
            papers_dir / "adam_optimizer.pdf",
        ),
        (
            "DDPM (Diffusion)",
            "https://arxiv.org/pdf/2006.11239.pdf",
            papers_dir / "ddpm_diffusion.pdf",
        ),
        (
            "Dropout",
            "https://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf",
            papers_dir / "dropout_srivastava14a.pdf",
        ),
        # 03_implementation (Open Source Books)
        (
            "Deep Learning with PyTorch",
            "https://isip.piconepress.com/courses/temple/ece_4822/resources/books/Deep-Learning-with-PyTorch.pdf",
            impl_dir / "deep_learning_with_pytorch.pdf",
        ),
        (
            "The Little Book of Deep Learning",
            "https://fleuret.org/public/lbdl.pdf",
            impl_dir / "the_little_book_of_deep_learning.pdf",
        ),
        # 01_fundamentals
        # NOTE: Standard textbooks (Goodfellow, etc.) often require manual download
        # due to copyright/hosting. Please manually add "Deep Learning Book" here.
        (
            "Linear Algebra for ML (Part 1)",
            "https://arxiv.org/pdf/1802.01528.pdf",
            fundamentals_dir / "linear_algebra_for_ml_part1.pdf",
        ),
    ]

    for title, url, path in downloads:
        print(f"\n== {title} ==")
        download_file(url, path)

    print("\nAll downloads attempted. Check messages above for any failures.")


if __name__ == "__main__":
    main()

