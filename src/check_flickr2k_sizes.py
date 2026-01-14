from pathlib import Path
from collections import Counter
from PIL import Image

def main():
    img_dir = Path("~/datasets/Flickr2K/Flickr2K_HR").expanduser()
    sizes = Counter()

    for img_path in img_dir.glob("*.png"):
        try:
            with Image.open(img_path) as img:
                sizes[img.size] += 1
        except Exception as e:
            print(f"Failed to read {img_path.name}: {e}")

    print(f"Total images: {sum(sizes.values())}")
    print("Most common resolutions:")
    for size, count in sizes.most_common(10):
        print(f"{size}: {count}")

if __name__ == "__main__":
    main()
