import cv2
from pathlib import Path
import argparse


def process_image(img_path, out_dir, patch_size=256):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  Ошибка: не удалось загрузить {img_path.name}")
        return 0
    
    # Конвертация в grayscale
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    h, w = img.shape
    
    # Padding до кратности patch_size
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size
    img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
    h, w = img.shape
    
    # Нарезка без перекрытия
    patch_count = 0
    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            patch = img[y:y+patch_size, x:x+patch_size]
            out_path = out_dir / f"{img_path.stem}_{y}_{x}.png"
            cv2.imwrite(str(out_path), patch)
            patch_count += 1
    
    return patch_count

def main():
    parser = argparse.ArgumentParser(description="Нарезка изображений на патчи")
    parser.add_argument("input", help="Путь к папке с исходными изображениями")
    parser.add_argument("-o", "--output", help="Путь для сохранения патчей (по умолчанию: input_patches)")
    parser.add_argument("-s", "--patch-size", type=int, default=256, help="Размер патча (по умолчанию: 256)")
    
    args = parser.parse_args()
    
    input_folder = Path(args.input)
    
    if args.output:
        output_folder = Path(args.output)
    else:
        output_folder = Path(f"{args.input}_patches")
    
    output_folder.mkdir(exist_ok=True, parents=True)
    
    print(f"Входная папка: {input_folder}")
    print(f"Выходная папка: {output_folder}")
    print(f"Размер патча: {args.patch_size}x{args.patch_size}")
    
    total_patches = 0
    processed = 0
    
    for img_path in input_folder.glob("*.*"):
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
            patches = process_image(img_path, output_folder, args.patch_size)
            if patches > 0:
                processed += 1
                total_patches += patches
                print(f"  {img_path.name}: {patches} патчей")
    
    print(f"Готово! Обработано: {processed}")
    print(f"Всего создано патчей: {total_patches}")

if __name__ == "__main__":
    main()