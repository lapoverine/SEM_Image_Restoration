import cv2
from pathlib import Path
import argparse


def augment_and_patch(input_folder, output_folder, patch_size=256):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True, parents=True)
    
    for img_path in input_folder.glob("*.*"):
        if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
            continue
        
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Не удалось загрузить: {img_path.name}")
            continue
        
        # Конвертация в grayscale
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Список аугментаций одного изображения
        versions = []
        
        # 1. Оригинал
        versions.append(img)
        
        # 2. Повороты
        versions.append(cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE))
        versions.append(cv2.rotate(img, cv2.ROTATE_180))
        versions.append(cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE))
        
        # 3. Отражения
        versions.append(cv2.flip(img, 1))  # горизонтально
        versions.append(cv2.flip(img, 0))  # вертикально
        
        total_patches = 0
        # Для каждой версии - нарезка на патчи
        for v_idx, version in enumerate(versions):
            h, w = version.shape
            
            # Padding до кратности patch_size
            pad_h = (patch_size - h % patch_size) % patch_size
            pad_w = (patch_size - w % patch_size) % patch_size
            version = cv2.copyMakeBorder(version, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
            h, w = version.shape
            
            # Нарезка
            p_idx = 0
            for y in range(0, h, patch_size):
                for x in range(0, w, patch_size):
                    patch = version[y:y+patch_size, x:x+patch_size]
                    out_name = f"{img_path.stem}_v{v_idx}_p{p_idx}.png"
                    cv2.imwrite(str(output_folder / out_name), patch)
                    p_idx += 1
            total_patches += p_idx
        
        print(f"{img_path.name}: {len(versions)} версий -> {total_patches} патчей")
    
    print(f"\nГотово! Патчи сохранены в: {output_folder.absolute()}")

def main():
    parser = argparse.ArgumentParser(description="Аугментация и нарезка изображений на патчи")
    parser.add_argument("input", help="Папка с исходными изображениями")
    parser.add_argument("output", help="Папка для сохранения патчей")
    parser.add_argument("-s", "--patch-size", type=int, default=256, help="Размер патча (по умолчанию: 256)")
    
    args = parser.parse_args()
    
    augment_and_patch(args.input, args.output, args.patch_size)

if __name__ == "__main__":
    main()