import cv2
import numpy as np
from pathlib import Path
import re
from collections import defaultdict
import argparse

class ReversePipeline:
    def __init__(self, patch_size=256):
        self.patch_size = patch_size
    
    def step1_reverse_augmentation(self, augmented_patches_dir, output_dir):
        aug_path = Path(augmented_patches_dir)
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        patches_by_image = defaultdict(list)
        
        for patch_file in aug_path.rglob("*.png"):
            match = re.match(r"(.+?)_orig_(\d+)_(\d+)\.png", patch_file.name)
            if match:
                base_name = match.group(1)
                aug_type = match.group(2)
                y = int(match.group(3))
                x = int(match.group(4))
                patches_by_image[base_name].append((x, y, patch_file))
        
        reconstructed_images = {}
        
        for base_name, patches in patches_by_image.items():
            max_x = max(p[0] for p in patches) + self.patch_size
            max_y = max(p[1] for p in patches) + self.patch_size
            
            reconstructed = np.zeros((max_y, max_x), dtype=np.uint8)
            
            for x, y, patch_file in patches:
                patch = cv2.imread(str(patch_file), cv2.IMREAD_GRAYSCALE)
                if patch is not None:
                    reconstructed[y:y+self.patch_size, x:x+self.patch_size] = patch
            
            output_file = out_path / f"{base_name}.png"
            cv2.imwrite(str(output_file), reconstructed)
            reconstructed_images[base_name] = reconstructed
        
        return reconstructed_images
    
    def step2_add_panels_bottom(self, main_content_dir, info_panels_dir, output_dir):
        main_path = Path(main_content_dir)
        info_path = Path(info_panels_dir)
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        main_files = {}
        for f in main_path.glob("*.*"):
            if f.suffix.lower() in ('.png', '.jpg', '.jpeg'):
                name = f.stem
                name = re.sub(r'_(reconstructed|orig|original)$', '', name)
                main_files[name] = f
        
        info_files = {}
        for f in info_path.glob("*.*"):
            if f.suffix.lower() in ('.png', '.jpg', '.jpeg'):
                name = f.stem
                name = re.sub(r'_(panel|info|infopanel)$', '', name)
                info_files[name] = f
        
        common_names = set(main_files.keys()) & set(info_files.keys())
        
        print(f"Найдено пар изображений: {len(common_names)}")
        
        if len(common_names) == 0:
            print("Не найдено совпадающих имен")
            print(f"main_content имена: {list(main_files.keys())[:5]}")
            print(f"info_panels имена: {list(info_files.keys())[:5]}")
            return {}
        
        restored = {}
        
        for name in common_names:
            print(f"\n  Восстановление: {name}")
            
            main_img = cv2.imread(str(main_files[name]))
            info_img = cv2.imread(str(info_files[name]))
            
            if main_img is None or info_img is None:
                print(f"    Ошибка загрузки")
                continue
            
            channels_main = main_img.shape[2] if len(main_img.shape) == 3 else 1
            channels_info = info_img.shape[2] if len(info_img.shape) == 3 else 1
            
            if channels_info == 3 and channels_main == 1:
                main_img = cv2.cvtColor(main_img, cv2.COLOR_GRAY2BGR)
            elif channels_main == 3 and channels_info == 1:
                info_img = cv2.cvtColor(info_img, cv2.COLOR_GRAY2BGR)

            h_main, w_main = main_img.shape[:2]
            h_info, w_info = info_img.shape[:2]
            
            if w_info != w_main:
                info_img = cv2.resize(info_img, (w_main, h_info))
                print(f"    Панель изменена: {w_info}x{h_info} -> {w_main}x{h_info}")
            
            result = np.vstack([main_img, info_img])
            
            output_file = out_path / f"{name}_original_with_panel.png"
            cv2.imwrite(str(output_file), result)
            restored[name] = result
        
        return restored
    
    def full_reverse_pipeline(self, augmented_patches_dir, main_content_dir, info_panels_dir, output_dir):
        final_output = Path(output_dir)
        final_output.mkdir(parents=True, exist_ok=True)
        
        print("Восстановление из аугментированных патчей")
        step1_output = final_output / "step1_reconstructed_without_panels"
        reconstructed = self.step1_reverse_augmentation(augmented_patches_dir, step1_output)
        
        print("Добавление информационных панелей снизу")
        step2_output = final_output / "step2_final_with_panels"
        final_images = self.step2_add_panels_bottom(main_content_dir, info_panels_dir, step2_output)
        
        print(f"Восстановлено изображений без панели: {len(reconstructed)}")
        print(f"Создано итоговых изображений с панелями: {len(final_images)}")
        
        return reconstructed, final_images


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Из аугментированных патчей в исходное фото с информационной строкой')
    parser.add_argument('--augmented_patches', type=str, required=True,
                        help='Путь к папке с аугментированными патчами')
    parser.add_argument('--main_content', type=str, required=True,
                        help='Путь к папке с основным содержимым (после удаления информационной строки)')
    parser.add_argument('--info_panels', type=str, required=True,
                        help='Путь к папке с вырезанными информационными строками')
    parser.add_argument('--output', type=str, required=True,
                        help='Выходная папка для результатов')
    parser.add_argument('--patch_size', type=int, default=256,
                        help='Размер патча (по умолчанию 256)')
    
    args = parser.parse_args()
    
    pipeline = ReversePipeline(patch_size=args.patch_size)
    
    pipeline.full_reverse_pipeline(
        augmented_patches_dir=args.augmented_patches,
        main_content_dir=args.main_content,
        info_panels_dir=args.info_panels,
        output_dir=args.output
    )