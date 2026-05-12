# Запуск скрипта python3 script.py /path/to/images/folder -o /path/to/results
import cv2
import numpy as np
import os
import sys
import argparse
from pathlib import Path

def remove_info_panels(image_path, debug=False):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Не удалось загрузить изображение: {image_path}")
        return None, None
    
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    _, thresh_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, thresh_normal = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    h_inv = analyze_edges(gray, thresh_inv, h, w, debug)
    h_normal = analyze_edges(gray, thresh_normal, h, w, debug)
    
    if h_inv > h_normal:
        thresh = thresh_inv
        if debug:
            print("Выбран inverted режим (текст светлый на темном фоне)")
    else:
        thresh = thresh_normal
        if debug:
            print("Выбран normal режим (текст темный на светлом фоне)")
    
    top_region = thresh[:int(h*0.2), :]
    top_text = analyze_region(top_region, w)
    bottom_region = thresh[int(h*0.8):, :]
    bottom_text = analyze_region(bottom_region, w)
    left_region = thresh[:, :int(w*0.2)]
    left_text = analyze_region(left_region, h)
    right_region = thresh[:, int(w*0.8):]
    right_text = analyze_region(right_region, h)
    
    if debug:
        print(f"Текст на верхнем крае: {top_text:.4f}")
        print(f"Текст на нижнем крае: {bottom_text:.4f}")
        print(f"Текст на левом крае: {left_text:.4f}")
        print(f"Текст на правом крае: {right_text:.4f}")
    
    crop_x1, crop_y1, crop_x2, crop_y2 = 0, 0, w, h
    
    if top_text > 0.03:
        crop_y1 = find_text_boundary(thresh, from_top=True)
        if debug:
            print(f"Обрезаем сверху до {crop_y1}")
    
    if bottom_text > 0.03:
        crop_y2 = find_text_boundary(thresh, from_top=False)
        if debug:
            print(f"Обрезаем снизу от {crop_y2}")
    
    if left_text > 0.03:
        crop_x1 = find_text_boundary_horizontal(thresh, from_left=True)
        if debug:
            print(f"Обрезаем слева до {crop_x1}")
    
    if right_text > 0.03:
        crop_x2 = find_text_boundary_horizontal(thresh, from_left=False)
        if debug:
            print(f"Обрезаем справа от {crop_x2}")
    
    result = img[crop_y1:crop_y2, crop_x1:crop_x2]
    
    if debug:
        print(f"\nОригинал: {w}x{h}")
        print(f"Обрезано: {result.shape[1]}x{result.shape[0]}")
    
    return result, (crop_y1, crop_y2, crop_x1, crop_x2)

def analyze_edges(gray, thresh, h, w, debug=False):
    edges = [
        thresh[:int(h*0.1), :],
        thresh[int(h*0.9):, :],
        thresh[:, :int(w*0.1)],
        thresh[:, int(w*0.9):]
    ]
    
    total_density = 0
    for edge in edges:
        if edge.size > 0:
            density = np.sum(edge == 255) / edge.size
            total_density += density
    
    return total_density / 4

def analyze_region(region, length):
    if region.shape[0] < 10:
        pixel_count = np.sum(region == 255)
        return pixel_count / (region.shape[0] * region.shape[1])
    else:
        pixel_count = np.sum(region == 255)
        return pixel_count / (region.shape[0] * region.shape[1])

def find_text_boundary(thresh, from_top=True, padding=0):
    h, w = thresh.shape
    
    if from_top:
        for y in range(h):
            row_density = np.sum(thresh[y, :] == 255) / w
            if row_density < 0.01:
                return max(0, y - padding)
        return 0
    else:
        for y in range(h-1, -1, -1):
            row_density = np.sum(thresh[y, :] == 255) / w
            if row_density < 0.01:
                return min(h, y + padding)
        return h

def find_text_boundary_horizontal(thresh, from_left=True, padding=0):
    h, w = thresh.shape
    
    if from_left:
        for x in range(w):
            col_density = np.sum(thresh[:, x] == 255) / h
            if col_density < 0.01:
                return max(0, x - padding)
        return 0
    else:
        for x in range(w-1, -1, -1):
            col_density = np.sum(thresh[:, x] == 255) / h
            if col_density < 0.01:
                return min(w, x + padding)
        return w

def get_relative_path(full_path, base_path):
    try:
        rel_path = os.path.relpath(full_path, base_path)
        return os.path.dirname(rel_path)
    except:
        return ""

def process_single_image(image_path, output_base_dir, base_input_dir, debug=False):
    try:
        result, (crop_y1, crop_y2, crop_x1, crop_x2) = remove_info_panels(image_path, debug)
        
        if result is None:
            return False
        
        img = cv2.imread(image_path)
        if img is None:
            return False
            
        h, w = img.shape[:2]
        
        rel_path = get_relative_path(image_path, base_input_dir)
        original_filename = Path(image_path).name
        
        if rel_path and rel_path != '.':
            main_dir = os.path.join(output_base_dir, 'main_content', rel_path)
            info_dir = os.path.join(output_base_dir, 'info_panels', rel_path)
        else:
            main_dir = os.path.join(output_base_dir, 'main_content')
            info_dir = os.path.join(output_base_dir, 'info_panels')
        
        os.makedirs(main_dir, exist_ok=True)
        os.makedirs(info_dir, exist_ok=True)
        
        panel_found = False
        
        if crop_y1 > 0:
            info_panel = img[crop_y1 - 1:, :]
            main_content = img[:crop_y1 - 1, :]
            panel_found = True
            print(f"  Информационная строка найдена сверху в {original_filename}")
            
            if main_content is not None and main_content.size > 0:
                cv2.imwrite(os.path.join(main_dir, original_filename), main_content)
            if info_panel is not None and info_panel.size > 0:
                cv2.imwrite(os.path.join(info_dir, original_filename), info_panel)
                
        elif crop_y2 < h:
            info_panel = img[:crop_y2 + 1, :]
            main_content = img[crop_y2 + 1:, :]
            panel_found = True
            print(f"  Информационная строка найдена снизу в {original_filename}")
            
            if main_content is not None and main_content.size > 0:
                cv2.imwrite(os.path.join(main_dir, original_filename), main_content)
            if info_panel is not None and info_panel.size > 0:
                cv2.imwrite(os.path.join(info_dir, original_filename), info_panel)
                
        elif crop_x1 > 0:
            info_panel = img[:, crop_x1 - 1:]
            main_content = img[:, :crop_x1 - 1]
            panel_found = True
            print(f"  Информационная строка найдена слева в {original_filename}")
            
            if main_content is not None and main_content.size > 0:
                cv2.imwrite(os.path.join(main_dir, original_filename), main_content)
            if info_panel is not None and info_panel.size > 0:
                cv2.imwrite(os.path.join(info_dir, original_filename), info_panel)
                
        elif crop_x2 < w:
            info_panel = img[:, :crop_x2 + 1]
            main_content = img[:, crop_x2 + 1:]
            panel_found = True
            print(f"  Информационная строка найдена справа в {original_filename}")
            
            if main_content is not None and main_content.size > 0:
                cv2.imwrite(os.path.join(main_dir, original_filename), main_content)
            if info_panel is not None and info_panel.size > 0:
                cv2.imwrite(os.path.join(info_dir, original_filename), info_panel)
        
        if not panel_found:
            print(f"  Информационная строка не найдена в {original_filename}, сохраняем в main_content")
            cv2.imwrite(os.path.join(main_dir, original_filename), img)
        
        return True
        
    except Exception as e:
        print(f"  Ошибка при обработке {image_path}: {str(e)}")
        return False

def process_folder(input_folder, output_base_dir, debug=False):
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
    
    image_files = []
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(supported_formats):
                full_path = os.path.join(root, file)
                image_files.append(full_path)
    
    if not image_files:
        print(f"Не найдено изображений в папке: {input_folder}")
        return
    
    print(f"Найдено {len(image_files)} изображений")
    
    success_count = 0
    for i, image_path in enumerate(image_files, 1):
        rel_path = os.path.relpath(image_path, input_folder)
        print(f"[{i}/{len(image_files)}] Обработка: {rel_path}")
        
        if process_single_image(image_path, output_base_dir, input_folder, debug):
            success_count += 1
    
    main_count = 0
    info_count = 0
    
    main_dir = os.path.join(output_base_dir, 'main_content')
    info_dir = os.path.join(output_base_dir, 'info_panels')
    
    if os.path.exists(main_dir):
        for root, dirs, files in os.walk(main_dir):
            for file in files:
                if file.lower().endswith(supported_formats):
                    main_count += 1
    
    if os.path.exists(info_dir):
        for root, dirs, files in os.walk(info_dir):
            for file in files:
                if file.lower().endswith(supported_formats):
                    info_count += 1
    
    print(f"\nУспешно обработано: {success_count} из {len(image_files)}")
    print(f"   main_content/   - {main_count} файлов (основное содержимое)")
    print(f"   info_panels/    - {info_count} файлов (информационные строки)")

def main():
    parser = argparse.ArgumentParser(description='Удаление информационных строк с изображений')
    parser.add_argument('input', help='Путь к входной папке с изображениями или к одному файлу')
    parser.add_argument('-o', '--output', help='Путь к выходной папке (по умолчанию: processed_results)', 
                       default='processed_results')
    parser.add_argument('-d', '--debug', action='store_true', help='Режим отладки')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Ошибка: Путь {args.input} не существует")
        sys.exit(1)
    
    if os.path.isfile(args.input):
        print(f"Обработка одного файла: {args.input}")
        input_dir = os.path.dirname(args.input) or '.'
        process_single_image(args.input, args.output, input_dir, args.debug)
    elif os.path.isdir(args.input):
        process_folder(args.input, args.output, args.debug)
    else:
        print(f"Ошибка: {args.input} не является файлом или папкой")
        sys.exit(1)

if __name__ == "__main__":
    main()