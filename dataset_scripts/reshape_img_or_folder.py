import argparse
from pathlib import Path
from PIL import Image


def crop_image(input_path: Path, output_path: Path, bottom: int = 0):
    try:
        img = Image.open(input_path)
    except Exception as e:
        print(f"  Ошибка открытия {input_path.name}: {e}")
        return False

    width, height = img.size
    original_size = (width, height)

    l = 0
    r = width
    t = 0
    b = height - bottom

    if r <= l or b <= t:
        print(f"  Ошибка: после обрезки не остаётся изображения (слишком большой отступ) в файле {input_path.name}")
        return False
    
    cropped = img.crop((l, t, r, b))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        cropped.save(output_path)
        print(f"  {input_path.name}: {original_size} -> {cropped.size}")
        return True
    except Exception as e:
        print(f"  Ошибка сохранения {output_path.name}: {e}")
        return False


def process_folder(input_folder: str, output_folder: str, bottom: int = 0, extensions: tuple = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'), recursive: bool = False):
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    
    if not input_path.exists():
        print(f"Ошибка: папка {input_folder} не существует")
        return
    
    if not input_path.is_dir():
        print(f"Ошибка: {input_folder} не является папкой")
        return
    
    image_files = []
    for ext in extensions:
        if recursive:
            image_files.extend(input_path.rglob(f"*{ext}"))
            image_files.extend(input_path.rglob(f"*{ext.upper()}"))
        else:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))
    
    # Удаляем дубликаты
    image_files = list(set(image_files))
    
    if not image_files:
        print(f"Не найдено изображений в папке {input_folder}")
        print(f"Поддерживаемые расширения: {', '.join(extensions)}")
        return
    
    print(f"\nНайдено {len(image_files)} изображений")
    print(f"Входная папка: {input_path.absolute()}")
    print(f"Выходная папка: {output_path.absolute()}")
    
    successful = 0
    failed = 0
    
    for img_file in image_files:
        if recursive:
            output_file = output_path / img_file.relative_to(input_path)
        else:
            output_file = output_path / img_file.name
        
        output_file = output_file.with_suffix(img_file.suffix)
        success = crop_image(img_file, output_file, bottom)
        
        if success:
            successful += 1
        else:
            failed += 1
    
    print(f"Готово! Успешно: {successful}, Ошибок: {failed}")
    print(f"Результаты сохранены в: {output_path.absolute()}")


def main():
    parser = argparse.ArgumentParser(
        description="Обрезка фото снизу (удаление информационной линии)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("input", help="Путь к исходному изображению или папке")
    parser.add_argument("-o", "--output", help="Путь для сохранения результата (файл или папка)")
    parser.add_argument("-b", "--bottom", type=int, default=0, help="Пикселей отрезать снизу (например, для удаления информационной линии)")
    
    parser.add_argument("-f", "--folder", action="store_true", help="Обработать все изображения в папке (входной путь - папка)")
    parser.add_argument("-R", "--recursive", action="store_true", help="Рекурсивно обрабатывать подпапки")
    parser.add_argument("-e", "--extensions", nargs="+", default=['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'],
                       help="Расширения файлов для обработки (по умолчанию: .jpg .jpeg .png .bmp .tiff .webp)")
    
    args = parser.parse_args()
    
    is_folder = args.folder or Path(args.input).is_dir()
    
    if is_folder:
        if not args.output:
            output_folder = str(Path(args.input).with_name(f"{Path(args.input).name}_cropped"))
        else:
            output_folder = args.output
        
        extensions = tuple(ext.lower() if ext.startswith('.') else f'.{ext.lower()}' 
                          for ext in args.extensions)
        
        process_folder(
            input_folder=args.input,
            output_folder=output_folder,
            bottom=args.bottom,
            extensions=extensions,
            recursive=args.recursive
        )
    else:
        # Обработка одного файла
        if not args.output:
            output_path = Path(args.input).with_name(f"{Path(args.input).stem}_cropped{Path(args.input).suffix}")
        else:
            output_path = args.output
        
        crop_image(Path(args.input), Path(output_path), args.bottom)


if __name__ == "__main__":
    main()