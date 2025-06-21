import os
from PIL import Image

def calculate_average_image_size(folder_path):
    total_width = 0
    total_height = 0
    image_count = 0
    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(supported_extensions):
            image_path = os.path.join(folder_path, filename)
            try:
                with Image.open(image_path) as img:
                    width, height = img.size
                    total_width += width
                    total_height += height
                    image_count += 1
            except Exception as e:
                print(f"无法处理图像 {filename}: {e}")

        if image_count >= 40000:
            break

    if image_count == 0:
        print("未找到有效的图像文件。")
        return

    average_width = total_width / image_count
    average_height = total_height / image_count

    print(f"平均宽度: {average_width:.2f} 像素")
    print(f"平均高度: {average_height:.2f} 像素")

folder_path = "./data/mscoco/train2017"  
calculate_average_image_size(folder_path)