import os
import json

# 数据集图片目录
image_dir = "/opt/data/private/helingfeng/RIE/Show-o-main/data/imgs"

# JSON 文件路径
json_file_path = "/opt/data/private/helingfeng/RIE/Show-o-main/data/editing_instruction_dict.json"

# 获取图片ID
def extract_image_ids(image_dir):
    image_ids = []
    for image_name in os.listdir(image_dir):
        if image_name.endswith(('.jpg', '.png', '.jpeg')):  # 检查图片扩展名
            # 提取图片ID (假设图片ID在文件名中以 'COCO_train2014_000000<ID>.jpg' 格式)
            image_id = str(int(image_name.split('_')[-1].split('.')[0]))
            image_ids.append(image_id)
    return set(image_ids)

# 获取JSON中的ID
def extract_json_ids(json_file_path):
    with open(json_file_path, 'r') as f:
        data = json.load(f)
        json_ids = set(data.keys())  # 获取JSON的所有key
    return json_ids

# 找到缺失的图片
def find_missing_images(image_dir, json_file_path):
    image_ids = extract_image_ids(image_dir)
    json_ids = extract_json_ids(json_file_path)
    missing_ids = image_ids - json_ids  # 获取图片ID中不在JSON中的ID
    missing_ids_2 = json_ids - image_ids
    return missing_ids, missing_ids_2

# 执行检测
missing_ids, missing_ids_2 = find_missing_images(image_dir, json_file_path)

# 输出结果
if missing_ids:
    print("Missing IDs:")
    print(f"Image with ID {missing_ids} is missing from JSON file.")
    print(f"JSON file with ID {missing_ids_2} is missing from Image.")
    # for missing_id in missing_ids:
    #     print(f"Image with ID {missing_id} is missing from JSON file.")
    #     print(f"JSON file with ID {missing_id_2} is missing from Image.")

else:
    print("All images have corresponding prompts in the JSON file.")