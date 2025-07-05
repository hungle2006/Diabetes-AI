import os
import cv2
import numpy as np
import pandas as pd
import glob
import zipfile
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from google.colab import drive

# 1. Mount Google Drive
drive.mount('/content/drive')

# 2. Define paths
drive_folder = "your link on Google Drive"
extract_root = "/content/extracted_zip_files"
os.makedirs(extract_root, exist_ok=True)

# unzip file ZIP
zip_files = glob.glob(os.path.join(drive_folder, "*.zip"))
for zip_path in zip_files:
    zip_name = os.path.basename(zip_path).replace(".zip", "")
    extract_path = os.path.join(extract_root, zip_name)
    if not os.path.exists(extract_path):  # Kiểm tra nếu chưa giải nén
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print(f"✅ Đã giải nén: {zip_path} → {extract_path}")
    else:
        print(f"Thư mục {extract_path} đã tồn tại, bỏ qua giải nén.")

# 3. Read CSV
csv_path = os.path.join(drive_folder, "messidor_data.csv")
df_train = pd.read_csv(csv_path)

# 4. Process iamges
def crop_image_from_gray_to_color(img, tol=7):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = gray > tol
    if mask.sum() == 0:
        return img
    rows = mask.any(axis=1)
    cols = mask.any(axis=0)
    cropped_img = img[np.ix_(rows, cols)]
    return cropped_img

def load_ben_color(path, sigmaX=10, IMG_SIZE=224):
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Không thể đọc được ảnh từ đường dẫn: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray_to_color(image, tol=7)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)
    return image

# 5. Xử lý và lưu ảnh
train_img_folder = os.path.join(extract_root, "messidor-2", "messidor-2","preprocess")  # Đường dẫn đến thư mục chứa ảnh
processed_folder = "/content/processed_train_images"
os.makedirs(processed_folder, exist_ok=True)

processed_ids = []
processed_paths = []

for idx, row in df_train.iterrows():
    img_filename = f"{row['id_code']}"
    img_path = os.path.join(train_img_folder, img_filename)

    if os.path.exists(img_path):
        try:
            proc_img = load_ben_color(img_path, sigmaX=10, IMG_SIZE=224)
            proc_img_bgr = cv2.cvtColor(proc_img, cv2.COLOR_RGB2BGR)
            save_path = os.path.join(processed_folder, img_filename)
            cv2.imwrite(save_path, proc_img_bgr)
            processed_ids.append(row['id_code'])
            processed_paths.append(save_path)
        except Exception as e:
            print(f"Lỗi khi xử lý ảnh {img_filename}: {e}")
    else:
        print(f"Không tìm thấy ảnh {img_filename} tại {img_path}")

print(f"Đã xử lý thành công {len(processed_ids)} ảnh.")

# 6. Upgrade with many images have been processed
df_train_processed = df_train[df_train['id_code'].isin(processed_ids)].copy()
df_train_processed['image_path'] = [os.path.join(processed_folder, f"{id_code}") for id_code in df_train_processed['id_code']]

# 7. Devide into train, validation, test
x = df_train_processed['image_path']
y = df_train_processed['diagnosis']

# shuffle images
x, y = shuffle(x, y, random_state=42)

# devide into train+validation and test (80% - 20%)
x_temp, test_x, y_temp, test_y = train_test_split(x, y, test_size=0.20, stratify=y, random_state=42)

# devide into train and validation (85% train, 15% val trong 80% dữ liệu ban đầu)
train_x, valid_x, train_y, valid_y = train_test_split(x_temp, y_temp, test_size=0.15/0.80, stratify=y_temp, random_state=42)

# print test information
print("Train X size:", len(train_x))
print("Train y size:", len(train_y))
print("Valid X size:", len(valid_x))
print("Valid y size:", len(valid_y))
print("Test X size:", len(test_x))
print("Test y size:", len(test_y))
