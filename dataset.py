import kagglehub
import os
import numpy as np
from PIL import Image

datasets_path = os.path.join(os.getcwd(), 'datasets')
image_size = (64, 64)

def find_jpg_files(directory):
    jpg_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.jpg'):
                jpg_files.append(os.path.join(root, file))
    return jpg_files

all_images = find_jpg_files(datasets_path)
print(len(all_images))

def save_all_images_to_npz(all_images):
    print("Saving images to ", "all_dogs.npz")
    final_array = np.array([np.array(Image.open(path).resize(image_size).convert("RGB")) for path in all_images], dtype=np.float32)
    print("Images were got")
    final_array /= 255.0
    print("Images were normalized")
    np.savez_compressed("all_dogs.npz", dogs=final_array, allow_pickle=True)
    print("Images were saved")

def get_all_images(all_images):
    if os.path.exists(os.path.join(os.getcwd(), "all_dogs.npz")):
        print("Dataset file was found. Loading...")
        return np.load("all_dogs.npz")['dogs']
    else:
        print("Dataset file was not found. Creating...")
        save_all_images_to_npz(all_images)
        return get_all_images(all_images)

full_dataset = get_all_images(all_images)
print("Dataset shape: ", full_dataset.shape)