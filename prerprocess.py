import os
import shutil

def preprocess_training(base_dir):
    clean_dir = os.path.join(base_dir, 'clean')
    noisy_dir = os.path.join(base_dir, 'noisy')

    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(noisy_dir, exist_ok=True)

    source_dirs = []
    for root, dirs, files in os.walk(base_dir):
        for dir_name in dirs:
            if 'GT' in dir_name:
                source_dirs.append(os.path.join(root, dir_name))

    if not source_dirs:
        raise ValueError("No directory containing 'GT' found")

    for source_dir in source_dirs:
        for filename in os.listdir(source_dir):
            if filename.endswith('.jpg'):
                shutil.move(os.path.join(source_dir, filename), os.path.join(clean_dir, filename))

    for root, dirs, files in os.walk(base_dir):
        for dir_name in dirs:
            if dir_name not in ['clean', 'noisy'] and 'GT' not in dir_name:
                current_dir = os.path.join(root, dir_name)
                for filename in os.listdir(current_dir):
                    if filename.endswith('.jpg'):
                        shutil.move(os.path.join(current_dir, filename), os.path.join(noisy_dir, filename))

    for root, dirs, files in os.walk(base_dir, topdown=False):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            if dir_name not in ['clean', 'noisy']:
                shutil.rmtree(dir_path)

    print('Training preprocessing done')

def preprocess_validation(base_dir):
    clean_dir = os.path.join(base_dir, 'clean')
    noisy_dir = os.path.join(base_dir, 'noisy')

    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(noisy_dir, exist_ok=True)

    source_dirs = []
    for root, dirs, files in os.walk(base_dir):
        for dir_name in dirs:
            if 'GT' in dir_name:
                source_dirs.append(os.path.join(root, dir_name))

    if not source_dirs:
        raise ValueError("No directory containing 'GT' found")

    for source_dir in source_dirs:
        for filename in os.listdir(source_dir):
            if filename.endswith('.jpg'):
                shutil.move(os.path.join(source_dir, filename), os.path.join(clean_dir, filename))

    for root, dirs, files in os.walk(base_dir):
        for dir_name in dirs:
            if dir_name not in ['clean', 'noisy'] and 'GT' not in dir_name:
                current_dir = os.path.join(root, dir_name)
                for filename in os.listdir(current_dir):
                    if filename.endswith('.jpg'):
                        shutil.move(os.path.join(current_dir, filename), os.path.join(noisy_dir, filename))

    for root, dirs, files in os.walk(base_dir, topdown=False):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            if dir_name not in ['clean', 'noisy']:
                shutil.rmtree(dir_path)

    print('Validation preprocessing done')

data_dir = '/data/event/unid/data/'
training_base_dir = os.path.join(data_dir, 'Training')
validation_base_dir = os.path.join(data_dir, 'Validation')

preprocess_training(training_base_dir)
preprocess_validation(validation_base_dir)

class CustomDataset(Dataset):
    def __init__(self, clean_image_paths, noisy_image_paths, transform=None):
        self.clean_image_paths = [os.path.join(clean_image_paths, x) for x in os.listdir(clean_image_paths)]
        self.noisy_image_paths = [os.path.join(noisy_image_paths, x) for x in os.listdir(noisy_image_paths)]
        self.transform = transform
        self.center_crop = CenterCrop(1080)
        self.resize = Resize((CFG['IMG_SIZE'], CFG['IMG_SIZE']))

        # Create a list of (noisy, clean) pairs
        self.noisy_clean_pairs = self._create_noisy_clean_pairs()

    def _create_noisy_clean_pairs(self):
        clean_to_noisy = {}
        for clean_path in self.clean_image_paths:
            clean_id = '_'.join(os.path.basename(clean_path).split('_')[:-1])
            clean_to_noisy[clean_id] = clean_path

        noisy_clean_pairs = []
        for noisy_path in self.noisy_image_paths:
            noisy_id = '_'.join(os.path.basename(noisy_path).split('_')[:-1])
            if noisy_id in clean_to_noisy:
                clean_path = clean_to_noisy[noisy_id]
                noisy_clean_pairs.append((noisy_path, clean_path))
            else:
                pass

        return noisy_clean_pairs

    def __len__(self):
        return len(self.noisy_clean_pairs)

    def __getitem__(self, index):
        noisy_image_path, clean_image_path = self.noisy_clean_pairs[index]

        noisy_image = Image.open(noisy_image_path).convert("RGB")
        clean_image = Image.open(clean_image_path).convert("RGB")

        # Central Crop and Resize
        noisy_image = self.center_crop(noisy_image)
        clean_image = self.center_crop(clean_image)
        noisy_image = self.resize(noisy_image)
        clean_image = self.resize(clean_image)

        if self.transform:
            noisy_image = self.transform(noisy_image)
            clean_image = self.transform(clean_image)

        return noisy_image, clean_image
