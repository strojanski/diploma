import os, random, itertools, csv, pathlib
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

random.seed(42)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

class UERCDataset(Dataset):
    TEST_RATIO = 0.1 # Out of all data
    VAL_RATIO = 0.2 # Out of train data
    IGNORE_ETHS_IN_COMPARISON = ['other']

    mode = None
    root_dir = None
    annotations = None
    full_image_list = None
    subjects_test = []
    subjects_covariate_test = []
    images = {} # for train, val and test
    classes_mapping = {} # for train, val and test
    num_of_images = {} # for train, val, test
    num_of_classes = {} # for train, val and test
    min_num_of_classes = {} # for train, val and test
    transforms = {}

    # mode can be: train, val, test
    def __init__(self, mode, root_dir, annotations_csv, full_image_list_csv, data_splits_csv=None):
        self.mode = mode
        self.images['train'] = []
        self.images['val'] = []
        self.images['test'] = []
        self.root_dir = root_dir
        with open(annotations_csv, "r") as file:
            reader = csv.reader(file, delimiter='\t')
            next(reader)  # skip header row
            self.annotations = [row for row in reader]

        # full_image_list is a list of all image in the dataset and used only to speed up computations. If it is not present it will be created:
        self.full_image_list = self.get_full_image_list(full_image_list_csv)

        # We need to split the data into train, validation and test sets. If the data splits are not present in the CSV file we will make splits and create the CSV:
        self.get_data_splits(data_splits_csv)
        
        self.transforms['train'] = transforms.Compose([
                                transforms.Resize(256),
                                transforms.RandomRotation(30),
                                transforms.CenterCrop(224),
                                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                                transforms.ToTensor(),
                                # transforms.RandomHorizontalFlip(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.transforms['test'] = transforms.Compose([
                                transforms.Resize(224),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.transforms['val'] = self.transforms['test']

    def get_full_image_list(self, full_image_list_csv):
        full_image_list = {}
       
        # If file exists just read it and return the list of all images:
        if os.path.exists(full_image_list_csv):
            with open(full_image_list_csv, "r") as file:
                reader = csv.reader(file)
                prev_subj = None
                tmp_subj_data = None
                for row in reader:
                    d = row[0].split('/')
                    subject = d[0]
                    filename = d[1]

                    if prev_subj != subject:
                        if prev_subj is not None:
                            full_image_list[prev_subj] = tmp_subj_data
                        tmp_subj_data = []
                    
                    tmp_subj_data.append(filename)
                    prev_subj = subject
                    
                full_image_list[prev_subj] = tmp_subj_data
                    
        
        # If file does not exist, create it:
        else:
            img_list = sorted(pathlib.Path(self.root_dir).glob("*/*.*"))
            with open(full_image_list_csv, "w", newline="") as csv_file:
                cw = csv.writer(csv_file, delimiter="\t")
                prev_subj = None
                tmp_subj_data = None
                for image_path in img_list:
                    if image_path.suffix.lower() in ['.png', '.jpg']:
                        img_pth = str(image_path)
                        subject = os.path.basename(os.path.dirname(img_pth))
                        filename = image_path.name
                        full = os.path.join(subject, filename)

                        cw.writerow([full])

                        if prev_subj != subject:
                            if prev_subj is not None:
                                full_image_list[prev_subj] = tmp_subj_data
                            tmp_subj_data = []
                        
                        tmp_subj_data.append(filename)

                        prev_subj = subject
                full_image_list[prev_subj] = tmp_subj_data

        return full_image_list

    def get_data_splits(self, data_splits_csv):
        if data_splits_csv is None or not os.path.exists(data_splits_csv):
            # Select test data. Since split is done subject wise, we only need subject list
            self.subjects_test, self.subjects_covariate_test = self.select_test_data()

            # Select train and validation data. Since the split between them is done within each subject we need to get full list of images and not only list of subjects:
            self.images['train'], self.images['val'] = self.select_train_val_data()

            self.images['test'] = []
            for subj in self.subjects_test:
                for el in self.full_image_list[subj]:
                    self.images['test'].append(os.path.join(subj, el))

            # Now we can save the data splits to a CSV file:
            if data_splits_csv is not None:
                self.save_data_splits_to_csv(data_splits_csv)
        else:
            # Splits already exist, just read them from the CSV file:
            self.images['train'], self.images['val'], self.images['test'], self.subjects_test = self.get_data_splits_from_csv(data_splits_csv)

            # However, we need to prepare covariates dict:
            subjects = self.annotations
            combinations = set(itertools.product(set([x[1] for x in subjects]), set([x[2] for x in subjects if x[2] not in self.IGNORE_ETHS_IN_COMPARISON])))
            combinations = sorted(list(combinations), key=lambda x: (x[0], x[1]))
            
            selected_items = {}
            for combination in combinations:
                selected_items[combination] = []
            selected_items[('all', 'all')] = []
            for subject in self.subjects_test:
                # r = self.annotations[int(subject)]
                r = None
                for ann in self.annotations:
                    if ann[0] == subject:
                        r = ann
                        break
                
                if r[2] not in self.IGNORE_ETHS_IN_COMPARISON:
                    selected_items[(r[1], r[2])].append(subject)
                    selected_items[('all', 'all')].append(subject)

            self.subjects_covariate_test = selected_items

        self.num_of_images = {}
        self.num_of_images['train'] = len(self.images['train'])
        self.num_of_images['val'] = len(self.images['val'])
        self.num_of_images['test'] = len(self.images['test'])

        cla_train_list = [item.split('/')[0] for item in self.images['train']]
        cla_test_list = [item.split('/')[0] for item in self.images['test']]
        self.num_of_classes = {}
        self.num_of_classes['train'] = len(set(cla_train_list))
        self.num_of_classes['val'] = self.num_of_classes['train']
        self.num_of_classes['test'] = len(set(cla_test_list))

        self.min_num_of_classes = {}
        self.min_num_of_classes['train'] = min(cla_train_list)
        self.min_num_of_classes['val'] = self.min_num_of_classes['train']
        self.min_num_of_classes['test'] = min(cla_test_list)

        self.classes_mapping = {}
        unique_cla_train = list(set(cla_train_list))
        self.classes_mapping['train'] = {elem: i for i, elem in enumerate(unique_cla_train)}
        self.classes_mapping['val'] = self.classes_mapping['train']
        unique_cla_train = list(set(cla_test_list))
        self.classes_mapping['test'] = {elem: i for i, elem in enumerate(unique_cla_train)}

            

    def select_test_data(self):
        # This function selects test samples. Crucial parts are:
        #   - we need to make split subject-wise (subjects present, that are not in train or validation set), i.e. open-set evaluation
        #   - all demographics should be represented in the test set
    
        subjects = self.annotations

        # calculate the number of items to select
        num_items_to_select = round(len(subjects) * self.TEST_RATIO)

        # use itertools.product() to generate all combinations of the specific columns
        combinations = set(itertools.product(set([x[1] for x in subjects]), set([x[2] for x in subjects if x[2] not in self.IGNORE_ETHS_IN_COMPARISON])))
        combinations = sorted(list(combinations), key=lambda x: (x[0], x[1]))
        # combinations.append(['all', 'all'])
        combinations_cp = combinations.copy()

        # randomly select items while maintaining all combinations
        selected_items = {}
        sel_it_count = 0
        for combination in combinations:
            selected_items[combination] = []
        selected_items[('all', 'all')] = []

        while sel_it_count < num_items_to_select:
            combination = combinations.pop()
            items_with_combination = [x for x in subjects if x[1] == combination[0] and x[2] == combination[1] and x[0] not in selected_items[combination]]
            if items_with_combination:
                el = random.choice(items_with_combination)[0]
                selected_items[combination].append(el)
                selected_items[('all', 'all')].append(el)
                sel_it_count += 1
            if not combinations:
                combinations = combinations_cp.copy()
    

        return selected_items[('all', 'all')], selected_items
    
    def select_train_val_data(self):
        # Randomly split between train and validation set.
        # However, we need to ignore subjects from the test set.
        # Split here is done within each subject (so that identities are shared), i.e. closed-set evaluation.

        subjects = self.annotations

        # Split images into train and validation sets, but only those that are not present in items_for_test:

        # For each subject in the dataset, we need to split randomly into train and validation sets (but only those subjects that are not present in items_for_test)
        # We need to make sure that all subjects are present in the train and validation sets.
        items_train = []
        items_val = []
        subjects = [x for x in self.full_image_list.keys() if x not in self.subjects_test]
        for subject in subjects:
            # For each subject get all images:
            images = self.full_image_list[subject]

            # Randomly split images into train and validation sets:
            random.shuffle(images)
            num_items_to_select = round(len(images) * self.VAL_RATIO)
            if num_items_to_select < 1:
                num_items_to_select = 1
            items_for_val = [os.path.join(subject, x) for x in images[:num_items_to_select]]
            items_for_train = [os.path.join(subject, x) for x in images[num_items_to_select:]]
        
            # Append to the list of items for train and validation sets:
            items_val.extend(items_for_val)
            items_train.extend(items_for_train)

        return items_train, items_val
    
    def save_data_splits_to_csv(self, data_splits_csv):
        # First read all the images from the dataset and then according to the presence in the provided lists, assign them to train, validation or test sets:
        # for image in sorted(os.listdir(root_dir)):
        
        data = []
        for el in self.images['train']:
            data.append([el, 'train'])
        for el in self.images['val']:
            data.append([el, 'val'])
        for subj in self.subjects_test:
            for el in self.full_image_list[subj]:
                data.append([os.path.join(subj, el), 'test'])
        data = sorted(data)
        
        with open(data_splits_csv, "w", newline="") as csv_file: 
            cw = csv.writer(csv_file, delimiter="\t")
            for d in data:
                cw.writerow(d)
                    
    def get_data_splits_from_csv(self, data_splits_csv):
        # read CSV file
        subjects_test = []
        images_train = []
        images_val = []
        images_test = []
        with open(data_splits_csv, "r") as file:
            reader = csv.reader(file, delimiter='\t')
            # next(reader)  # skip header row
            for row in reader:
                if row[1] == 'train':
                    images_train.append(row[0])
                elif row[1] == 'val':
                    images_val.append(row[0])
                elif row[1] == 'test':
                    images_test.append(row[0])
                    subj = row[0].split('/')[0]
                    if subj not in subjects_test:
                        subjects_test.append(subj)
                else:
                    raise ValueError('Unknown data split: {}'.format(row[1]))
        return images_train, images_val, images_test, subjects_test
            

    def __len__(self):
        return len(self.images[self.mode])

    def __getitem__(self, idx):

        img_name = self.images[self.mode][idx]
        transform = self.transforms[self.mode]

        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        label_def = os.path.dirname(img_name)
        label = self.classes_mapping[self.mode][label_def]

        if transform:
            image = transform(image)

        return image, label


if __name__ == "__main__":


    # public_image_list.csv is there only to speed up operations. If it does not exist, it will be created.
    # data_splits_csv is optional, you can make your own splits. Also, if it is provided here but does not exist it will be created. If it exists, it will be used.
    uercdataset = UERCDataset('test', 'public', 'public_annotations.csv', '../runs/public_image_list.csv', data_splits_csv='../runs/train_val_test_splits.csv')
