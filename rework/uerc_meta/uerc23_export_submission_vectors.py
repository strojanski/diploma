from uerc23 import *

STORE_FEATURE_VECTORS = True
FEATURES_PATH = os.path.join(RUN_PATH, config.get('OUTPUT_PATHS', 'features_dir'))

if __name__ == "__main__":

    # Folder of the UERC23 dataset:
    images_path =     os.path.join('data', 'sequestered_anonymized')
    annotations_csv = os.path.join('data', 'sequestered_anonymized_annotations.csv')
    data_splits_csv = os.path.join('data', 'sequestered_anonymized_splits.csv')
    
    full_image_csv = os.path.join('runs', 'sequestered_anonymized_image_list.csv')

    uerc23 = UERC23(images_path, annotations_csv, full_image_csv, data_splits_csv=data_splits_csv)
    uerc23.compute_feature_vectors(['0'], 'sequestered_anonymized')