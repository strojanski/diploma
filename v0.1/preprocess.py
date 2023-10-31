import copy
import numpy as np
from PIL import Image
from torchvision import transforms


def resize_input(input_data: dict, tgt_size=224) -> dict:
    '''
        Input data: dictionary of images per person
        Output data: dictionary of images per person, resized to 2 tgt_size x 2 tgt_size
    '''
    
    
    resized_data = copy.deepcopy(input_data)
    for key in input_data.keys():   # For each person
        for i in range(len(input_data[key])):   # For each image
            # Resize into 224x448
            resized_data[key][i] = np.array(Image.fromarray(resized_data[key][i]).resize((int(.5*tgt_size), tgt_size)))
            
            # Zero pad to make it square
            pad = np.abs(resized_data[key][i].shape[1] - resized_data[key][i].shape[0]) // 2
            resized_data[key][i] = np.pad(resized_data[key][i], ((0, 0), (pad, pad), (0, 0)), 'constant')    
            # resized_data[key][i] = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(resized_data[key][i])
            #resized_data[key][i] = cv2.cvtColor(resized_data[key][i], cv2.COLOR_RGB2GRAY)
            
            preprocess = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            resized_data[key][i] = preprocess(resized_data[key][i])
            
            
    return resized_data


if __name__ == '__main__':
    pass
