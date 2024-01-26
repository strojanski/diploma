import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def get_image_resolution(image_path):
    with Image.open(image_path) as img:
        width, height = img.size
    return width, height

counts = {}
sizes_x, sizes_y = [], []

try:
    cnt = 0
    for f in os.listdir("UERC/"):
        counts[f] = 0
        for ff in os.listdir(f"UERC/{f}"):
            cnt+=1
            counts[f] += 1
            # w, h = get_image_resolution(f"UERC/{f}/{ff}")
            # sizes_x.append(w)
            # sizes_y.append(h)
            
            # if cnt % 100 == 0:
            #     print(cnt, w, h)
except KeyboardInterrupt:
    np.save("sizex.npy", sizes_x)
    np.save("sizey.npy", sizes_y)    
        
# print(counts)

# all_cts = list(counts.values())
# print(min(all_cts), max(all_cts), np.mean(all_cts), sum(all_cts))

# print(sizes_x, sizes_y)
# print(np.mean(sizes_x), np.mean(sizes_y))
# # Tilt labels by 90 degrees
# # plt.bar(counts.keys(), counts.values(), width=.5, color='b')
# plt.plot(counts.values())
# plt.show()

# # Get random guess accuracy
# n_imgs = sum(counts.values())
# n_classes = len(counts.keys())

# max_prob = max(counts.values())
# print(max_prob)
# print(max_prob/n_imgs)

# baselines = []

# for k, v in counts.items():
#     baselines.append(v/n_imgs)
    
    
# probability = 0
# for b in baselines:
#     probability += b * (1/n_classes)
# print(probability)

n_imgs = sum(counts.values())
bs = 256
n_batches = n_imgs // bs
a = np.array(np.loadtxt("loss/loss_history_googlenet_1_256.txt"))

offset = 0

losses = [sum(a[offset:offset+n_batches]) for offset in range(0, len(a), n_batches)]
print(losses)
print(n_imgs)

plt.plot(losses) 
plt.show()
