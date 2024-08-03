import torch
import pickle
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt

ngpu = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
while True:
  try:
    real_batch_num = input("Enter real batch num: ")
    img_list_num = input("Enter img list num: ")

    with open(f'data/output/real_batch_{real_batch_num}.pickle', 'rb') as f:
      real_batch = pickle.load(f)
    with open(f'data/output/img_list_{img_list_num}.pickle', 'rb') as img:
      img_list = pickle.load(img)
    
    
    plt.figure(figsize=(15,15))
    plt.subplot(1,2,1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(
      vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),
      (1,2,0))
    )
    
    # Plot the fake images from the last epoch
    plt.subplot(1,2,2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list,(1,2,0)))
    plt.show()

  except KeyboardInterrupt:
    print('exiting')
    break
