# Literatura

- [Overview](https://ieeexplore.ieee.org/abstract/document/9040673?casa_token=YlNN2WodQzMAAAAA:pKwX_hu-PEJF9QNEzM79EVxAuKSzSWJiH_QIdi_b75wtEwAH0fKM5k3IL_jTUDptEm5Ycm0zo76c)
  - OSR - open set recognition - instead of falsely classifying element into one of the known classes, recognize it as an unknown class 
  - KKCs, KUCs, UKCs, UUCs
  - Openness: 0 - closed set problem, the higher it is, the more open set problem it is
  - Discriminative vs generative models
  - Various model description
  - DNN - replace Softmax layer with an Openmax (or G-Openmax) layer
  - Open world recognition
  - Experiments and evaluation

- [OpenMax + GANs](https://arxiv.org/abs/1707.07418)
  - Proposes G-OpenMax
  - Idea: Generate UUC with a GAN and include it in training
  - Drastic improvements

- [Pairwise filters](https://www.sciencedirect.com/science/article/abs/pii/S016786551500327X)
  - Iris recognition on pictures taken in uncontrolled environment (distance, lighting, etc.)
  - CNN with pairs of images as input 
  - Pairwise filter layer - takes pairs as input and convolves them to generate similarity map

- [Simple nearest neighbor](http://ears.fri.uni-lj.si/papers/uerc19arxiv_with_appendix.pdf)
  - Ear recognition in uncontrolled environments, existing methods, data acquisition
  - Challenge of best ear recognition
  - Ensemble methods, CNNs, 30 million - 2 billion parameters
  - yaw, pitch, occlusion invariance 

- [Open set face recognition](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w6/papers/Gunther_Toward_Open-Set_Face_CVPR_2017_paper.pdf) 110. M. Günther, S. Cruz, E. M. Rudd and T. E. Boult, "Toward open-set face recognition", Proc. IEEE Conf. Comput. Vis. Pattern Recognit. Workshops, pp. 573-582, 2017.
  - verification vs recognition
  - 3 approaches on VGG face network deep features
    - cosine similarities between deep features to reject unknowns - works on closed sets
    - LDA on features - detects KUCs but not UUCs
    - train EVM on cosine distances using KUCs - works the same on closed and open sets 

- [Towards Open Set Deep Networks](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Bendale_Towards_Open_Set_CVPR_2016_paper.pdf)
  - NNs easy to fool
  - Add OpenMax to DNN
  - Openmax = softmax + ability to predict an unknown class
  - Multi class meta recognition with activation vectors
  - closed set, open set, fooling set and adversarial set

- [Learning a Neural-network-based Representation for Open Set Recognition](https://epubs.siam.org/doi/abs/10.1137/1.9781611976236.18)
  - Proposes a new cost function and an approach for learning a representation that facilitates osr
  - ii-loss (intra spread - inter separation)

- [Towards open-set face recognition using hashing functions](https://ieeexplore.ieee.org/abstract/document/8272751)
