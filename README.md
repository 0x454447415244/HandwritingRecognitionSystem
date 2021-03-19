# Handwriting Recognition System

This repository is the Tensorflow implementation of the Handwriting Recognition System described in [Handwriting Recognition of Historical Documents with Few Labeled Data](https://www.researchgate.net/publication/325993975_Handwriting_Recognition_of_Historical_Documents_with_Few_Labeled_Data) (please cite the paper if you use this code in your research paper). This code was also used for the baseline system in [Fine-tuning Handwriting Recognition systems with Temporal Dropout](https://www.researchgate.net/publication/348958179_Fine-tuning_Handwriting_Recognition_systems_with_Temporal_Dropout).

This code is free for academic and research use. For commercial use of the code please contact [Edgard Chammas](mailto:contact@edgard.net).

To help run the system, sample images from [ICDAR2017 Competition on Handwritten Text Recognition on the READ Dataset](https://scriptnet.iit.demokritos.gr/competitions/8/) are added.

<img src="https://github.com/0x454447415244/HandwritingRecognitionSystem/raw/master/image.jpg" width="30%">

## Configuration
General configuration can be found in config.py

CNN-specific architecture configuration can be found in cnn.py

## Training
```
python train.py
```
This will generate a text log file and a Tensorflow summary.

<img src="https://github.com/0x454447415244/HandwritingRecognitionSystem/blob/master/TensorBoard.png" width="100%">

## Decoding
```
python test.py
```
This will generate, for each image, the line transcription. The output will be written to decoded.txt by default.

```
python compute_probs.py
```
This will generate, for each image, the posterior probabilities at each timestep. Files will be stored in Probs by default.

## Dependencies
- Tensorflow
- OpenCV-Python

## Citation
Please cite the following paper if you use this code in your research paper:
```
@inproceedings{chammas2018handwriting,
  title={Handwriting Recognition of Historical Documents with few labeled data},
  author={Chammas, Edgard and Mokbel, Chafic and Likforman-Sulem, Laurence},
  booktitle={2018 13th IAPR International Workshop on Document Analysis Systems (DAS)},
  pages={43--48},
  year={2018},
  organization={IEEE}
}
```

## Acknowledgment
We gratefully acknowledge the support of NVIDIA Corporation with the donation of the Titan Xp GPU used for this research.

<img src="https://upload.wikimedia.org/wikipedia/sco/thumb/2/21/Nvidia_logo.svg/1280px-Nvidia_logo.svg.png" width="20%">

## Contributions
Feel free to send your pull request or open issues.

