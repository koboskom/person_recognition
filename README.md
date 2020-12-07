### Required installations
### Downoload yolov4 weights and paste in data directory
https://drive.google.com/file/d/19eICWXLr0zc4ea4vm1kx9Nyk64e42B00/view?usp=sharing
### Conda (Recommended) also using GPU is highly recommended to ensure best fps ratio

```bash
# Tensorflow CPU
conda env create -f conda-cpu.yml
conda activate yolov4-cpu

# Tensorflow GPU
conda env create -f conda-gpu.yml
conda activate yolov4-gpu
```

### Pip
(TensorFlow 2 packages require a pip version >19.0.)
```bash
# TensorFlow CPU
pip install -r requirements.txt

# TensorFlow GPU
pip install -r requirements-gpu.txt
```

### Nvidia Driver (For GPU, if you are not using Conda Environment and haven't set up CUDA yet)
Make sure to use CUDA Toolkit version 10.1 as it is the proper version for the TensorFlow version used in this repository.
https://developer.nvidia.com/cuda-10.1-download-archive-update2

### to run program after conda installation, in yolov4-gpu mode type
```bash
python guii.py
```
After that you will see tkinker window app
1. pick a video file to process
2. run the program 
3. open modified video file and logs to see results


### References  

  * [tensorflow-yolov4-tflite](https://github.com/hunglc007/tensorflow-yolov4-tflite)
  * [Deep SORT Repository](https://github.com/nwojke/deep_sort)
  * [The AI Guy] (https://github.com/theAIGuysCode)
