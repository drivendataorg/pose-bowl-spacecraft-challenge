# Solution -  T5 decoder-based deep learning

Username: dylanliu

## Summary

This solution contains 3 parts: a data processing part, a training part and an inference part. The data processing part processes image data into inter-image rotation features, the training part trains many T5 decoder models, and the inference part uses the trained models to do prediction on new image data.

# Setup

First of all, my solution was run on Linux, I'm not sure if it's ok on other operating systems.

1. Install Python 3.10.13 and Pytroch 2.1.2+CUDA if you don't have them on your machine. The solution was originally run on Python 3.10.13 and pytroch 2.1.2, but later versions may still be ok.  
How to install Python: https://docs.python.org/3/using/unix.html#on-linux  
How to install Pytroch: https://pytorch.org/get-started/previous-versions/  

2. Install the required Python packages:
```
pip install -r requirements.txt
```

# Run training

Before starting training, you first need to process the raw data by the command line: 
`python make_dataset.py path_to/train_data`  
Where 'path_to/data' is the training data path. After processing, the processed data can be found under the folder 'data_cache'.  
  
  
Then run training by the command line:   
`python run_training.py`  
After training, the trained models can be found under the folder 'assets\models'.  


# Run inference

Tun inference by the command line: 
`python run_inference.py path_to/test_data submission.csv`  
Where 'path_to/test_data' is the inference data path, and 'submission.csv' is the path and filename of submission output.  
