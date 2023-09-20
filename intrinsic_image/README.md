The following code would help you to setup the environment. We use conda to create an environment and install correponding packages. 


```bash
conda create -n intrinsic python=3.6 -y
conda activate intrinsic
pip install scikit-image
pip install pypng
pip install pyamg
pip install matplotlib
```

Run `python3.6 comparison.py`. This will produce `results` folder which contains the seperation of intrinsic images.