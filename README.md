# adversarial-patch
PyTorch implementation of adversarial patch 

This is an implementation of the <a href="https://arxiv.org/pdf/1712.09665.pdf">Adversarial Patch paper</a>. Not official and likely to have bugs/errors.

## How to run:

Data set-up:

 - Follow instructions https://github.com/amd/OpenCL-caffe/wiki/Instructions-to-create-ImageNet-2012-data . The validation set should be in path `./imagenet/val/`. There should be 1000 directories, each with 50 images.
 
Run attack:

- `python make_patch.py --cuda --netClassifier inceptionv3 --max_count 500 --image_size 299 --patch_type circle --outf log`

## Results:

Using patch shapes of both circles and squares gave good results (both achieved 100% success on the training set and eventually > 90% success on test set)

I managed to recreate the toaster example in the original paper. It looks slightly different but it is evidently a toaster.

![Alt text](1981_859_adversarial.png?raw=true "") This is a toaster

Square patches are a little more homogenous due to that I only rotate by multiples of 90 degrees.

![Alt text](1978_859_adversarial.png?raw=true "") This is also a toaster

## Issues:

- Cannot make a perfect circle with numpy/pytorch. The hack I came up with makes the boundary slightly hexagonal.

- Rather slow if max_count and conf_target are large.

- Probably lots of redundant calls and variables.


