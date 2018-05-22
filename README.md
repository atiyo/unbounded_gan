# Unbounded Discriminators in GANs

Discriminators in Generative Adversarial Networks (GANs) are typically taken to be
classifiers with sigmoid/softmax outputs. These outputs might easily be
saturated and hinder the ability of generator networks to learn.

What if we remove these activations? Apart from providing stronger gradients to
the generator network, this might also open the opportunity to experiment with
novel loss functions (see below).

At the very least, this should be an easy thing to
try, since it only involves deleting a single line from the discriminator.


## L1 GAN

As an example, consider an L1 analog of the [Least Squares GAN (Mao et al.,
2016)](https://arxiv.org/abs/1611.04076). In particular (and with abuse of
notation), train the discriminator to minimise `abs(D(data) - 1) +
abs(D(G(z)))`, and train the generator to minimise `abs(D(G(z)) - 1)`.


### Discriminator with sigmoid output

In the case where the discriminator network is constrained to lie in the
interval (0,1), the derivatives of the L1 loss functions are the same as
derivatives of the resulting loss functions from the [Wasserstein GAN (Arjovsky
et al., 2017)](https://arxiv.org/abs/1701.07875). The Wasserstein GAN framework
relies on the discriminator having Lipshitz constraints. We have not imposed
any such constraints. This provides some heuristic basis for believing that the
L1 loss functions above with a sigmoid output should fail to produce any decent
results. Indeed, this is the case with numeric experiments.  After 100 epochs
of training on Fashion MNIST, we have the following:

![sigmoid_output](l1_output/output_99.jpg "Sigmoid
output from discriminator")

### Discriminator with unbounded output

Simply removing the sigmoid activation leads to much better results. Note that
comparisons to the Wasserstein GAN are less appropriate with the expanded
codomain of the discriminator. For comparison with above, here are the results
after 100 epochs of training with an unbounded discriminator:

![sigmoid_output](l1_unbounded_output/output_99.jpg "Unbounded output
from discriminator")


## To reproduce results

Scripts are written in Python3 with PyTorch 0.3.1 (built from source) and
torchvision 0.2.0. Code for generating the sigmoidal output comes from running
`l1_gan.py`. The unbounded gan results come from running `l1_gan_unbounded.py`.
