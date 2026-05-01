---
layout: post
title: "Sudoku as a Vision Problem"
---
## Introduction
Sudoku is a game consisting of a 9x9 grid, partially filled with numbers between
1-9. The objective of the game is to fill each cell with a number such that:
- Each row contains unique numbers
- Each column consists of unique numbers
- Each non overlapping 3x3 subgrid consists of unique numbers

For computers to solve sudoku, the most basic method is backtracking. While that
guarantees to find a solution if it exists, it can be used only as a fallback
for stochastic or constraint satisfication methods to speed up solving\[[1](#ref1)\].

On the deep learning side of things, Several approaches have been used ranging 
from CNNs, LSTMs etc. GNNs also find use here by treating sudoku as a graph 
colouring problem. LLMs have been tried as well but do not seem to perform 
well. Tiny Recursive Model(TRM)\[[2](#ref2)\] showed that a 7M parameter model 
can get 87.4% on Sudoku-Extreme\[[3](#ref3)\].

I was reading "ARC Is a Vision Problem!"\[[4](#ref4)\] and decided to 
try something similar for sudoku by treating it as a per-pixel classification 
problem. As I show below, It is possible to solve sudoku in one pass but with
a low accuracy at the scale used. Perhaps scaling up(10M params) and training
for more steps would improve performance.

Most of my architecture is inspired from this work\[[5](#ref5)\]. The method
uses only convolutions to get 94.69% cell accuracy and solved 95/100 games
tested on. While the results are impressive, it does list down improvements
that can be made, such as:
- Test on puzzles with fewer clues.
- Evaluate on expert-level Sudoku.
- Investigate ResNet-style skip connections.
- Explore attention mechanisms (spatial, not sequential).

The first 2 points can be addressed by using the sudoku-extreme dataset, and
last 2 by using Vision Transformers\[[6](#ref6)\]. I believe I can make few
more improvements as well.

The code is available at this link \[[7](#ref7)\].

## Architecture

<img src="{{ "/assets/sudoku-as-a-vision-problem/SudokuVIT.svg" | relative_url }}" alt="Vision Transformer for Sudoku" class="centre-image"><br/>

- Grid is flattened and each digit is passed through an embedding layer.
- Learned positional embeddings of each cell's row, column and subgrid postions
are added.
- This is now passed through the encoder\[[8](#ref8)\] layers.
- Transformer output is reshaped back to a grid
- A residual block\[[9](#ref9)\] of 1×1 convolutions is applied to add some non-linearity
to the pixel-wise classification part.
- One final 1×1 convolution to map each cell to the distribution of its solved
counterpart.

The attnetion in the transformer is also modified to add an inductive bias
(such as in \[[10](#ref10)\]) specific to sudoku i.e. higher priority to cells in the same row, column and
subgrid.

Standard Attention uses:

$$ \large A = softmax(\frac{QK^{T}} {\sqrt{d_k}}) $$

Relational Attention uses:

$$ \large A = softmax(\frac{QK^{T}} {\sqrt{d_k}} + B) $$

where:

$$ \large B_h = \sum_{r=1}^{4} \alpha_{h,r} \cdot M_r $$

$\alpha_{h,r}$ are learnable scalars for each combination of attention head
and relational mask, initialised to 0.

$M_r$ are relational masks: 4, 81×81 matrices which are set to 0 except when
pair of cells are in the:
- Same row
- Same column
- Same subgrid
- Same position(self)

in which case those cells are set to 1.
## Training Details
I train for 200k steps with batch size 256. The peak learning rate is
4×10<sup>-4</sup> with a linear warmup for the first 5% of steps followed
by cosine decay. AdamW\[[11](#ref11)\] is used as the optimizer
β<sub>1</sub>=0.9 and β<sub>2</sub>=0.999 and a weight decay of
5×10<sup>-4</sup>. A validation set is formed by randomly sampling 10% of the
training set. Model is saved upon new best validation loss.

The following augmentations are used to improve model performance:
- Digit relabelling
- Swapping rows within bands
- Swapping columns within stacks
- Swapping bands
- Swapping stacks
- Random transpose

The loss function uses cross entropy loss along with a global constraint loss.
Cross entropy guides the model to make individual cells correct while the
constraint loss guides the model to penalize duplicate values.

Let x denote a cell index (flattening $(i,j)$), $k \in \{1,\dots,9\}$, and
$p_{x,k}$ the softmax probability. Let $y_x$ be the true label.

Define a collection of constraint groups $\mathcal{G}$, where each group
$g \in \mathcal{G}$ is a set of 9 cells (all rows, columns, and boxes). Then:

$$ \mathcal{L}_{\mathrm{CE}} = \mathbb{E}_{x}\!\left[-\log p_{x,y_x}\right] $$

$$ \mathcal{L}_{\mathrm{constraint}} = \mathbb{E}_{g \in \mathcal{G}} \; \mathbb{E}_{k}\!\left[\left(\sum_{x \in g} p_{x,k} - 1\right)^2\right] $$

$$ \mathcal{L} = \mathcal{L}_{\mathrm{CE}} + \lambda \, \mathcal{L}_{\mathrm{constraint}} $$

## Results

<table class="standard-table">
  <thead>
    <tr>
	  <th class="standard-table-heading">Test CE Loss</th>
	  <th class="standard-table-heading">Test Constraint Loss</th>
	  <th class="standard-table-heading">Test Fully Solved Accuracy</th>
	  <th class="standard-table-heading">Test Correct Cells Accuracy</th>
    </tr>
  </thead>
  <tr>
	  <th class="standard-table-entry">0.3979</th>
	  <th class="standard-table-entry">0.0033</th>
	  <th class="standard-table-entry">23.55%</th>
	  <th class="standard-table-entry">82.68%</th>
  </tr>
</table>

The model shows okay performance overall being able to solve about a quarter of
the puzzles in just one pass. Scaling up the model, training for more steps,
predicting only empty cells and maybe using a KL Divergance based constraint
loss should improve performance. The key point is that it shows solving sudoku
in one pass is possible.

## Ablations <span id="ablations"></span>

<table class="standard-table">
  <thead>
    <tr>
      <th class="standard-table-heading">Model</th>
	  <th class="standard-table-heading">Parameter Count</th>
	  <th class="standard-table-heading">FLOPs</th>
	  <th class="standard-table-heading">Train Steps</th>
	  <th class="standard-table-heading">Validation Loss</th>
	  <th class="standard-table-heading">Validation Fully Solved Accuracy</th>
	  <th class="standard-table-heading">Validation Correct Cells Accuracy</th>
    </tr>
  </thead>
  <tr>
      <th class="standard-table-entry">Baseline</th>
	  <th class="standard-table-entry">1.63M</th>
	  <th class="standard-table-entry">293M</th>
	  <th class="standard-table-entry">200k</th>
	  <th class="standard-table-entry">0.4502</th>
	  <th class="standard-table-entry">13.91%</th>
	  <th class="standard-table-entry">79.75%</th>
  </tr>
  <tr>
      <th class="standard-table-entry">d<sub>ff</sub>=384 + SwiGLU</th>
	  <th class="standard-table-entry">1.75M</th>
	  <th class="standard-table-entry">312M</th>
	  <th class="standard-table-entry">200k</th>
	  <th class="standard-table-entry">0.4353</th>
	  <th class="standard-table-entry">14.84%</th>
	  <th class="standard-table-entry">80.48%</th>
  </tr>
  <tr>
      <th class="standard-table-entry">Replace 3D with 1D Positional Embeddings</th>
	  <th class="standard-table-entry">1.76M</th>
	  <th class="standard-table-entry">312M</th>
	  <th class="standard-table-entry">200k</th>
	  <th class="standard-table-entry">0.4768</th>
	  <th class="standard-table-entry">11.42%</th>
	  <th class="standard-table-entry">78.46%</th>
  </tr>
  <tr>
      <th class="standard-table-entry">Bring Back 3D Positional Embeddings + Relative Bias Attention</th>
	  <th class="standard-table-entry">1.75M</th>
	  <th class="standard-table-entry">314M</th>
	  <th class="standard-table-entry">200k</th>
	  <th class="standard-table-entry">0.3844</th>
	  <th class="standard-table-entry">21.77%</th>
	  <th class="standard-table-entry">82.88%</th>
  </tr>
  <tr>
      <th class="standard-table-entry">+ 0.1 Constraint Loss</th>
	  <th class="standard-table-entry">1.75M</th>
	  <th class="standard-table-entry">314M</th>
	  <th class="standard-table-entry">200k</th>
	  <th class="standard-table-entry">0.3858(0.3836)</th>
	  <th class="standard-table-entry">22.11%</th>
	  <th class="standard-table-entry">82.94%</th>
  </tr>
  <tr>
      <th class="standard-table-entry">+ 0.05 Weight Decay</th>
	  <th class="standard-table-entry">1.75M</th>
	  <th class="standard-table-entry">314M</th>
	  <th class="standard-table-entry">200k</th>
	  <th class="standard-table-entry"><b>0.3659(0.3589)</b></th>
	  <th class="standard-table-entry">24.52%</th>
	  <th class="standard-table-entry"><b>83.84%</b></th>
  </tr>
  <tr>
      <th class="standard-table-entry">+ Constraint loss linear schedule to 1</th>
	  <th class="standard-table-entry">1.75M</th>
	  <th class="standard-table-entry">314M</th>
	  <th class="standard-table-entry">200k</th>
	  <th class="standard-table-entry">0.3762(0.3653)</th>
	  <th class="standard-table-entry">25.46%</th>
	  <th class="standard-table-entry">83.76%</th>
  </tr>
  <tr>
      <th class="standard-table-entry">+ Constraint loss linear schedule to 2</th>
	  <th class="standard-table-entry">1.75M</th>
	  <th class="standard-table-entry">314M</th>
	  <th class="standard-table-entry">200k</th>
	  <th class="standard-table-entry">0.3834(0.3686)</th>
	  <th class="standard-table-entry">26.06%</th>
	  <th class="standard-table-entry">83.63%</th>
  </tr>
  <tr>
      <th class="standard-table-entry">+ Constraint loss linear schedule to 4 in 90% and then held</th>
	  <th class="standard-table-entry">1.75M</th>
	  <th class="standard-table-entry">314M</th>
	  <th class="standard-table-entry">200k</th>
	  <th class="standard-table-entry">0.3874(0.3685)</th>
	  <th class="standard-table-entry">27.06%</th>
	  <th class="standard-table-entry">83.72%</th>
  </tr>
  <tr>
      <th class="standard-table-entry">+ Constraint loss linear schedule to 8 in 90% and then held</th>
	  <th class="standard-table-entry">1.75M</th>
	  <th class="standard-table-entry">314M</th>
	  <th class="standard-table-entry">200k</th>
	  <th class="standard-table-entry">0.3977(0.3763)</th>
	  <th class="standard-table-entry"><b>27.24%</b></th>
	  <th class="standard-table-entry">83.51%</th>
  </tr>
</table>

Baseline Parameters:

- Embedding Dimension: 128
- Number of Layers: 8
- Number of Attention Heads: 4
- Feed Forward Dimension: 512
- Activation Function: GELU

## The Importance of Global Receptive Fields
Being able to "see" the entire board at once seems to play a key role at how
well a model can solve the puzzles. To show this, I train 2
ResNet\[[9](#ref9)\] like models, which have local receptive fields and 2
MLP-Mixers\[[12](#ref12)\], which can view all cells at once. Both models follow
a similar architecture as the transformer but the encoder layers are replaced
with the following types of layers:

- Pre-norm\[[13](#ref13)\] 3x3 residual block with LayerNorm.

<img src="{{ "/assets/sudoku-as-a-vision-problem/StandardResnet.svg" | relative_url }}" alt="Residual Block Architecture" class="centre-image"><br/>

- Similar Resnet as above but with 3x3, 1x9 and 9x1 convolutions all applied in parallel, added up then passed
through a 1x1 convolution.

<img src="{{ "/assets/sudoku-as-a-vision-problem/AnisotropicSudokuResnet.svg" | relative_url }}" alt="Multi-Kernel Residual Block Architecture" class="centre-image"><br/>

The intuition behind the this setup is to hard-code the inductive bias for sudoku
as in:
- 3x3 convolution for the subgrids.
- 1x9 convolution for rows.
- 9x1 convolution for columns.

- Standard MLP-Mixer
<img src="{{ "/assets/sudoku-as-a-vision-problem/MLPMixer.svg" | relative_url }}" alt="MLP-MixerResidual Block Architecture" class="centre-image"><br/>

These are the results.

<table class="standard-table">
  <thead>
    <tr>
      <th class="standard-table-heading">Model</th>
	  <th class="standard-table-heading">Parameter Count</th>
	  <th class="standard-table-heading">FLOPs</th>
	  <th class="standard-table-heading">Train Steps</th>
	  <th class="standard-table-heading">Validation Loss</th>
	  <th class="standard-table-heading">Validation Fully Solved Accuracy</th>
	  <th class="standard-table-heading">Validation Correct Cells Accuracy</th>
    </tr>
  </thead>
  <tr>
      <th class="standard-table-entry">Standard Resnet</th>
	  <th class="standard-table-entry">9.6M</th>
	  <th class="standard-table-entry">1.33G</th>
	  <th class="standard-table-entry">200k</th>
	  <th class="standard-table-entry">0.6075</th>
	  <th class="standard-table-entry">2.06%</th>
	  <th class="standard-table-entry">71.77%</th>
  </tr>
  <tr>
      <th class="standard-table-entry">Multi-resolution Kernels Resnet</th>
	  <th class="standard-table-entry">26.54M</th>
	  <th class="standard-table-entry">198M</th>
	  <th class="standard-table-entry">100k</th>
	  <th class="standard-table-entry">1.5133</th>
	  <th class="standard-table-entry">0%</th>
	  <th class="standard-table-entry">38.78%</th>
  </tr>
  <tr>
      <th class="standard-table-entry"><b>MLP Mixer</b></th>
	  <th class="standard-table-entry"><b>1.59M</b></th>
	  <th class="standard-table-entry"><b>310M</b></th>
	  <th class="standard-table-entry"><b>200k</b></th>
	  <th class="standard-table-entry"><b>0.4507</b></th>
	  <th class="standard-table-entry"><b>11.12%</b></th>
	  <th class="standard-table-entry"><b>79.78%</b></th>
  </tr>
  <tr>
      <th class="standard-table-entry">MLP Mixer + 3D Positional Embeddings</th>
	  <th class="standard-table-entry">1.6M</th>
	  <th class="standard-table-entry">311M</th>
	  <th class="standard-table-entry">200k</th>
	  <th class="standard-table-entry">0.4694</th>
	  <th class="standard-table-entry">8.32%</th>
	  <th class="standard-table-entry">78.88%</th>
  </tr>
</table>

The second model was supposed to train for 200k but the training ended early as
loss did not improve beyond 18k steps.

Despite both the Resnets being larger and more compute-intensive than any of
the models in the [Ablations](#ablations) section and the MLP-Mixers,
they are outperformed by them. While convolutions are a local operation, 4
3x3 convolutions are enough to cover the entire 9x9 board and the models had 12
more 3x3 convolutions to further refine their representations. The fact that
convolutions struggle at solving sudoku suggests that having a global receptive
field is essential.

## References
[1] <span id="ref1"></span>[Sudoku solving algorithms](https://en.wikipedia.org/wiki/Sudoku_solving_algorithms){:target="_blank" rel="noopener"}

[2] <span id="ref2"></span>[Less is More: Recursive Reasoning with Tiny Networks](https://arxiv.org/abs/2510.04871){:target="_blank" rel="noopener"}

[3] <span id="ref3"></span>[sudoku-extreme](https://ieee-dataport.org/documents/sudoku-extreme){:target="_blank" rel="noopener"}

[4] <span id="ref4"></span>[ARC Is a Vision Problem!](https://arxiv.org/abs/2511.14761){:target="_blank" rel="noopener"}

[5] <span id="ref5"></span>[When Simplicity Outperforms Complexity: A Pure CNN Approach to Sudoku Solving](https://www.researchgate.net/publication/400051402_When_Simplicity_Outperforms_Complexity_A_Pure_CNN_Approach_to_Sudoku_Solving){:target="_blank" rel="noopener"}

[6] <span id="ref6"></span>[An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929){:target="_blank" rel="noopener"}

[7] <span id="ref7"></span>[Vision Transformers for Sudoku](https://www.kaggle.com/code/ashw1np27/sudoku-vit){:target="_blank" rel="noopener"}

[8] <span id="ref8"></span>[Attention Is All You Need](https://arxiv.org/abs/1706.03762){:target="_blank" rel="noopener"}

[9] <span id="ref9"></span>[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385){:target="_blank" rel="noopener"}

[10] <span id="ref10"></span>[Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030){:target="_blank" rel="noopener"}

[11] <span id="ref11"></span>[Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101){:target="_blank" rel="noopener"}

[12] <span id="ref12"></span>[MLP-Mixer: An all-MLP Architecture for Vision](https://arxiv.org/abs/2105.01601){:target="_blank" rel="noopener"}

[13] <span id="ref13"></span>[Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027){:target="_blank" rel="noopener"}
