# UNLDSR

## Abstract
Classical variational methods for solving image processing problems are more interpretable and flexible than
pure deep learning approaches, but their performance is limited by the use of rigid priors. Deep unfolding
networks combine the strengths of both by unfolding the steps of the optimization algorithm used to estimate
the minimizer of an energy functional into a deep learning framework. In this paper, we propose an unfolding
approach to extend a variational model exploiting self-similarity of natural images in the data fidelity term for
single-image super-resolution. The proximal, downsampling and upsampling operators are written in terms
of a neural network specifically designed for each purpose. Moreover, we include a new multi-head attention
module to replace the nonlocal term in the original formulation. A comprehensive evaluation covering a wide
range of sampling factors and noise realizations proves the benefits of the proposed unfolding techniques. The
model shows to better preserve image geometry while being robust to noise.

## Results
### PSNR
| Noise | BIC   | VCLD  | VNLD  | UCLD  | UNLD      |
|-------|-------|-------|-------|-------|-----------|
| 0     | 26,33 | 28,29 | 29,40 | 29,31 | **29,57** |
| 5     | 25,69 | 27,61 | 27,95 | 28,39 | **28,49** |
| 10    | 24,33 | 26,58 | 26,63 | 27,01 | **27,30** |
| 25    | 20,18 | 24,02 | 24,48 | 24,79 | **24,86** |

### SSIM
| Noise | BIC   | VCLD  | VNLD  | UCLD  | UNLD      |
|-------|-------|-------|-------|-------|-----------|
| 0     | 0,784 | 0,811 | 0,871 | 0,870 | **0,876** |
| 5     | 0,716 | 0,781 | 0,805 | 0,822 | **0,826** |
| 10    | 0,595 | 0,738 | 0,739 | 0,758 | **0,775** |
| 25    | 0,342 | 0,597 | 0,624 | 0,648 | **0,654** |