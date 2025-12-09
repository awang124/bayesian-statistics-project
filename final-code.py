#!/usr/bin/env python
# coding: utf-8

# ### Imports

import tqdm
import torch
import gpytorch
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


# ### Display Image Pair

def visimg(img1, img2, figsize=(10, 4)):
    fig, axs = plt.subplots(ncols=2, figsize=figsize)
    axs[0].imshow(img1, cmap="grey")
    axs[1].imshow(img2, cmap="grey")
    axs[0].axis("off")
    axs[1].axis("off")
    plt.tight_layout()
    plt.show()


# ### Display Optical Flow Field

def visflo(u, v, sr, scale, agg=max, log=True, figsize=(5, 4)):
    fig, axs = plt.subplots(figsize=figsize)
    yy, xx = np.mgrid[0:u.shape[0]:sr[0], 0:u.shape[1]:sr[1]]
    uu, vv = u[::sr[0], ::sr[1]], v[::sr[0], ::sr[1]]
    axs.quiver(xx, yy, uu, -vv, scale=scale)
    axs.invert_yaxis()
    axs.axis("off")
    plt.show()


# ### Display Optical Flow Covariance (Eigenvalues)

def viscov(cc, agg=max, log=True, figsize=(5, 4)):
    fig, axs = plt.subplots(figsize=figsize)
    yy, xx = np.mgrid[0:cov.shape[0]:sr[0], 0:cov.shape[1]:sr[1]]

    # COMPUTE COVARIANCE EIGENVALUES
    eig = np.zeros_like(xx, dtype=np.float32)
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            eig[i, j] = agg(np.linalg.eigvals(cc[i, j]))
    eig = np.log(eig) if log else eig
    
    cm = axs.pcolormesh(xx, yy, eig, cmap="Greys_r")
    plt.colorbar(cm, ax=axs)
    axs.invert_yaxis()
    axs.axis("off")
    plt.show()


# ### Wang-Orquiza (Maximum-Likelihood) Method

def gaussianKernel(ksize=5, sigma=None, norm=True):
    if not sigma:
        sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
    row = np.arange(ksize) - (ksize - 1) / 2
    xx, yy = np.meshgrid(row, row)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return kernel / kernel.sum() if norm else kernel

def maximumLikelihood(img1, img2, ksize=9, sigma=1):
    k = ksize // 2

    # PAD IMAGES (SO EACH PIXEL HAS FULL KSIZE * KSIZE WINDOW)
    pad1 = cv.copyMakeBorder(
        img1.astype(np.float32), k, k, k, k, cv.BORDER_REFLECT
    )
    pad2 = cv.copyMakeBorder(
        img2.astype(np.float32), k, k, k, k, cv.BORDER_REFLECT
    )

    # COMPUTE IMAGE GRADIENTS
    Ix = cv.Sobel(pad1, cv.CV_32F, 1, 0, ksize=3)
    Iy = cv.Sobel(pad2, cv.CV_32F, 0, 1, ksize=3)
    It = pad2 - pad1

    (h, w) = img1.shape
    u = np.zeros((h, w), dtype=np.float32)
    v = np.zeros((h, w), dtype=np.float32)
    c = np.zeros((h, w, 2, 2), dtype=np.float32)
    noise = (sigma / gaussianKernel(ksize, norm=False)).ravel()

    for i in range(h):
        for j in range(w):
            m, n = i + k, j + k

            # GET GRADIENTS IN THIS PATCH
            ix = Ix[m-k:m+k+1, n-k:n+k+1].ravel()
            iy = Iy[m-k:m+k+1, n-k:n+k+1].ravel()
            it = It[m-k:m+k+1, n-k:n+k+1].ravel()

            # COMPUTE FLOW FOR THIS PATCH
            nabla = np.vstack((ix, iy))
            covar = np.linalg.inv((nabla / noise) @ nabla.T)
            mean  = covar @ (nabla / noise) @ -it
            u[i, j] = mean[0]
            v[i, j] = mean[1]
            c[i, j] = covar
    return u, v, c


# ### Gaussian Process Model Definition

class GPR(gpytorch.models.ApproximateGP):
    def __init__(self, num_tasks, num_inducing_points):
        inducing_points = torch.rand(num_tasks, num_inducing_points, 2)

        # DEFINE VARIATIONAL DISTRIBUTION (FOR OPTIMIZATION)
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(-2), batch_shape=torch.Size([num_tasks])
        )
        variational_strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(
            gpytorch.variational.VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=True
            ),
            num_tasks=num_tasks
        )
        super().__init__(variational_strategy)

        # DEFINE MEAN KERNEL (CONSTANT HERE)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_tasks]))

        # DEFINE COVARIANCE KERNEL (RBF HERE)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([num_tasks])),
            batch_shape=torch.Size([num_tasks])
        )

    def forward(self, X):
        # COMPUTE FUNCTIION VALUE AT INPUTS X
        mean = self.mean_module(X)
        covar = self.covar_module(X)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


# ### Gaussian Process Training

def train(model, likelihood, X, Y, lr, epochs):
    model.train()
    likelihood.train()

    params = [{"params": model.parameters()}, {"params": likelihood.parameters()}]
    optimizer = torch.optim.Adam(params, lr=lr)

    elbo = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=X.size(0))
    it = tqdm.tqdm_notebook(range(epochs), desc="Epoch")

    for _ in it:
        optimizer.zero_grad()
        output = model(X)
        loss = -elbo(output, Y)
        it.set_postfix(loss=loss.item())
        loss.backward()
        optimizer.step()


# ### Gaussian Process Posterior Prediction

def pred(model, likelihood, sy, X):
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = model(X)
        mean = sy.inverse_transform(pred.mean.cpu().numpy())
        var  = pred.variance.cpu().numpy() * sy.var_
    return mean, var


# ### Gaussian Process Complete Setup

def gaussianProcessRegress(uu, vv, cc, indpts, lr, epochs):
    sy, sx = StandardScaler(), StandardScaler()
    yy, xx = np.mgrid[0:uu.shape[0]:1, 0:uu.shape[1]:1]
    X = sx.fit_transform(np.dstack((xx, yy)).reshape(-1, 2))
    Y = sy.fit_transform(np.dstack((uu, vv)).reshape(-1, 2))
    C = cc.reshape(-1, 2, 2)
    C = np.vstack((C[..., 0, 0], C[..., 1, 1])).T

    X = torch.tensor(X, dtype=torch.float32).cuda()
    Y = torch.tensor(Y, dtype=torch.float32).cuda()
    C = torch.tensor(C, dtype=torch.float32).cuda()
    model = GPR(num_tasks=2, num_inducing_points=indpts).cuda()
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(2).cuda()

    train(model, likelihood, X, Y, lr, epochs)
    mean, var = pred(model, likelihood, sy, X)
    del model, likelihood, X, Y, C
    torch.cuda.empty_cache()

    u = mean[:, 0].reshape(uu.shape)
    v = mean[:, 1].reshape(vv.shape)
    c = var.reshape(uu.shape + (2,))
    return u, v, c


# ### Test Image Pair

img1 = cv.imread("yosemite/YOS02.bmp")
img2 = cv.imread("yosemite/YOS03.bmp")
img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
visimg(img1, img2)

# AVOID TEMPORAL ALIASING (STANDARD PREPROCESSING STEP)
img1 = cv.GaussianBlur(img1, (9, 9), -1)
img2 = cv.GaussianBlur(img2, (9, 9), -1)


# ### Wang-Orquiza Results

uu, vv, cc = maximumLikelihood(img1, img2, 15)
visflo(uu, vv, (9, 9), 20)
viscov(cc)


# ### GPR Results

uu, vv, cc = gaussianProcessRegress(uu, vv, cc, 100, 0.1, 1000)
visflo(uu, vv, (9, 9), 20)
