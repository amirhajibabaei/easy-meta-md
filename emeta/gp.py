# +
"""
Given a positive function f (e.g. PDF, hist, etc)
which goes to zero at long distances, "GPModel"
generates a gaussian process regression model for 
f with tuned hyper-params and idealy optimally 
sampled (X, Y).
"""
import torch
import gpytorch


class GPModel(gpytorch.models.ExactGP):
    def __init__(self, dim):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        train_x = torch.empty(0, dim)
        train_y = torch.empty(0)
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def append(self, x, y):
        X = torch.cat([self.train_inputs[0], x.view(1, -1)])
        Y = torch.cat([self.train_targets, y.view(1)])
        self.set_train_data(X, Y, strict=False)

    def build(self, func, inputs, atol=0.1, train=True):
        self.sample(func, inputs, atol=atol, train=train)
        self.bootstrap(func, atol=atol)

    def sample(self, func, inputs, atol=0.1, train=True):
        for i in inputs:
            x = self.optimize_inducing(func, i)
            f = func(x)
            if self.train_inputs[0].size(0) == 0:
                self.append(x, f)
            else:
                delta = (self(x).mean-f).abs()
                if delta > atol:
                    self.append(x, f)
            if train:
                self.optimize_hyperparams()

    def bootstrap(self, func, atol=0.1, iterations=2):
        for _ in range(iterations):
            inputs = self.train_inputs[0]
            self.set_train_data(torch.empty(0, inputs.size(1)),
                                torch.empty(0), strict=False)
            self.sample(func, inputs, atol=atol, train=False)
            self.optimize_hyperparams()

    def optimize_hyperparams(self):
        self.train()
        self.likelihood.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)
        _loss = None
        while True:
            optimizer.zero_grad()
            output = self(self.train_inputs[0])
            loss = -mll(output, self.train_targets)
            loss.backward()
            optimizer.step()
            if loss_break(_loss, loss):
                break
            _loss = loss
        self.eval()
        self.likelihood.eval()

    def optimize_inducing(self, func, i):
        x = i.detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([x], lr=0.1)
        _loss = None
        while True:
            optimizer.zero_grad()
            y = func(x)
            try:
                var = self(x).variance
            except:
                var = torch.ones(1)
            loss = -y*var
            loss.backward()
            optimizer.step()
            if loss_break(_loss, loss):
                break
            _loss = loss
        return x.detach()


def loss_break(_loss, loss):
    return _loss is not None and (loss-_loss).abs() < loss.abs()*1e-3


def test():
    import pylab as plt

    def test_model(model):
        test_x = torch.arange(-10, 10, 0.1)
        pred = model.likelihood(model(test_x))
        lower, upper = pred.confidence_region()
        plt.plot(test_x, pred.mean.detach(), label='model')
        plt.fill_between(test_x, lower.detach(), upper.detach(), alpha=0.5)
        test_y = func(test_x)
        plt.plot(test_x, test_y, ':', label='real')
        plt.scatter(model.train_inputs[0], model.train_targets)
        plt.legend()
        print(model.train_inputs[0].shape)

    def func(x):
        return x.pow(2).neg().div(4).exp().view(-1)

    inputs = torch.randn(20, 1)
    model = GPModel(1)
    model.build(func, inputs)
    test_model(model)


if __name__ == '__main__':
    test()
