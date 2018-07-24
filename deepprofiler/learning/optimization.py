import GPy, GPyOpt
import deepprofiler.learning.training

class Optimize(object):

    def __init__(self, config, dset, epoch=1, seed=None):
        config["model"]["comet_ml"] = False
        self.config = config
        self.dset = dset
        self.epoch = epoch
        self.seed = seed
        self.bounds = [{'name': 'learning_rate', 'type': 'continuous',  'domain': (0.000001, 1.0)}]
    
    def model(self):
        evaluation = deepprofiler.learning.training.learn_model(self.config, self.dset, self.epoch, self.seed, verbose=0)
        return evaluation

    def f(self, x):
        self.config["model"]["params"]['learning_rate'] = float(x[:,0])
        evaluation = self.model()
        return evaluation[0]

    def optimize(self):
        opt = GPyOpt.methods.BayesianOptimization(f=self.f, domain=self.bounds)
        opt.run_optimization(max_iter=2)
        print("""
        Optimized Parameters:
        \t{0}:\t{1}
        """.format(self.bounds[0]["name"],opt.x_opt[0]))
        print("optimized loss: {0}".format(opt.fx_opt))
