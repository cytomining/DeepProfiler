import GPy, GPyOpt
import deepprofiler.learning.training

def parse_tuple(string):
    try:
        s = eval(str(string))
        if type(s) == tuple:
            return s
        return
    except:
        return

class Optimize(object):

    def __init__(self, config, dset, epoch=1, seed=None):
        config["train"]["comet_ml"]["track"] = False
        self.config = config
        self.dset = dset
        self.epoch = epoch
        self.seed = seed
        self.bounds = []
        for i in range(len(self.config["optim"]["names"])):
            if self.config["optim"]["names"][i] == "logarithmic":
                self.bounds.append({
                    'name': self.config["optim"]["names"][i],
                    'type': "continuous",
                    'domain': parse_tuple(self.config["optim"]["domains"][i])
                })
            else:
                self.bounds.append({
                    'name': self.config["optim"]["names"][i],
                    'type': self.config["optim"]["types"][i],
                    'domain': parse_tuple(self.config["optim"]["domains"][i])
                })
    
    
    def model(self):
        evaluation = deepprofiler.learning.training.learn_model(self.config, self.dset, self.epoch, self.seed, verbose=0)
        return evaluation

    def f(self, x):
        for i in range(len(self.config["optim"]["names"])):
            if self.config["optim"]["types"][i] == "continuous":
                self.config['train']["model"]["params"][self.config["optim"]["names"][i]] = float(x[:,i])
            elif self.config["optim"]["types"][i] == "discrete":
                self.config['train']["model"]["params"][self.config["optim"]["names"][i]] = int(x[:,i])
            elif self.config["optim"]["types"][i] == "logarithmic":
                self.config['train']["model"]["params"][self.config["optim"]["names"][i]] = float(10**x[:,i])
        evaluation = self.model()
        return evaluation[0]

    def optimize(self):
        opt = GPyOpt.methods.BayesianOptimization(f=self.f, domain=self.bounds, maximize=False)
        opt.run_optimization(max_iter=self.config["optim"]["max_iter"])
        print("Optimized Parameters:")
        for i in range(len(self.config["optim"]["names"])):
            print(
            """
            \t{0}:\t{1}
            """.format(self.bounds[i]["name"],opt.x_opt[i]))
        print("optimized loss: {0}".format(opt.fx_opt))
