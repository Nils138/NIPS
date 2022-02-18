""" Test controller """
from os.path import join, exists
from utils.rollout import RolloutGenerator
import torch

if __name__ == '__main__':
    logdir = 'logs'
    ctrl_file = join(logdir, 'ctrl', 'best.tar')
    assert exists(ctrl_file),\
        "Controller was not trained..."
    
    device = torch.device('cpu')
    
    generator = RolloutGenerator(logdir, device, 250)
    score = 0
    for i in range(100):
        with torch.no_grad():
            score += generator.rollout(None)
    print("Score: {}/100".format(score/10))
