import models.config_DeepMIH as c
from models.rrdb_denselayer_DeepMIH import *


class Dense(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, input, output):
        super(Dense, self).__init__()

        self.dense = ResidualDenseBlock_out(input, output, nf=c.nf, gc=c.gc)

    def forward(self, x):
        out = self.dense(x)

        return out


