class SwinTransformerBlock:

    # Stage with W-MSA (Window Multi-Head Self-Attention)

    def layer_norm1(self):
        raise  NotImplementedError
    def W_MSA(self):
        raise  NotImplementedError

    def residual_add(self):
        raise  NotImplementedError

    def layer_norm_2(self):
        raise  NotImplementedError

    def MLP(self):
        raise  NotImplementedError

    def residual_add(self):
        raise NotImplementedError


    #Stage with SW-MSA (Shifted Window MSA)
    def layer_norm_1(self):
        raise NotImplementedError
    def SW_MSA(self):
        raise NotImplementedError

    def residual_add(self):
        raise NotImplementedError

    def layer_norm_2(self):
        raise NotImplementedError

    def MLP(self):
        raise NotImplementedError

    def residual_add(self):
        raise NotImplementedError


