from torch.nn import Module, BatchNorm1d, ReLU, LeakyReLU, Conv1d, ModuleList, Softmax, Sigmoid, Flatten, Dropout2d, Linear
import numpy as np

CARDINALITY_ITEM = 16

class ResidualUnit(Module):
    def __init__(self, l, w, ar, bot_mul=1):
        super().__init__()
        bot_channels = int(round(l * bot_mul))
        self.batchnorm1 = BatchNorm1d(l)
        self.relu = LeakyReLU(0.1)
        self.batchnorm2 = BatchNorm1d(l)
        self.C = bot_channels//CARDINALITY_ITEM
        self.conv1 = Conv1d(l, l, w, dilation=ar, padding=(w-1)*ar//2, groups=self.C)
        self.conv2 = Conv1d(l, l, w, dilation=ar, padding=(w-1)*ar//2, groups=self.C)

    def forward(self, x, y):
        # x1 = self.relu(self.batchnorm1(self.conv1(x)))
        # x2 = self.relu(self.batchnorm2(self.conv2(x1)))
        x1 = self.conv1(self.relu(self.batchnorm1(x)))
        x2 = self.conv1(self.relu(self.batchnorm1(x1)))
        return x + x2, y


class Skip(Module):
    def __init__(self, l):
        super().__init__()
        self.conv = Conv1d(l, l, 1)

    def forward(self, x, y):
        return x, self.conv(x) + y


class Cropping1D(Module):
    def __init__(self, cropping):
        super().__init__()
        self.cropping = cropping
    
    def forward(self, x):
        return x[:, :, self.cropping[0]:-self.cropping[1]]


class SPLAN(Module):
    def __init__(self, L=64, W=np.array([11]*8+[21]*4+[41]*4), AR=np.array([1]*4+[4]*4+[10]*4+[25]*4), flanking_size=80):
        super().__init__()
        self.CL = 2 * (AR * (W - 1)).sum()  # context length
        self.flanking_size = flanking_size
        self.conv1 = Conv1d(4, L, 1)
        self.skip1 = Skip(L)
        self.residual_blocks = ModuleList()
        for i, (w, r) in enumerate(zip(W, AR)):
            self.residual_blocks.append(ResidualUnit(L, w, r))
            if (i+1) % 4 == 0:
                self.residual_blocks.append(Skip(L))
        if (len(W)+1) % 4 != 0:
            self.residual_blocks.append(Skip(L))
        # Determine cropping size based on context length calculation
        self.crop = Cropping1D((self.flanking_size//2, self.flanking_size//2))  # Adjust this based on your specific needs
        self.last_cov = Conv1d(L, 3, 1)
        self.softmax = Softmax(dim=1)

    def forward(self, x):
        x, skip = self.skip1(self.conv1(x), 0)
        for m in self.residual_blocks:
            x, skip = m(x, skip)
        # print("Shape of skip: ", skip.shape)
        x = self.crop(skip)  # Apply cropping here
        # print("Shape after cropping: ", x.shape)
        #######################################
        # predicting pb for every bp
        #######################################
        x = self.last_cov(x)
        x = self.softmax(x)
        return x
    
    
# def SpliceAI(L, W, AR):
#     # L: Number of convolution kernels
#     # W: Convolution window size in each residual unit
#     # AR: Atrous rate in each residual unit

#     assert len(W) == len(AR)

#     CL = 2 * np.sum(AR*(W-1))

#     input0 = Input(shape=(None, 4))
#     conv = Conv1D(L, 1)(input0)
#     skip = Conv1D(L, 1)(conv)

#     for i in range(len(W)):
#         conv = ResidualUnit(L, W[i], AR[i])(conv)
        
#         if (((i+1) % 4 == 0) or ((i+1) == len(W))):
#             # Skip connections to the output after every 4 residual units
#             dense = Conv1D(L, 1)(conv)
#             skip = add([skip, dense])

#     # skip = Cropping1D(CL/2)(skip)

#     # Calculate cropping amount, ensuring it's an integer
#     cropping_amount = int(np.round(CL / 2))
#     # Use a tuple with the calculated amount for both the beginning and end
#     skip = Cropping1D((cropping_amount, cropping_amount))(skip)

#     output0 = [[] for t in range(1)]

#     for t in range(1):
#         output0[t] = Conv1D(3, 1, activation='softmax')(skip)
    
#     model = Model(inputs=input0, outputs=output0)

#     return model

