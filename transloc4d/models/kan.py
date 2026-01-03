import torch
import torch.nn as nn
import MinkowskiEngine as ME
from ikan.ChebyKAN import ChebyKAN, ChebyKANLinear

class KanLinear(nn.Module):

    def __init__(self, input_channel, output_channel):
        super(KanLinear, self).__init__()

        self.ChebyKANLinear = ChebyKANLinear(input_channel, output_channel)

    def forward(self, feat):

        features = feat.decomposed_features #这可以把稀疏tensor转成正常卷积处理的tensor
        batch_size = len(features)

        for i in range(batch_size):
            features_single = features[i].unsqueeze(0)
            a= self.ChebyKANLinear(features_single)
            features[i] = a[0]

        feat = ME.SparseTensor(
            features=torch.cat(features, 0),
            coordinates=feat.coordinates,
            coordinate_manager=feat.coordinate_manager,
        )

        return feat