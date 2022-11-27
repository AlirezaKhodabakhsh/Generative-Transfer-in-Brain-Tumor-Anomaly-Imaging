from torch import nn

class Discriminator(nn.Module):
    """
    features : 16
    channels
    img_size
    optimization : 'gan'/'wgan'
    """
    def __init__(self, features, channels, img_size, optimization='gan', last_bias = True):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv_layers = nn.Sequential(
            *discriminator_block(channels, features*1, bn=False),
            *discriminator_block(features*1, features*2),
            *discriminator_block(features*2, features*4),
            *discriminator_block(features*4, features*8),
        )

        self.ds_size = img_size // 2 ** 4
        if optimization == 'gan':
            self.adverse_layer = nn.Sequential(
                nn.Linear(features*8 * self.ds_size ** 2, 1, bias=last_bias),
                nn.Sigmoid()
            )
        else:
            self.adverse_layer = nn.Sequential(
                nn.Linear(features*8 * self.ds_size ** 2, 1, bias=last_bias)
            )


    def forward(self, image):
        out = self.conv_layers(image)
        out = out.view(out.shape[0], -1)
        validity = self.adverse_layer(out)
        return validity