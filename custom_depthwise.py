from keras.layers import DepthwiseConv2D
from keras.utils import register_keras_serializable

@register_keras_serializable()
class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, groups=1, **kwargs):
        kwargs.pop('groups', None)  # Remove the groups argument
        super(CustomDepthwiseConv2D, self).__init__(**kwargs)

    def get_config(self):
        config = super(CustomDepthwiseConv2D, self).get_config()
        config['groups'] = 1  # Or whatever value you need
        return config
