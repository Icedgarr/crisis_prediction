import tensorflow as tf
from tensorflow.keras.losses import Loss, Reduction


class WeightedBinaryCrossEntropy(Loss):
    """
    Args:
      pos_weight: Scalar to affect the positive labels of the loss function.
      weight: Scalar to affect the entirety of the loss function.
      from_logits: If input has already been transformed with sigmoid set the parameter to False, otherwise set to True
      reduction: Type of tf.keras.losses.Reduction to apply to loss.
      name: Name of the loss function.
    """

    def __init__(self, pos_weight, weight, from_logits=False,
                 reduction=Reduction.AUTO,
                 name='weighted_binary_crossentropy'):
        super().__init__(reduction=reduction, name=name)
        self.pos_weight = pos_weight
        self.weight = weight
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        if not self.from_logits:
            # Manually calculate the weighted cross entropy.
            # Formula is qz * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
            # where z are labels, x is logits, and q is the weight.
            # Since the values passed are from sigmoid (assuming in this case)
            # sigmoid(x) will be replaced by y_pred
            # qz * -log(sigmoid(x)) 1e-6 is added as an epsilon to stop passing a zero into the log
            x_1 = y_true * self.pos_weight * -tf.math.log(y_pred + 1e-6)
            # (1 - z) * -log(1 - sigmoid(x)). Epsilon is added to prevent passing a zero into the log
            x_2 = (1 - y_true) * -tf.math.log(1 - y_pred + 1e-6)
            return tf.add(x_1, x_2) * self.weight
        # Use built in function
        return tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred, self.pos_weight) * self.weight
