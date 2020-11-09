import tensorflow as tf


def ssim_loss(dynamic_range):
    def loss(imgs_true, imgs_pred):
        return 1 - tf.image.ssim(imgs_true, imgs_pred, dynamic_range)

    return loss


def mssim_loss(dynamic_range):
    def loss(imgs_true, imgs_pred):
        return 1 - tf.image.ssim_multiscale(imgs_true, imgs_pred, dynamic_range)
        # return (1 - tf.image.ssim_multiscale(imgs_true, imgs_pred, dynamic_range)) ** 2

    return loss


def l2_loss(imgs_true, imgs_pred):
    return tf.nn.l2_loss(imgs_true - imgs_pred)


# https://www.tensorflow.org/api_docs/python/tf/nn/l2_loss?hl=ko
