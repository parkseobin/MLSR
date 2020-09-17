import tensorflow as tf
from IDN_definition import IDN



class IDNModel(object):
    def __init__(self, tensor_input, gt_output, scope_name):
        self.tensor_input = tensor_input
        self.scope_name = scope_name
        self.gt_output = gt_output
        self.model_name = 'IDN'


    def build_model(self):
        img_lr, img_bicubic = self.tensor_input

        tf.get_variable_scope().reuse_variables()
        if(self.scope_name == self.model_name):
            output = IDN(img_lr, img_bicubic, 2)
        else:
            with tf.variable_scope(self.scope_name, reuse=tf.AUTO_REUSE):
                output = IDN(img_lr, img_bicubic, 2)

        self.output = output
        self.loss = tf.reduce_mean((output - self.gt_output)**2)

        y_vector = [0.25678824, 0.50412941, 0.09790588]
        output_y = output[:, :, :, 0:1] * y_vector[0] + output[:, :, :, 1:2]*y_vector[1] + output[:, :, :, 2:3]*y_vector[2]
        gt_output_y = self.gt_output[:, :, :, 0:1] * y_vector[0] + self.gt_output[:, :, :, 1:2]*y_vector[1] + self.gt_output[:, :, :, 2:3]*y_vector[2]
        self.output = tf.clip_by_value(output, 0, 255)
        output = self.output
        self.psnr = tf.image.psnr(output, self.gt_output, max_val=255)
        self.psnr_y = tf.image.psnr(output_y[:, 2:-2, 2:-2, :], gt_output_y[:, 2:-2, 2:-2, :], max_val=255)
        #self.psnr_y = tf.image.psnr(output_y, gt_output_y, max_val=255)
        self.ssim = tf.image.ssim_multiscale(output, self.gt_output, max_val=255)
        self.ssim_y = tf.image.ssim(output_y, gt_output_y, max_val=255)
        img_bicubic_y = img_bicubic[:, :, :, 0:1] * y_vector[0] + img_bicubic[:, :, :, 1:2]*y_vector[1] + img_bicubic[:, :, :, 2:3]*y_vector[2]
        self.bicubic_psnr = tf.image.psnr(img_bicubic_y, gt_output_y, max_val=255)