import tensorflow as tf
import numpy as np
from PIL import Image
import os
import datetime



class SRTrainer(object):
    def __init__(self, dataset, network_model_class, args):
        self.batch_size = args.batch_size
        self.log_step = args.log_step
        self.validation_step = args.validation_step
        self.train_iteration = args.train_iteration
        self.param_restore_path = args.param_restore_path
        self.param_save_path = args.param_save_path
        self.lr_beta = args.lr_beta  # beta
        self.lr_alpha = args.lr_alpha    #alpha
        self.gradient_number = args.gradient_number
        self.dataset = dataset
        self.network_model_class = network_model_class
        self.build_success = False
        

    def set_optimizer(self):
        with tf.variable_scope('optimizer', reuse=tf.AUTO_REUSE):
            # Extract variables
            final_vars = [v for v in tf.global_variables() if v.name.startswith(self.final_network.model_name)]
            update_vars = [v for v in tf.global_variables() if v.name.startswith('update')]
            update_gradients = tf.gradients(self.update_network.loss, update_vars)
            accum_vars = [tf.Variable(tf.zeros_like(value), trainable=False) for value in update_gradients]

            # Inner udpate
            self.update_opt = tf.train.GradientDescentOptimizer(self.lr_alpha, name='update_opt')
            #self.update_opt = tf.train.AdamOptimizer(self.lr_alpha, name='update_opt')
            self.init_update_opt = tf.variables_initializer(self.update_opt.variables())
            self.update_opt = self.update_opt.minimize(self.update_network.loss, var_list=update_vars)

            # Outer udpate
            self.init_accumulator = [accum_var.assign(tf.zeros_like(accum_var)) for accum_var in accum_vars]
            self.accumulate_grads = [accum_vars[i].assign_add(update_gradients[i] / self.batch_size) for i in range(len(update_gradients))]
            self.fomaml_opt = tf.train.AdamOptimizer(self.lr_beta, name='fomaml_opt').apply_gradients(zip(accum_vars, final_vars))

            self.copy_opt = tf.train.GradientDescentOptimizer(self.lr_alpha, name='copy_opt')
            self.init_copy_opt = tf.variables_initializer(self.copy_opt.variables())
            self.copy_opt = self.copy_opt.minimize(self.copied_network.loss)
    

    def build(self):
        self.img_lr = tf.placeholder(tf.float32, (None, None, None, 3))
        self.img_bicubic = tf.placeholder(tf.float32, (None, None, None, 3))
        self.img_hr = tf.placeholder(tf.float32, (None, None, None, 3))

        self.final_network = self.network_model_class((self.img_lr, self.img_bicubic), self.img_hr, '')
        self.final_network.build_model()

        # Make model for update
        self.update_network = self.network_model_class((self.img_lr, self.img_bicubic), self.img_hr, 'update')
        self.update_network.build_model()

        # Make model for test
        self.copied_network = self.network_model_class((self.img_lr, self.img_bicubic), self.img_hr, 'copy')
        self.copied_network.build_model()

        # Making sync operation
        copy_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'copy')
        target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.final_network.model_name)
        update_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'update')
        copy_ops = []
        update_ops = []
        for i in range(len(copy_vars)):
            copy_ops.append(copy_vars[i].assign(target_vars[i]))
            update_ops.append(update_vars[i].assign(target_vars[i]))
        self.copy_sync_op = tf.group(*copy_ops)
        self.update_sync_op = tf.group(*update_ops)

        self.set_optimizer()
        self.build_success = True
        print('>> build complete!')


    def train_one_step(self, sess, epoch):
        train_img_lr, train_img_bicubic, train_img_hr, eval_img_lr, eval_img_bicubic, eval_img_hr \
            = sess.run(self.dataset.dequeue_many)
        loss = np.zeros((self.batch_size,), dtype=np.float32)
        psnr = np.zeros((self.batch_size,), dtype=np.float32)
        for i in range(self.batch_size):
            #1 copy latest parameter
            sess.run([self.update_sync_op, self.init_update_opt])

            #2 update parameter for given iteration number with train data
            update_iteration = self.gradient_number
            for j in range(update_iteration):
                sess.run(self.update_opt, feed_dict={
                    self.img_lr: [train_img_lr[i]],
                    self.img_bicubic: [train_img_bicubic[i]], 
                    self.img_hr: [train_img_hr[i]]
                })

            _, loss[i], psnr[i] = sess.run([
                self.accumulate_grads, self.update_network.loss, self.update_network.psnr_y
                ], feed_dict={
                    self.img_lr: [eval_img_lr[i]],
                    self.img_bicubic: [eval_img_bicubic[i]],
                    self.img_hr: [eval_img_hr[i]]
                })

        #3 update the parameter with accumulated gradients
        sess.run(self.fomaml_opt)
        sess.run(self.init_accumulator)

        return loss.mean(), psnr.mean()


    def validation(self, sess):
        test_size = len(self.dataset.test_list)
        updated_psnr = np.zeros((test_size,))
        base_psnr = np.zeros((test_size,))
        updated_ssim = np.zeros((test_size,))
        base_ssim = np.zeros((test_size,))
        bicubic_psnr = np.zeros((test_size,))
        for i in range(test_size):
            img_lr, img_bicubic, img_hr, maml_img_lr, maml_img_bicubic, maml_img_hr \
                = self.dataset.next(test=True)
            sess.run([self.copy_sync_op, self.init_copy_opt])

            base_psnr[i], bicubic_psnr[i], base_ssim[i] = sess.run([
                self.copied_network.psnr_y, self.copied_network.bicubic_psnr,
                self.copied_network.ssim_y
            ], feed_dict={
                    self.img_lr: maml_img_lr, 
                    self.img_bicubic: maml_img_bicubic, 
                    self.img_hr: maml_img_hr
            })

            for _ in range(self.gradient_number):
                sess.run([self.copy_opt], feed_dict={
                        self.img_lr: img_lr, 
                        self.img_bicubic: img_bicubic,
                        self.img_hr: img_hr,
                    })

            updated_psnr[i], updated_ssim[i] = sess.run([
                self.copied_network.psnr_y, self.copied_network.ssim_y
            ], feed_dict={
                    self.img_lr: maml_img_lr, 
                    self.img_bicubic: maml_img_bicubic, 
                    self.img_hr: maml_img_hr
            })

        return updated_psnr.mean(), base_psnr.mean(), bicubic_psnr.mean(), updated_ssim.mean(), base_ssim.mean()
       

    def train(self):
        assert self.param_save_path is not None, 'param_save_path is None'
        if(not os.path.exists(self.param_save_path)):
            os.makedirs(self.param_save_path)
        self.build()
        detector_saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.final_network.model_name))

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            self.dataset.start_enqueue_daemon(sess)
            if(self.param_restore_path != None):
                restore_path = os.path.join(self.param_restore_path, 'model.ckpt')
                detector_saver.restore(sess, restore_path)
                print('>> restored parameter from {}'.format(restore_path), flush=True)
            print('\n[*] Start training MLSR\n\n')

            loss_log, psnr_log, best_psnr_test = 0, 0, 0
            for i in range(1, self.train_iteration+1):
                train_loss, train_psnr = self.train_one_step(sess, i)
                loss_log += train_loss
                psnr_log += train_psnr
                if(i % self.log_step == 0):
                    loss_log /= self.log_step
                    psnr_log /= self.log_step
                    now = datetime.datetime.now()
                    print("[{}]".format(now.strftime('%Y-%m-%d %H:%M:%S')), flush=True)
                    print("Step: [{}/{}]\t Loss: {:.6f}\tPSNR: {:.6f}\n".format(i, self.train_iteration, loss_log, train_psnr), flush=True)
                    loss_log, psnr_log = 0, 0

                updated_psnr = None
                base_psnr = None
                if(i % self.validation_step == 0):
                    updated_psnr, base_psnr, bicubic_psnr, updated_ssim, base_ssim = self.validation(sess)
                    print(">> Test PSNR: (base: {}), (bicubic: {}), (updated: {})\n".format(
                        base_psnr, bicubic_psnr, updated_psnr
                    ), flush=True)
                    print(">> Test SSIM: (base: {}), (updated: {})\n\n".format(
                        base_ssim, updated_ssim
                    ), flush=True)

                    if(updated_psnr > best_psnr_test):
                        best_psnr_test = updated_psnr
                        detector_saver.save(sess, os.path.join(self.param_save_path, 'model.ckpt'))
                    detector_saver.save(sess, os.path.join(self.param_save_path, 'last.ckpt'))