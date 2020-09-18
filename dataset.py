import os
import tensorflow as tf
import numpy as np
from PIL import Image
import time
import threading



class Dataset(object):
    def __init__(self, args):
        self.train_directory = args.train_directory
        self.validation_directory = args.validation_directory
        self.batch_size = args.batch_size
        self.train_list = os.listdir(self.train_directory)
        self.test_list = os.listdir(self.validation_directory)
        assert len(self.train_list) > 0 and len(self.test_list) > 0, 'Empty dataset'
        self.test_i = 0
        self.patch_size = args.patch_size   # HR patch size
        self.queue_size = 256
        self.make_queue()


    def make_queue(self):
        image_shape_hr = (self.patch_size, self.patch_size, 3)
        image_shape_lr = (self.patch_size//2, self.patch_size//2, 3)
        image_shape_lr_ = (self.patch_size//4, self.patch_size//4, 3)

        self.maml_img_lr = tf.placeholder(tf.float32, (None,) + image_shape_lr)
        self.maml_img_bicubic = tf.placeholder(tf.float32, (None,) + image_shape_hr)
        self.maml_img_hr = tf.placeholder(tf.float32, (None,) + image_shape_hr)
        self.img_lr = tf.placeholder(tf.float32, (None,) + image_shape_lr_)
        self.img_bicubic = tf.placeholder(tf.float32, (None,) + image_shape_lr)
        self.img_hr = tf.placeholder(tf.float32, (None,) + image_shape_lr)
        # Dequeues element in random order
        queue = tf.RandomShuffleQueue(self.queue_size, self.batch_size, 
            dtypes=(tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32), 
            shapes=(image_shape_lr_, image_shape_lr, image_shape_lr,
            image_shape_lr, image_shape_hr, image_shape_hr))

        self.enqueue_many = queue.enqueue_many([self.img_lr, self.img_bicubic, self.img_hr,
            self.maml_img_lr, self.maml_img_bicubic, self.maml_img_hr])
        self.dequeue_many = queue.dequeue_many(self.batch_size)


    def start_enqueue_daemon(self, sess):
        def enqueue_thread(sess):
            while(True):
                img_lr, img_bicubic, img_hr, maml_img_lr, maml_img_bicubic, maml_img_hr \
                    = self.next(test=False)
                sess.run([self.enqueue_many], feed_dict={
                    self.img_lr: img_lr, 
                    self.img_bicubic: img_bicubic,
                    self.img_hr: img_hr,
                    self.maml_img_lr: maml_img_lr, 
                    self.maml_img_bicubic: maml_img_bicubic,
                    self.maml_img_hr: maml_img_hr
                })
                time.sleep(0.02)

        thread_number = 1
        threads = []
        for i in range(thread_number):
            t = threading.Thread(target=enqueue_thread, args=(sess,), daemon=True)
            t.start()
            threads.append(t)

        return threads


    def augmentation(self, input_img):
        '''
        input_img: Pillow Image object
        returns: Pillow Image object
        '''
        aug_methods = [
            Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM,
            Image.ROTATE_90, Image.ROTATE_180, 
            Image.ROTATE_270, Image.TRANSPOSE,
            Image.TRANSVERSE
        ]

        if(np.random.randint(len(aug_methods) + 1) == 0):
            return input_img
        else:
            return input_img.transpose(np.random.choice(aug_methods))


    def choose_random_image(self, test):
        if(test):
            random_img = os.path.join(self.validation_directory, self.test_list[self.test_i])
            self.test_i += 1
            if(self.test_i >= len(self.test_list)):
                self.test_i = 0
        else:
            random_img = os.path.join(self.train_directory, np.random.choice(self.train_list))
        
        random_img = Image.open(random_img).convert('RGB')
        random_img = random_img.crop((0, 0, 
            random_img.size[0] - random_img.size[0]%8, random_img.size[1] - random_img.size[1]%8
        ))

        return random_img


    def next(self, test):
        if(test):
            maml_hr_img = self.choose_random_image(test)
        else:
            #maml_hr_img = self.global_img   
            maml_hr_img = self.choose_random_image(test)

            # patch size on HR image
            if(maml_hr_img.size[1] <= self.patch_size or maml_hr_img.size[0] <= self.patch_size):
                return self.next(test)

            left = np.random.randint(maml_hr_img.size[0] - self.patch_size)
            upper = np.random.randint(maml_hr_img.size[1] - self.patch_size)
            maml_hr_img = maml_hr_img.crop((left, upper, left + self.patch_size, upper + self.patch_size))

        lev2 = maml_hr_img.size
        lev1 = (lev2[0]//2, lev2[1]//2)
        lev0 = (lev1[0]//2, lev1[1]//2)

        maml_lr_img = maml_hr_img.resize(lev1, resample=Image.BICUBIC)
        maml_bicubic_img = maml_lr_img.resize(lev2, resample=Image.BICUBIC)

        hr_img = maml_lr_img.copy()
        lr_img = hr_img.resize(lev0, resample=Image.BICUBIC)
        bicubic_img = lr_img.resize(lev1, resample=Image.BICUBIC)

        maml_hr_img = np.array(maml_hr_img, dtype=np.float32, ndmin=4)
        maml_lr_img = np.array(maml_lr_img, dtype=np.float32, ndmin=4)
        maml_bicubic_img = np.array(maml_bicubic_img, dtype=np.float32, ndmin=4)
        hr_img = np.array(hr_img, dtype=np.float32, ndmin=4)
        lr_img = np.array(lr_img, dtype=np.float32, ndmin=4)
        bicubic_img = np.array(bicubic_img, dtype=np.float32, ndmin=4)

        return lr_img, bicubic_img, hr_img, maml_lr_img, maml_bicubic_img, maml_hr_img