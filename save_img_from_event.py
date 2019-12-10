import os
import scipy.misc
import tensorflow as tf

def save_images_from_event(fn, tag, output_dir='./'):
    assert(os.path.isdir(output_dir))

    image_str = tf.placeholder(tf.string)
    im_tf = tf.image.decode_image(image_str)

    sess = tf.InteractiveSession()
    with sess.as_default():
        count = 0
        for e in tf.train.summary_iterator(fn):
            for v in e.summary.value:
                if v.tag == tag:
                    im = im_tf.eval({image_str: v.image.encoded_image_string})
                    output_fn = os.path.realpath('{}/image_{:05d}.png'.format(output_dir, count))
                    print("Saving '{}'".format(output_fn))
                    scipy.misc.imsave(output_fn, im)
                    count += 1  

path_to_event_file = '/home/minghanz/tmp/mono_model_KITTI_Lidar/val_Tue Dec  3 23:48:28 2019/events.out.tfevents.1575434908.MCity-GPU-Server'
tag_name = 'disp_0/gt_0'
output_dir = '/mnt/storage8t/minghanz/monodepth2_tmp/img_from_event'
save_images_from_event(path_to_event_file, tag_name, output_dir)