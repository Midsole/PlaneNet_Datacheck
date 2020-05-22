import tensorflow as tf
import numpy as np
# import IPython.display as display


HEIGHT=192
WIDTH=256
NUM_PLANES = 20
NUM_THREADS = 4
numOutputPlanes = 20

print(tf.__version__)
print("Finish parameter setting")

filename = "/planes_scannet_val.tfrecords"
filename_queue = tf.train.string_input_producer(['/planes_scannet_val.tfrecords'], num_epochs=1)
reader = tf.TFRecordReader()
print(filename_queue)
_, serialized_example = reader.read(filename_queue)
print(serialized_example)
features = tf.parse_single_example(
            serialized_example,
            # Defaults are not specified since both keys are required.
            features={
                #'height': tf.FixedLenFeature([], tf.int64),
                #'width': tf.FixedLenFeature([], tf.int64),
                'image_raw': tf.FixedLenFeature([], tf.string),
                'image_path': tf.FixedLenFeature([], tf.string),
                'num_planes': tf.FixedLenFeature([], tf.int64),
                'plane': tf.FixedLenFeature([NUM_PLANES * 3], tf.float32),
                #'plane_relation': tf.FixedLenFeature([NUM_PLANES * NUM_PLANES], tf.float32),
                'segmentation_raw': tf.FixedLenFeature([], tf.string),
                'depth': tf.FixedLenFeature([HEIGHT * WIDTH], tf.float32),
                'normal': tf.FixedLenFeature([HEIGHT * WIDTH * 3], tf.float32),
                'semantics_raw': tf.FixedLenFeature([], tf.string),                
                'boundary_raw': tf.FixedLenFeature([], tf.string),
                'info': tf.FixedLenFeature([4 * 4 + 4], tf.float32),                
            })

sess=tf.Session()
print("stuck after this:1")
sess.run(tf.global_variables_initializer())
print("stuck after this:2")
image = tf.decode_raw(features['image_raw'], tf.uint8)
print(type(features['image_raw']))
print(type(image))
print("stuck after this:3")
# img_numpy=image.eval(session=sess)
# print(type(image))
# print(type(img_numpy))
# image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
# print(type(image))
# image = tf.reshape(image, [HEIGHT, WIDTH, 3])
# print(type(image))

# print(image)

# depth = features['depth']
# depth = tf.reshape(depth, [HEIGHT, WIDTH, 1])

# normal = features['normal']
# normal = tf.reshape(normal, [HEIGHT, WIDTH, 3])
# normal = tf.nn.l2_normalize(normal, dim=2)

# semantics = tf.decode_raw(features['semantics_raw'], tf.uint8)
# semantics = tf.cast(tf.reshape(semantics, [HEIGHT, WIDTH]), tf.int32)

# numPlanes = tf.minimum(tf.cast(features['num_planes'], tf.int32), numOutputPlanes)
# numPlanesOri = numPlanes
# numPlanes = tf.maximum(numPlanes, 1)

# planes = features['plane']
# planes = tf.reshape(planes, [NUM_PLANES, 3])
# planes = tf.slice(planes, [0, 0], [numPlanes, 3])

# shuffle_inds = tf.one_hot(tf.range(numPlanes), numPlanes)
# planes = tf.transpose(tf.matmul(tf.transpose(planes), shuffle_inds))
# planes = tf.reshape(planes, [numPlanes, 3])
# planes = tf.concat([planes, tf.zeros([numOutputPlanes - numPlanes, 3])], axis=0)
# planes = tf.reshape(planes, [numOutputPlanes, 3])

# boundary = tf.decode_raw(features['boundary_raw'], tf.uint8)
# boundary = tf.cast(tf.reshape(boundary, (HEIGHT, WIDTH, 2)), tf.float32)
# segmentation = tf.decode_raw(features['segmentation_raw'], tf.uint8)
# segmentation = tf.reshape(segmentation, [HEIGHT, WIDTH, 1])
# coef = tf.range(numPlanes)
# coef = tf.reshape(tf.matmul(tf.reshape(coef, [-1, numPlanes]), tf.cast(shuffle_inds, tf.int32)), [1, 1, numPlanes])
# plane_masks = tf.cast(tf.equal(segmentation, tf.cast(coef, tf.uint8)), tf.float32)
# plane_masks = tf.concat([plane_masks, tf.zeros([HEIGHT, WIDTH, numOutputPlanes - numPlanes])], axis=2)
# plane_masks = tf.reshape(plane_masks, [HEIGHT, WIDTH, numOutputPlanes])

# non_plane_mask = 1 - tf.reduce_max(plane_masks, axis=2, keep_dims=True)
# print("finish data reading")
# print(plane_masks)
# print(image)
