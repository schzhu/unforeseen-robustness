"""
    Convert the downloaded Objectron data to the format compatible to ours.
     - Use       : SequenceExamples
     - Scale     : 640x480 to 64x64
     - Annotation: keep as is; may not affect our pipeline
     - Frames    :
       + Use the first frame as an anchor
       + Use the other frames as transformations
"""
import os, sys
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import numpy as np

# tensorflow
import tensorflow as tf

# custom (copied from the repo)
from objectron.schema import features
from objectron.dataset import box
from objectron.dataset import graphics


# ------------------------------------------------
#   Globals: dataset specific
# ------------------------------------------------
CHANNELS = 3
WIDTH    = 480
HEIGHT   = 640
RESIZE   = 64

# The 3D bounding box has 9 vertices, 0: is the center, and the 8 vertices of the 3D box.
NUM_KEYPOINTS = 9
NUM_FRAMES    = 4
MAX_VIDEOS    = 10      # but I assume that each file contains one video

# ------------------------------------------------
#   Globals: I/O specific
# ------------------------------------------------
_input_prefix  = 'datasets/objectron/sequences'
_output_prefix = 'datasets/objectron/preprocess'


# ------------------------------------------------
#   Globals: classification task specific 
# ------------------------------------------------
_class_names = [
    'bike',
    'book',
    'bottle',
    'camera',
    'cereal_box',
    'chair',
    'cup',
    'laptop',
    'shoe',
]


"""
    Return the required information
"""
def _parse_tfrecord(example):
    context, data = tf.io.parse_single_sequence_example(
                            example, 
                            sequence_features=features.SEQUENCE_FEATURE_MAP,
                            context_features=features.SEQUENCE_CONTEXT_MAP
                        )
    return context, data


"""
    Extract the data from the TFRecord file
    : datasets/objectron/preprocess
        -> train -> frames -> class -> video -> 64x64 png files
                 -> bboxes -> class -> video -> textfiles
        -> test  -> frames -> class -> video -> 64x64 png files
                 -> bboxes -> class -> video -> textfiles
    [Note: using this format, we can load by using PyTorch dataloaders.]
"""
def _extract_tfrecords(filename, outdir, classname, split='train', plot=False):
    # load the tfrecord file
    sequence_file = tf.io.gfile.glob(filename)
    sequence_data = tf.data.TFRecordDataset(sequence_file)
    sequence_data = sequence_data.map(_parse_tfrecord)

    # store location
    store_frame_dir = os.path.join(outdir, split, 'frames', classname)
    store_b_box_dir = os.path.join(outdir, split, 'bboxes', classname)

    # loop over the entire sequence files
    for context, data in sequence_data.take(MAX_VIDEOS): 

        # : prepare plotting
        if plot: fig, ax = plt.subplots(1, NUM_FRAMES, figsize = (12, 16))

        # : extract some context information
        #  > the number of frames
        #  > the unique sequence id (class/batch-i/j) useful for visualization/debugging
        num_frames = context['count']
        video_id   = context['sequence_id']
        
        # : convert from the TF Tensor format to Python native
        num_frames = num_frames.numpy()
        video_id   = video_id.numpy().decode('UTF-8')
        video_id   = '-'.join(video_id.split('/')[1:])
        
        # DEBUG
        if True:
            print(' : video information')
            print('   > ID    :', video_id)
            print('   > Frame :', num_frames)

        # : set the location to store frames/bounding box
        each_frame_dir = os.path.join(store_frame_dir, video_id)
        if not os.path.exists(each_frame_dir): os.makedirs(each_frame_dir)
        each_bboxs_dir = os.path.join(store_b_box_dir, video_id)
        if not os.path.exists(each_bboxs_dir): os.makedirs(each_bboxs_dir)

        # : data-holder
        sequence_bbox = []

        # : loop over the frames
        for fidx in range(num_frames):

            # --------------------------------
            #   Frames
            # --------------------------------
            
            # :: filename to store a frame
            each_frame_fname = os.path.join(each_frame_dir, 'frame-{}-{}.png'.format(fidx, num_frames))
        
            # :: load the frame
            each_frame = data[features.FEATURE_NAMES['IMAGE_ENCODED']][fidx]
            each_frame = tf.image.decode_png(each_frame, channels=3)
            each_frame.set_shape([HEIGHT, WIDTH, CHANNELS])

            # :: store the frame
            each_frame = each_frame.numpy()
            each_frame = Image.fromarray(each_frame)
            each_frame = each_frame.resize((RESIZE, RESIZE))
            each_frame.save("{}.png".format(each_frame_fname), format='png')

            # --------------------------------
            #   Bounding boxes
            # --------------------------------
            each_ninstances = data[features.FEATURE_NAMES['INSTANCE_NUM']][fidx].numpy()[0]
            each_keypoints  = data[features.FEATURE_NAMES['POINT_2D']].values.numpy().reshape(num_frames, each_ninstances, NUM_KEYPOINTS, 3)
            each_boundbox   = each_keypoints[fidx, 0, :, :]     # only for the first instance
            sequence_bbox.append(each_boundbox)
            
            # :: (disable) to overlay the bounding box
            # for instance_id in range(each_ninstances):
            #     each_frame = graphics.draw_annotation_on_image( \
            #         each_frame, each_keypoints[fidx, instance_id, :, :], [9])
            
            # :: plot (up to the limit)
            if plot and fidx < NUM_FRAMES:
                ax[fidx].grid(False)
                ax[fidx].imshow(each_frame)
                ax[fidx].get_xaxis().set_visible(False)
                ax[fidx].get_yaxis().set_visible(False)

        # : end for fidx...

        # : show
        if plot:
            fig.tight_layout()
            plt.show()

        # : store the bounding box information
        each_bboxs_fname = os.path.join(each_bboxs_dir, 'bounding_boxes.npy')
        sequence_bbox = np.stack(sequence_bbox, axis=0)
        np.save(each_bboxs_fname, sequence_bbox)

    # done.


def _scan_tfrecords_files(basedir):
    """
        Basedir: datasets/objectron/sequences (default)
    """
    # data holder
    tfrecord_filedict = { \
        each_class: { 'train': [], 'test': [] } \
        for each_class in _class_names }

    # loop over the class dir
    for each_class in tqdm(_class_names, desc=' : [scan-files]'):
        each_class_basedir = os.path.join(basedir, each_class)
        if not os.path.exists(each_class_basedir): continue

        # : load train files
        for each_file in os.listdir(each_class_basedir):
            if '{}_train-'.format(each_class) not in each_file: continue
            tfrecord_filedict[each_class]['train'].append(os.path.join(each_class_basedir, each_file))

        # : load test files
        for each_file in os.listdir(each_class_basedir):
            if '{}_test-'.format(each_class) not in each_file: continue
            tfrecord_filedict[each_class]['test'].append(os.path.join(each_class_basedir, each_file))

    # end for ...
    return tfrecord_filedict


def _preprocess_tfrecord_files(files, outdir=None):
    if not outdir: exit(1)

    # create the output dir
    if not os.path.exists(outdir): os.makedirs(outdir)

    # loop over the data and store the files
    for each_class, each_data in tfrecord_files.items():
        
        # : process the train data
        each_train_files = each_data['train']
        for each_file in each_train_files:
            try:
                _extract_tfrecords(each_file, outdir, each_class, split='train')
            except:
                print(' : skip [{}]'.format(each_file))

        # : process the test data
        each_test_files  = each_data['test']
        for each_file in each_test_files:
            try:
                _extract_tfrecords(each_file, outdir, each_class, split='test')
            except:
                print(' : skip [{}]'.format(each_file))

    # end for ...
    return


"""
    Process the tfrecords shards and save in the following format.
    : datasets/objectron/preprocess 
        -> train -> images      -> class -> video -> 64x64 png files
                 -> annotations -> class -> video -> textfiles
        -> test  -> images      -> class -> video -> 64x64 png files
                 -> annotations -> class -> video -> textfiles
    [Note: using this format, we can load by using PyTorch dataloaders.]
"""
if __name__ == "__main__":

    # scan files in the folders and check the statistics
    tfrecord_files = _scan_tfrecords_files(_input_prefix)

    # checkpoint: print the statistics
    print (' : TFRecord file statistics')
    for each_class, each_data in tfrecord_files.items():
        num_train = len(each_data['train'])
        num_test  = len(each_data['test'])
        print ('   > [{:>10s}]: {:>3d} train files and {:>3d} test files'.format( \
            each_class, num_train, num_test))


    # pre-process those data and construct the new data
    _preprocess_tfrecord_files(tfrecord_files, outdir=_output_prefix)

    # checkpoint: print the statistics
    # FIXME...
    

    print(' : Done')
    # done.
