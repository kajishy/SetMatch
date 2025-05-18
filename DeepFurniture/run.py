import tensorflow as tf
import matplotlib.pylab as plt
import os
import numpy as np
import pdb
import copy
import pickle
import sys
import argparse
import make_dataset as data
#import save_image as image
sys.path.insert(0, "../")
#import models_point_nets as models

from tensorflow.keras.optimizers import Adam
import util

import pdb
import time
import importlib
#----------------------------
# 画像の保存用
# 各種パスの設定
data_path = "pickle_data"  # test_fur.pklがあるディレクトリ
image_path = "/DeepFurniture/uncompressed_data/furnitures"  # 画像ファイルのあるディレクトリ
output_path = "output_images"  # 出力先ディレクトリ
#----------------------------
# set parameters

# get options
parser = util.parser_run()
args = parser.parse_args()
# 動的にモジュールをインポート
model_name = util.model_name(args.model)
models = importlib.import_module(model_name)

# インポートしたモジュールを使用
print(f"Loaded module: {models.__name__}")
# mode name
mode = util.mode_name(args.mode)
model_name = util.model_name(args.model)
#max number of items
max_item_num = 8
test_cand_num = 5
#max_data_num = 10000

# number of epochs
epochs = 200

# early stoppoing parameter
patience = 10

# batch size
batch_size = 15

# number of representive vectors
rep_vec_num = 1

# negative down sampling
is_neg_down_sample = True

# set random seed
os.environ['TF_DETERMINISTIC_OPS'] = '1'
np.random.seed(args.trial)
tf.random.set_seed(args.trial)

"""
# memory growth
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    ""
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device[0], True)
        print('{} memory growth: {}'.format(device, tf.config.experimental.get_memory_growth(device)))
    ""
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print('{} memory growth: {}'.format(physical_devices[0], tf.config.experimental.get_memory_growth(physical_devices[0])))
else:
    print("Not enough GPU hardware devices available")
"""
gpu_num = 1
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_visible_devices(gpus[gpu_num], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[gpu_num], True)
        print('{} memory growth: {}'.format(gpus[gpu_num], tf.config.experimental.get_memory_growth(gpus[gpu_num])))
    except RuntimeError as e:
        print(e)
#----------------------------

#----------------------------
# make Path

# make experiment path containing CNN and set-to-set model
experimentPath = 'experiment'
if not os.path.exists(experimentPath):
    os.makedirs(experimentPath)

# make set-to-set model path
modelPath = os.path.join(experimentPath, f'{mode}_{args.baseChn}')

if args.is_mixer:
    modelPath+=f'_mixer{args.item_perm_order}'

    if args.is_set_perm:
        modelPath+=f'_set_perm'
        modelPath+=f'_models_{model_name}'

else:
    if args.is_set_norm:
        modelPath+=f'_setnorm'
        modelPath+=f'_models_{model_name}'

    if args.is_cross_norm:
        modelPath+=f'_crossnorm'
        modelPath+=f'_models_{model_name}'

modelPath = os.path.join(modelPath,f"max_item_num{max_item_num}")
modelPath = os.path.join(modelPath,f"layer{args.num_layers}")
modelPath = os.path.join(modelPath,f"num_head{args.num_heads}")
modelPath = os.path.join(modelPath,f"{args.trial}")
if not os.path.exists(modelPath):
    path = os.path.join(modelPath,'model')
    os.makedirs(path)

    path = os.path.join(modelPath,'result')
    os.makedirs(path)
#----------------------------

#----------------------------
# make data
#pdb.set_trace()
train_generator = data.trainDataGenerator(batch_size=batch_size, max_item_num=max_item_num)
x_valid, x_size_valid, y_valid = train_generator.data_generation_val()

# set data generator for test
test_generator = data.testDataGenerator(cand_num=test_cand_num)
x_test = test_generator.x
x_size_test = test_generator.x_size
y_test = test_generator.y
id_test = test_generator.id
test_batch_size = test_generator.batch_grp_num

# load init seed vectors
#pdb.set_trace()
#----------------------------
#-----------------------------------------------------------------------

#----------------------------
# set-matching network
if args.model == 6:
    print("Attention")
    model = models.SMN(isCNN=False, is_final_linear=True, is_set_norm=args.is_set_norm, is_cross_norm=args.is_cross_norm, num_layers=args.num_layers, num_heads=args.num_heads, baseChn=args.baseChn, mode=mode, rep_vec_num=rep_vec_num, is_neg_down_sample=is_neg_down_sample)
elif args.model == 5:
    print("DuMLP")
    print("item_perm_order(isSoftmax):",args.item_perm_order)
    model = models.SMN(isSoftMax=args.item_perm_order, max_item_num=max_item_num, hidden_dim=args.hidden_dim, isCNN=False, is_final_linear=True, num_layers=args.num_layers, num_heads=args.num_heads, baseChn=args.baseChn, mode=mode, is_neg_down_sample=is_neg_down_sample)
else :#is_final_linear=false
    model = models.SMN(is_mixer=args.is_mixer, max_item_num=max_item_num, item_perm_order=args.item_perm_order, is_set_perm=args.is_set_perm, isCNN=False, is_final_linear=True, is_set_norm=args.is_set_norm, is_cross_norm=args.is_cross_norm, num_layers=args.num_layers, num_heads=args.num_heads, baseChn=args.baseChn, mode=mode, rep_vec_num=rep_vec_num, is_neg_down_sample=is_neg_down_sample)

checkpoint_path = os.path.join(modelPath,"model/cp.ckpt")
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_binary_accuracy', save_weights_only=True, mode='max', save_best_only=True, save_freq='epoch', verbose=1)
cp_earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_binary_accuracy', patience=patience, mode='max', min_delta=0.001, verbose=1)
result_path = os.path.join(modelPath,"result/result.pkl")
#pdb.set_trace()

if not os.path.exists(result_path) or 1:
    #pdb.set_trace()


    # setting training, loss, metric to model
    optimizer = Adam(learning_rate=0.002)#0.002, clipnorm=1.0
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['binary_accuracy'], run_eagerly=True)#"adam"

    batch_size_valid = batch_size
    valid_dataset = tf.data.Dataset.from_tensor_slices(((x_valid, x_size_valid), y_valid)).batch(batch_size_valid)
    #pdb.set_trace()
    
    # execute training
    #history = model.fit(train_generator, epochs=epochs, validation_data=((x_valid, x_size_valid), y_valid),
        #shuffle=True, callbacks=[cp_callback,cp_earlystopping])
    
    history = model.fit(train_generator, epochs=epochs, validation_data=valid_dataset,
        shuffle=True, callbacks=[cp_callback,cp_earlystopping])
    #history = model.fit(train_generator, epochs=epochs, validation_data=valid_data_set,
    #    shuffle=True, callbacks=[cp_callback,cp_earlystopping])#gradient_logger,

    # accuracy and loss
    acc = history.history['binary_accuracy']
    val_acc = history.history['val_binary_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # plot loss & acc
    util.plotLossACC(modelPath,loss,val_loss,acc,val_acc)

    # dump to pickle
    with open(result_path,'wb') as fp:
        pickle.dump(acc,fp)
        pickle.dump(val_acc,fp)
        pickle.dump(loss,fp)
        pickle.dump(val_loss,fp)
else:
    # load trained parameters
    print("load models")
    model.load_weights(checkpoint_path)
#----------------------------

#---------------------------------
# calc test loss and accuracy, and save to pickle
test_loss_path = os.path.join(modelPath, "result/test_loss_acc.txt")
if not os.path.exists(test_loss_path) or 1:
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['binary_accuracy'],run_eagerly=True)
    #pdb.set_trace()
    test_loss, test_acc = model.evaluate((x_test,x_size_test),y_test,batch_size=test_batch_size,verbose=0)

    # compute cmc
    _, predSMN, _ , elapsed_time= model.predict((x_test, x_size_test), batch_size=test_batch_size, verbose=1)
    cmcs = util.calc_cmcs(predSMN, y_test, batch_size=test_batch_size)

    average_time = np.mean(elapsed_time)
    #pdb.set_trace()
    """
    set_id = test_generator.set_id
    image_folder = "/DeepFurniture/uncompressed_data/furnitures"
    scene_folder = "/DeepFurniture/uncompressed_data/scenes"
    if args.model == 4 or args.model == 6:
        model_id = mode
    else :
        model_id = model_name
    image.visualize_batches(predSMN,id_test,set_id,model_id,image_folder,scene_folder, "output_images/4_17")
    """
    #total_time = np.sum(elapsed_time)
    model.summary()

    with open(test_loss_path,'w') as fp:
        fp.write('test loss:' + str(test_loss) + '\n')
        fp.write('test accuracy:' + str(test_acc) + '\n')
        fp.write('average time:' + str(average_time) + '\n')
        #fp.write('total time:' + str(total_time) + '\n')
        fp.write('test cmc:' + str(cmcs) + '\n')

    path = os.path.join(modelPath, "result/test_loss_acc.pkl")
    with open(path,'wb') as fp:
        pickle.dump(test_loss,fp)
        pickle.dump(test_acc,fp)
        pickle.dump(cmcs,fp)
#---------------------------------
