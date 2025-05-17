import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Layer, LayerNormalization
import sys
import pdb
import itertools
import time

#----------------------------

#----------------------------
# MLP  
class MLP(Layer):

    def __init__(self, hidden_dim=128, out_dim=64, isSoftMax=0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.isSoftMax = isSoftMax
        self.smallV = 1e-8

        self.linear = Dense(self.hidden_dim, activation="linear")
        self.linear2 = Dense(out_dim, activation="linear")
    
    def masked_softmax(self, x):
        # 0 value is treated as mask
        mask = tf.not_equal(x,0)
        x_exp = tf.where(mask,tf.exp(x-tf.reduce_max(x,axis=-1,keepdims=1)),tf.zeros_like(x))
        softmax = x_exp/(tf.reduce_sum(x_exp,axis=-1,keepdims=1) + 1e-10)

        return softmax
        

    def call(self, x):
        #pdb.set_trace()
        if tf.rank(x) == 4:
            mask = tf.tile(tf.cast(tf.reduce_sum(tf.cast(x!=0,float),axis=-1,keepdims=1)!=0,float),[1,1,1,self.out_dim])
        x = self.linear(x)
        x = tfa.activations.gelu(x)
        x = self.linear2(x)
        if tf.rank(x) == 4:
            x = x*mask

        if self.isSoftMax==1:
            x = self.masked_softmax(x)

        #x = tf.nn.gelu(x)
        
        return x
#----------------------------
   
#----------------------------
class mixerLayer(Layer):
    def __init__(self, item_num=5, item_dim=64, hidden_dim=128, isSoftMax=0):
        super().__init__()
        self.item_num = item_num
        self.isSoftMax = isSoftMax
        self.MLP_dim = 128
        self.MLP_channel = MLP(self.MLP_dim, out_dim=item_dim)
        self.MLP_phi1 = MLP(self.MLP_dim, out_dim=item_dim)
        self.MLP_phi2 = MLP(self.MLP_dim, out_dim=hidden_dim, isSoftMax=isSoftMax)
        self.MLP_phi3 = MLP(32, out_dim=1)
        self.MLP_phi3_liner = Dense(1, activation="linear")
        #self.temperature = 2.0
        self.norm_item = LayerNormalization()
        self.norm_item2 = LayerNormalization()
        self.norm_channel = LayerNormalization()

    def call(self, x, x_size):
        #pdb.set_trace()


        #pdb.set_trace()
        shape = tf.shape(x)
        mask = tf.tile(tf.cast(tf.reduce_sum(tf.cast(x!=0,float),axis=-1,keepdims=1)!=0,float),[1,1,1,shape[-1]])
        x_orig = x


        """
        #pdb.set_trace()
        x = self.norm_item(x)
        x = x*mask
        x_phi1 = self.MLP_phi1(x) #phi1 [nset,nset,n,itemdim] -> [nset,nset,n,itemdim]
        x_phi2 = self.MLP_phi2(x) #phi2 [nset,nset,n,itemdim] -> [nset,nset,n,h]
        #x = tf.matmul(self.norm_item(tf.transpose(x,[0,1,3,2])),x_phi)
        x_phi1 = x_phi1+x_orig
        x = tf.matmul(tf.transpose(self.norm_item2(x_phi1)*mask,[0,1,3,2]),x_phi2)
        #x = tf.matmul(tf.transpose(x_phi1,[0,1,3,2]),x_phi2)
        #x = self.norm_item2(x)
        #x = self.MLP_phi3(x) #phi3 [nset,nset,itemdim,h] -> [nset,nset,itemdim,1] 2 layeres
        x = tfa.activations.gelu(x)
        x = self.MLP_phi3_liner(x) #phi3 [nset,nset,itemdim,h] -> [nset,nset,itemdim,1] dence
        x = tf.tile(x, [1, 1, 1, self.item_num])
        x = tf.transpose(x,[0,1,3,2])
        """

        #pdb.set_trace()
        x_phi1 = self.MLP_phi1(self.norm_item(x)*mask) #phi1 [nset,nset,n,itemdim] -> [nset,nset,n,itemdim]
        #x_phi1 = x_phi1+x_orig
        #x [nset,nset,n,itemdim] -> [nset,nset,n,n,itemdim]
        x = tf.tile(tf.expand_dims(x_phi1, axis=-2), [1, 1, 1, self.item_num, 1]) - tf.tile(tf.expand_dims(x, axis=-3), [1, 1, self.item_num, 1, 1])
        x_phi2 = self.MLP_phi2(x) #phi2 [nset,nset,n,n,itemdim] -> [nset,nset,n,n,h]
        
        x_phi1 = tf.tile(tf.expand_dims(x_phi1, axis=-3), [1, 1, self.item_num, 1, 1])#phi1 [nset,nset,n,itemdim] -> [nset,nset,n,n,itemdim]
        x = tf.matmul(tf.transpose(x_phi1,[0,1,2,4,3]),x_phi2)
        #x = self.MLP_phi3(x) #phi3 [nset,nset,n,itemdim,h] -> [nset,nset,n,itemdim,1] 2 layeres
        x = tfa.activations.gelu(x)
        x = self.MLP_phi3_liner(x) #phi3 [nset,nset,n,itemdim,h] -> [nset,nset,n,itemdim,1] dence
        x = tf.squeeze(x, axis=-1)

        x = x + x_orig
        #x += x_orig + x_phi1
        #x += x_phi1
        x = x*mask


        return x
#----------------------------

#----------------------------
# multi-head CS function to make cros-set matching score map
class cross_set_score(Layer):
    def __init__(self, head_size=20, num_heads=2):
        super(cross_set_score, self).__init__()
        self.head_size = head_size
        self.num_heads = num_heads

        # multi-head linear function, l(x|W_0), l(x|W_1)...l(x|W_num_heads) for each item feature vector x.
        # one big linear function with weights of W_0, W_1, ..., W_num_heads outputs head_size*num_heads-dim vector
        self.linear = Dense(units=self.head_size*self.num_heads,use_bias=False)
        self.linear2 = Dense(1,use_bias=False)

    def call(self, x, nItem):
        nSet_x = tf.shape(x)[0]
        nSet_y = tf.shape(x)[1]
        nItemMax = tf.shape(x)[2]
        sqrt_head_size = tf.sqrt(tf.cast(self.head_size,tf.float32))
        
        # linear transofrmation from (nSet_x, nSet_y, nItemMax, Xdim) to (nSet_x, nSet_y, nItemMax, head_size*num_heads)
        x = self.linear(x)

        # reshape (nSet_x, nSet_y, nItemMax, head_size*num_heads) to (nSet_x, nSet_y, nItemMax, num_heads, head_size)
        # transpose (nSet_x, nSet_y, nItemMax, num_heads, head_size) to (nSet_x, nSet_y, num_heads, nItemMax, head_size)
        x = tf.transpose(tf.reshape(x,[nSet_x, nSet_y, nItemMax, self.num_heads, self.head_size]),[0,1,3,2,4])
        
        # compute inner products between all pairs of items with cross-set feature (cseft)
        # Between set #1 and set #2, cseft x[0,1] and x[1,0] are extracted to compute inner product when nItemMax=2
        # More generally, between set #i and set #j, cseft x[i,j] and x[j,i] are extracted.
        # Outputing (nSet_x, nSet_y, num_heads)-score map        
        scores = tf.stack(
            [[
                tf.reduce_sum(tf.reduce_sum(
                tf.keras.layers.ReLU()(tf.matmul(x[j,i],tf.transpose(x[i,j],[0,2,1]))/sqrt_head_size)
                ,axis=1),axis=1)/nItem[i]/nItem[j]
                for i in range(nSet_x)] for j in range(nSet_y)]
            )
            
        # linearly combine multi-head score maps (nSet_x, nSet_y, num_heads) to (nSet_x, nSet_y, 1)
        scores = self.linear2(scores)

        return scores
#----------------------------

#----------------------------
# CNN
class CNN(Model):
    def __init__(self, baseChn=32, class_num=2, num_conv_layers=3, max_channel_ratio=2, is_item_label=False):
        super(CNN, self).__init__()
        self.baseChn = baseChn
        self.num_conv_layers = num_conv_layers
        self.is_item_label = is_item_label

        self.convs = [tf.keras.layers.Conv2D(filters=baseChn*np.min([i+1,max_channel_ratio]), strides=(2,2), padding='same', kernel_size=(3,3), activation='relu', use_bias=False, name='class') for i in range(num_conv_layers)]
        self.globalpool = tf.keras.layers.GlobalAveragePooling2D()

        self.fc_cnn_final1 = Dense(baseChn, activation='relu', name='class')
        self.fc_cnn_final2 = Dense(class_num, activation='softmax', name='class')

    def call(self, x):
        x, x_size = x

        # reshape (nSet, nItemMax, H, W, C) to (nSet*nItemMax, H, W, C)
        shape = tf.shape(x)
        nSet = shape[0]
        nItemMax = shape[1]
        x = tf.reshape(x,[-1,shape[2],shape[3],shape[4]])
        debug = {}

        # CNN
        for i in range(self.num_conv_layers):
            x = self.convs[i](x)
        x = self.globalpool(x)
        
        # classificaiton of set
        if self.is_item_label:
            output = self.fc_cnn_final1(x)
        else:
            output = self.fc_cnn_final1(tf.reshape(x,[nSet,-1]))

        output = self.fc_cnn_final2(output)

        return x, output

    # remove zero padding item from label and pred to use loss calculation
    def remove_zero_padding(self,label,pred):
        bool_tensor = tf.not_equal(label,-1)
        return label[bool_tensor], pred[bool_tensor]

    # train step
    def train_step(self,data):
        x, y_true = data
        x, x_size = x

        with tf.GradientTape() as tape:
            # predict
            _, y_pred = self((x, x_size), training=True)

            if self.is_item_label:
                # resahpe (nSet, nItemMax) to (nSet * nItemMax)
                y_true = tf.reshape(y_true,[-1])

                # remove zero padding item
                y_true, y_pred = self.remove_zero_padding(y_true, y_pred)

            # loss
            loss = self.compiled_loss(y_true, y_pred, regularization_losses=self.losses)
     
        # train using gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        self.optimizer.apply_gradients(
            (grad, var)
            for (grad, var) in zip(gradients, trainable_vars)
            if grad is not None)

        # update metrics
        self.compiled_metrics.update_state(y_true, y_pred)

        # return metrics as dictionary
        return {m.name: m.result() for m in self.metrics}

    # test step
    def test_step(self, data):
        x, y_true = data
        x, x_size = x

        # predict
        _, y_pred = self((x, x_size), training=False)

        if self.is_item_label:
            # resahpe (nSet, nItemMax) to (nSet * nItemMax)
            y_true = tf.reshape(y_true,[-1])

            # remove zero padding item
            y_true, y_pred = self.remove_zero_padding(y_true, y_pred)        
        
        # loss
        self.compiled_loss(y_true, y_pred, regularization_losses=self.losses)

        # update metrics
        self.compiled_metrics.update_state(y_true, y_pred)

        # return metrics as dictionary
        return {m.name: m.result() for m in self.metrics}
#----------------------------

#----------------------------
# set matching network
class SMN(Model):
    def __init__(self,isSoftMax=0, isCNN=True, max_item_num=5, hidden_dim=0, is_final_linear=True, num_layers=1, num_heads=2, mode='MLP', baseChn=32, max_channel_ratio=2, is_neg_down_sample=False):
        super(SMN, self).__init__()
        self.isCNN = isCNN
        self.num_layers = num_layers
        self.mode = mode
        self.baseChn = baseChn
        self.is_final_linear = is_final_linear
        self.is_neg_down_sample = is_neg_down_sample
        self.isSoftMax = isSoftMax
        #self.hidden_dim = hidden_dim
        self.hidden_dim = 128

        if self.mode.find('setRepVec') > -1:
            max_item_num += 1

        #---------------------
        # cnn
        self.CNN = []
        self.fc_cnn_proj = Dense(baseChn*max_channel_ratio, activation=tfa.activations.gelu, use_bias=False, name='setmatching')
        #---------------------
        self.norm_1 = LayerNormalization()
        self.norm_2 = LayerNormalization()
        #---------------------
        # encoder
        self.self_mixer = [mixerLayer(item_num=max_item_num, item_dim=baseChn*max_channel_ratio, hidden_dim=num_heads, isSoftMax=isSoftMax) for i in range(num_layers)]
        #---------------------

        #---------------------
        # decoder        
        self.cross_mixer = [mixerLayer(item_num=max_item_num*2, item_dim=baseChn*max_channel_ratio, hidden_dim=num_heads, isSoftMax=isSoftMax) for i in range(num_layers)]
        #---------------------
     
        #---------------------
        # head network
        self.cross_set_score = cross_set_score(head_size=baseChn*max_channel_ratio, num_heads=num_heads)
        self.fc_final1 = Dense(baseChn, name='setmatching')
        self.fc_final2 = Dense(1, activation='sigmoid', name='setmatching')
        self.MLP_phi1 = MLP(hidden_dim=self.hidden_dim, out_dim=self.baseChn*max_channel_ratio)  # phi1
        self.MLP_phi2 = MLP(hidden_dim=self.hidden_dim, out_dim=num_heads, isSoftMax=isSoftMax)  # phi2
        #self.MLP_phi2 = MLP(hidden_dim=self.hidden_dim, out_dim=1, item_perm_order=self.item_perm_order)  # phi2
        self.MLP_reshape = MLP(hidden_dim=32, out_dim=1)  # MLP
        self.MLP_phi3_liner = Dense(1, activation="linear",name='phi3')
        #---------------------

    # compute score of set-pair using dot product
    def dot_set_score(self, x):
        """
        nSet_x, nSet_y, dim = x.shape
       
        score = tf.stack([[tf.tensordot(x[i,j],x[j,i],1) for i in range(nSet_x)] for j in range(nSet_y)])
        score = tf.expand_dims(score,-1)/tf.cast(dim,float)

        return score
        """
        nSet_x, nSet_y, dim = x.shape
        # score = tf.stack([[tf.tensordot(x[i,j],x[j,i],1) for i in range(nSet_x)] for j in range(nSet_y)]) #以前の方式
        # score = tf.random.uniform(shape=(nSet_x, nSet_y), minval=0, maxval=100, dtype=tf.float32) # ランダムな値を返す
        score = tf.einsum('ijk,jik->ij', x, x) #アインシュタイン記法
        score = tf.expand_dims(score,-1)/tf.cast(dim,float)
        return score

    def call(self, x):
        #pdb.set_trace()
        x, x_size = x

        debug = {}
        shape = tf.shape(x)
        nSet = shape[0]
        nItemMax = shape[1]

        #pdb.set_trace()

        # CNN
        if self.isCNN:
            x, predCNN = self.CNN((x,x_size),training=False)
        else:
            x = self.fc_cnn_proj(x)
            predCNN = []
        
        # reshape (nSet*nItemMax, D) to (nSet, nItemMax, D)
        x = tf.reshape(x,[nSet, nItemMax, -1])

        # reshape (nSet, nItemMax, D) -> (nSet, nSet, nItemMax, D)
        x = tf.tile(tf.expand_dims(x,1),[1,nSet,1,1])

        debug['x_cnn'] = x
        
        debug['x_encoder_layer_0'] = x

        x_2enc = x


        #---------------------
        # encoder self-mixer
        #pdb.set_trace()        
        for i in range(self.num_layers):
            x = self.self_mixer[i](x,x_size)
            debug[f'x_encoder_layer_{i+1}'] = x    


        x_enc = x
        #---------------------
        #pdb.set_trace()

        #---------------------
        # decoder cross-mixer
        debug[f'x_decoder_layer_0'] = x
        #pdb.set_trace()

        x = tf.concat([x,tf.transpose(x,[1,0,2,3])],axis=2)

        for i in range(self.num_layers):
            x = self.cross_mixer[i](x,x_size)
            debug[f'x_decoder_layer_{i+1}'] = x[:,:,:nItemMax]

        x = x[:,:,:nItemMax]
        x_dec = x
        #---------------------

        #---------------------
        # calculation of score
        if self.mode=='CSS':
            score = self.cross_set_score(x,x_size)   #(nSet,nSet,1)

        elif self.mode=='MLP':
            #pdb.set_trace()
            mask = tf.tile(tf.cast(tf.reduce_sum(tf.cast(x!=0,float),axis=-1,keepdims=1)!=0,float),[1,1,1,x.shape[-1]])
            x = self.norm_1(x)
            x = x*mask
            phi1 = self.MLP_phi1(x) #(nSet,nSet,nItemMax,D) -> (nSet,nSet,nItemMax,D_1)
            phi2 = self.MLP_phi2(x) #(nSet,nSet,nItemMax,D) -> (nSet,nSet,nItemMax,D_2) 
            #phi1 = self.norm_2(phi1+x)*mask
            x = tf.matmul(tf.transpose(phi1,[0,1,3,2]),phi2) #(nSet,nSet,nItemMax,D_1),(nSet,nSet,nItemMax,D_2) -> (nSet,nSet,D_1,D_2) 
            #x = tf.reshape(x,[x.shape[0],x.shape[1],1,x.shape[2]*x.shape[3]]) #(nSet,nSet,D_1,D_2) -> (nSet,nSet,1,D_1*D_2)
            #x = self.norm_2(x)
            """
            print("LNgamma:",self.norm_1.gamma.numpy())
            print("phi1 mean:", tf.reduce_mean(phi1).numpy())
            print("mlp24 std:", tf.math.reduce_std(phi1).numpy())
            print("mlp24 min/max:", tf.reduce_min(phi1).numpy(), tf.reduce_max(phi1).numpy())

            print("phi2 mean:", tf.reduce_mean(phi2).numpy())
            print("mlp24 std:", tf.math.reduce_std(phi2).numpy())
            print("mlp24 min/max:", tf.reduce_min(phi2).numpy(), tf.reduce_max(phi2).numpy())
            """
            #x = self.MLP_reshape(x) #(nSet,nSet,D_1,D_2) -> (nSet,nSet,D_1,1)
            x = tfa.activations.gelu(x)
            x = self.MLP_phi3_liner(x)
            x_proj = tf.transpose(x,[0,1,3,2])
            #x_proj = self.norm_2(x_proj)
            x_rep = x_proj[:,:,0,:] #(nSet,nSet,1,D) -> (nSet,nSet,D)
            score = self.dot_set_score(x_rep)   

        elif self.mode=='maxPooling':#sumpooling
            # zero-padding mask
            mask = tf.tile(tf.cast(tf.reduce_sum(tf.cast(x!=0,float),axis=-1,keepdims=1)!=0,float),[1,1,1,x.shape[-1]])
            x = self.norm_1(x)
            x = x*mask
            x_rep = tf.reduce_sum(x,axis=2)   #(nSet,nSet,nItemMax,D) -> (nSet,nSet,D)
            #x_rep = self.MLP_channel(x_rep)
            """
            shape = tf.shape(x)
            mask = tf.tile(tf.reduce_sum(x,axis=-1,keepdims=1)!=0,[1,1,1,shape[-1]])

            x_inf = tf.where(mask,x,tf.ones_like(x)*-np.inf)
            x_rep = tf.reduce_max(x,axis=2)   #(nSet,nSet,nItemMax,D) -> (nSet,nSet,D)
            """

            score = self.dot_set_score(x_rep)


        #score = tf.nn.sigmoid(score)
        debug['score'] = score
        
        # linearly convert matching-score to class-score
        size_d = tf.shape(score)[2]

        if self.is_final_linear:
            predSMN = self.fc_final2(tf.reshape(score,[-1,size_d]))
        else:
            #pdb.set_trace()
            fc_final1 = self.fc_final1(tf.reshape(score,[-1,size_d]))
            predSMN = self.fc_final2(fc_final1)
        #---------------------

        return predCNN, predSMN, debug

    # convert class labels to cross-set label（if the class-labels are same, 1, otherwise 0)
    def cross_set_label(self, y):
        # rows of table
        y_rows = tf.tile(tf.expand_dims(y,-1),[1,tf.shape(y)[0]])
        # cols of table       
        y_cols = tf.tile(tf.transpose(tf.expand_dims(y,-1)),[tf.shape(y)[0],1])

        # if the class-labels are same, 1, otherwise 0
        labels = tf.cast(y_rows == y_cols, float)         
        return labels

    def toBinaryLabel(self,y):
        dNum = tf.shape(y)[0]
        y = tf.map_fn(fn=lambda x:0 if tf.less(x,0.5) else 1, elems=tf.reshape(y,-1))

        return tf.reshape(y,[dNum,-1])

    def neg_down_sampling(self, y_true, y_pred):
        # split to positive or negative data
        mask_pos = tf.not_equal(y_true,0)
        mask_neg = tf.not_equal(y_true,1)
        
        # number of pos and neg
        num_pos = tf.reduce_sum(tf.cast(mask_pos,tf.int32))
        num_neg = tf.reduce_sum(tf.cast(mask_neg,tf.int32))
        
        # split
        y_true_pos = tf.boolean_mask(y_true,mask_pos)
        y_pred_pos = tf.boolean_mask(y_pred,mask_pos)
        y_true_neg = tf.boolean_mask(y_true,mask_neg)
        y_pred_neg = tf.boolean_mask(y_pred,mask_neg)

        # select neg data
        # select neg data
        thre = tf.cast(1.0-num_pos/num_neg,float)
        mask_neg_thre = tf.greater(tf.random.uniform([num_neg]),thre)
        y_true_neg = tf.boolean_mask(y_true_neg,mask_neg_thre)
        y_pred_neg = tf.boolean_mask(y_pred_neg,mask_neg_thre)

        # concat
        y_true = tf.concat([y_true_pos,y_true_neg],axis=0)
        y_pred = tf.concat([y_pred_pos,y_pred_neg],axis=0)

        return y_true, y_pred

    # train step
    def train_step(self,data):
        x, y_true = data
        x, x_size = x

        with tf.GradientTape() as tape:
            # predict
            predCNN, predSMN, debug = self((x, x_size), training=True)
            
            y_pred = predSMN

            # convert to cross-set label
            y_true = self.cross_set_label(y_true)
            y_true = tf.reshape(y_true,-1)

            # mask for the pair of same sets
            mask = tf.not_equal(tf.reshape(tf.linalg.diag(tf.ones(x.shape[0])),-1),1)
            y_true = tf.boolean_mask(y_true, mask)
            y_pred = tf.boolean_mask(y_pred, mask)
            

            # down sampling
            if self.is_neg_down_sample:
                y_true, y_pred = self.neg_down_sampling(y_true, y_pred)
                

            # loss
            loss = self.compiled_loss(y_true, y_pred, regularization_losses=self.losses)
     
        # train using gradients
        trainable_vars = self.trainable_variables

        # train parameters excepts for CNN
        trainable_vars = [v for v in trainable_vars if 'cnn' not in v.name]
        gradients = tape.gradient(loss, trainable_vars)

        self.optimizer.apply_gradients(
            (grad, var)
            for (grad, var) in zip(gradients, trainable_vars)
            if grad is not None)
        """

        # 出力: 重みごとの勾配を表示
        for var, grad in zip(trainable_vars, gradients):
            if grad is not None:  # 勾配がNoneでない場合のみ出力
                grad_mean = tf.reduce_mean(grad).numpy()  # NumPy配列に変換して安全に表示
                print(f"Grad for {var.name}: {grad_mean}")
            else:
                print(f"Grad for {var.name}: None")
                """

        # update metrics
        self.compiled_metrics.update_state(y_true, y_pred)

        # return metrics as dictionary
        return {m.name: m.result() for m in self.metrics}

    # test step
    def test_step(self, data):
        x, y_true = data
        x , x_size = x

        # predict
        predCNN, predSMN, debug = self((x, x_size), training=False)
        y_pred = predSMN

        # convert to cross-set label
        y_true = self.cross_set_label(y_true)
        y_true = tf.reshape(y_true,-1)

        # mask for the pair of same sets
        mask = tf.not_equal(tf.reshape(tf.linalg.diag(tf.ones(x.shape[0])),-1),1)
        y_true = tf.boolean_mask(y_true, mask)
        y_pred = tf.boolean_mask(y_pred, mask)

        # down sampling
        if self.is_neg_down_sample:
            y_true, y_pred = self.neg_down_sampling(y_true, y_pred)

        # loss
        self.compiled_loss(y_true, y_pred, regularization_losses=self.losses)

        # update metrics
        self.compiled_metrics.update_state(y_true, y_pred)

        # return metrics as dictionary
        return {m.name: m.result() for m in self.metrics}

    # predict step
    def predict_step(self,data):
        #pdb.set_trace()
        batch_data = data[0]
        x, x_size = batch_data
        
        start_time = time.time()  # 開始時刻

        # predict
        predCNN, predSMN, debug = self((x, x_size), training=False)

        end_time = time.time()  # 終了時刻
        elapsed_time = end_time - start_time  # 所要時間

        return predCNN, predSMN, debug, elapsed_time
#----------------------------

