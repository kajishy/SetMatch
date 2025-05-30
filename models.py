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
# MLP  
class MLP(Layer):
    def __init__(self, hidden_dim=128, out_dim=64, is_SoftMax=0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.is_SoftMax = is_SoftMax
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

        if self.is_SoftMax==1:
            x = self.masked_softmax(x)

        #x = tf.nn.gelu(x)
        
        return x
#----------------------------

#----------------------------
class mixerLayer(Layer):
    def __init__(self, item_num=5, item_dim=64, hidden_dim=128, is_SoftMax=0, is_set_perm=False):
        super().__init__()
        self.item_num = item_num
        self.is_SoftMax = is_SoftMax
        self.MLP_channel = MLP(item_dim*2, out_dim=item_dim)
        self.MLP1 = MLP(item_dim*2, out_dim=item_dim)
        self.MLP2 = MLP(item_dim*2, out_dim=hidden_dim, is_SoftMax=is_SoftMax)
        self.MLP3 = MLP(item_dim/2, out_dim=1)
        self.MLP3_linear = Dense(1, activation="linear")
        if is_set_perm:
            self.norm_item = layer_normalization(is_set_norm=True, is_cross_norm=True)
        else:
            self.norm_item = layer_normalization(is_set_norm=True)

    def call(self, x, x_size):
        #pdb.set_trace()
        shape = tf.shape(x)
        mask = tf.tile(tf.cast(tf.reduce_sum(tf.cast(x!=0,float),axis=-1,keepdims=1)!=0,float),[1,1,1,shape[-1]])
        x_orig = x

        x = self.norm_item(x,x_size)*mask
        x_mlp1 = self.MLP1(x) #x_mlp1 [nset,nset,n,itemdim] -> [nset,nset,n,itemdim]
        x_diff = tf.tile(tf.expand_dims(x_mlp1, axis=-2), [1, 1, 1, self.item_num, 1]) - tf.tile(tf.expand_dims(x, axis=-3), [1, 1, self.item_num, 1, 1])
        x_diff_mlp = self.MLP2(x_diff) #x_mlp2 [nset,nset,n,n,itemdim] -> [nset,nset,n,n,h]

        x = tf.matmul(tf.transpose(x_diff_mlp,[0,1,3,4,2]),tf.tile(tf.expand_dims(x_mlp1, axis=-3), [1, 1, self.item_num, 1, 1]))

        #x = tfa.activations.gelu(x)
        x = self.MLP3_linear(tf.transpose(x,[0,1,2,4,3])) #phi3 [nset,nset,n,itemdim,h] -> [nset,nset,n,itemdim,1] dence
        x = tf.squeeze(x, axis=-1)

        x = x + x_orig
        x = x*mask

        return x
#----------------------------

#----------------------------
# normalization
class layer_normalization(Layer):
    def __init__(self, epsilon=1e-3, is_set_norm=False, is_cross_norm=False):
        super(layer_normalization, self).__init__()
        self.epsilon = epsilon
        self.is_cross_norm = is_cross_norm
        self.is_set_norm = is_set_norm

    def call(self, x, x_size):
        smallV = 1e-8
        if self.is_set_norm:
            if self.is_cross_norm:
                x = tf.concat([tf.transpose(x,[1,0,2,3]),x], axis=2)
                x_size=tf.expand_dims(x_size,-1)
                x_size_tile=x_size+tf.transpose(x_size)
            else:        
                shape = tf.shape(x)
                x_size_tile = tf.tile(tf.expand_dims(x_size,1),[1,shape[1]])
            #x_size_tile = tf.cast(x_size_tile, tf.float32)#念のため
            # change shape        
            shape = tf.shape(x)
            x_reshape = tf.reshape(x,[shape[0],shape[1],-1])

            # zero-padding mask
            mask = tf.reshape(tf.tile(tf.cast(tf.reduce_sum(tf.cast(x!=0,float),axis=-1,keepdims=1)!=0,float),[1,1,1,shape[-1]]),[shape[0],shape[1],-1])

            # mean and std of set
            mean_set = tf.reduce_sum(x_reshape,-1)/(x_size_tile*tf.cast(shape[-1],float))
            diff = x_reshape-tf.tile(tf.expand_dims(mean_set,-1),[1,1,shape[2]*shape[3]])
            std_set = tf.sqrt(tf.reduce_sum(tf.square(diff)*mask,-1)/(x_size_tile*tf.cast(shape[-1],float)))
        
            # output
            output = diff/tf.tile(tf.expand_dims(std_set + smallV,-1),[1,1,shape[2]*shape[3]])*mask
            output = tf.reshape(output,[shape[0],shape[1],shape[2],shape[3]])

            if self.is_cross_norm:
                output = tf.split(output,2,axis=2)[0]
        else:
            shape = tf.shape(x)

            # mean and std of items
            mean = tf.reduce_mean(x, axis=-1, keepdims=True)
            std = tf.math.reduce_std(x, axis=-1, keepdims=True)
            norm = tf.divide((x - mean), std + self.epsilon)
            
            # zero-padding mask
            mask = tf.tile(tf.cast(tf.reduce_sum(tf.cast(x!=0,float),axis=-1,keepdims=1)!=0,float),[1,1,1,shape[-1]])

            # output
            output = tf.where(mask==1, norm, tf.zeros_like(x))

        return output
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
# self- and cross-set attention
class set_attention(Layer):
    def __init__(self, head_size=20, num_heads=2, activation="softmax", self_attention=False):
        super(set_attention, self).__init__()
        self.head_size = head_size
        self.num_heads = num_heads        
        self.activation = activation
        self.self_attention = self_attention
        self.pivot_cross = False
        self.rep_vec_num = 1

        # multi-head linear function, l(x|W_0), l(x|W_1)...l(x|W_num_heads) for each item feature vector x.
        # one big linear function with weights of W_0, W_1, ..., W_num_heads outputs head_size*num_heads-dim vector
        self.linearQ = Dense(units=self.head_size*self.num_heads, use_bias=False, name='set_attention')
        self.linearK = Dense(units=self.head_size*self.num_heads, use_bias=False, name='set_attention')
        self.linearV = Dense(units=self.head_size*self.num_heads, use_bias=False, name='set_attention')
        self.linearH = Dense(units=self.head_size, use_bias=False, name='set_attention')

    def call(self, x, y):
        # number of sets
        nSet_x = tf.shape(x)[0]
        nSet_y = tf.shape(y)[0]
        nItemMax_x = tf.shape(x)[2]
        nItemMax_y = tf.shape(y)[2]
        sqrt_head_size = tf.sqrt(tf.cast(self.head_size,tf.float32))

        if self.self_attention:
            x = tf.reshape(x,[-1, nItemMax_x, self.head_size])
            y = tf.reshape(y,[-1, nItemMax_y, self.head_size])

        else:   # cross-attention
            x = tf.reshape(tf.transpose(x,[1,0,2,3]),[-1, nItemMax_x, self.head_size])   # nSet*nSet: (x1, x2, x3, ..., x10, x1, x2, x3, ..., x10, ...)
            y = tf.reshape(y,[-1, nItemMax_y, self.head_size])   # nSet*nSet: (y1, y1, y1, ..., y2, y2, y2, ..., y10, y10, y10)      

        # input (nSet, nSet, nItemMax, dim)
        # linear transofrmation (nSet, nSet, nItemMax, head_size*num_heads)
        y_K = self.linearK(y)   # Key
        y_V = self.linearV(y)   # Value
        x = self.linearQ(x)     # Query

        if self.pivot_cross: # pivot-cross
            y_K = tf.concat([y_K, x],axis=1)   # Key
            y_V = tf.concat([y_V, x],axis=1)   # Value            
            nItemMax_y += nItemMax_x

        # reshape (nSet*nSet, nItemMax, num_heads*head_size) to (nSet*nSet, nItemMax, num_heads, head_size)
        # transpose (nSet*nSet, nItemMax, num_heads, head_size) to (nSet*nSet, num_heads, nItemMax, head_size)
        x = tf.transpose(tf.reshape(x,[-1, nItemMax_x, self.num_heads, self.head_size]),[0,2,1,3])
        y_K = tf.transpose(tf.reshape(y_K,[-1, nItemMax_y, self.num_heads, self.head_size]),[0,2,1,3])
        y_V = tf.transpose(tf.reshape(y_V,[-1, nItemMax_y, self.num_heads, self.head_size]),[0,2,1,3])

        # inner products between all pairs of items, outputing (nSet*nSet, num_heads, nItemMax_x, nItemMax_y)-score map    
        xy_K = tf.matmul(x,tf.transpose(y_K,[0,1,3,2]))/sqrt_head_size

        def masked_softmax(x):
            # 0 value is treated as mask
            mask = tf.not_equal(x,0)
            x_exp = tf.where(mask,tf.exp(x-tf.reduce_max(x,axis=-1,keepdims=1)),tf.zeros_like(x))
            softmax = x_exp/(tf.reduce_sum(x_exp,axis=-1,keepdims=1) + 1e-10)

            return softmax

        # normalized by softmax
        attention_weight = masked_softmax(xy_K)

        # computing weighted y_V, outputing (nSet*nSet, num_heads, nItemMax_x, head_size)
        weighted_y_Vs = tf.matmul(attention_weight, y_V)

        # reshape (nSet*nSet, num_heads, nItemMax_x, head_size) to (nSet*nSet*nItemMax_x, head_size*num_heads)
        weighted_y_Vs = tf.reshape(tf.transpose(weighted_y_Vs,[0,2,1,3]),[-1, nItemMax_x, self.num_heads*self.head_size])
        
        # combine multi-head to (nSet*nSet*nItemMax_x, head_size)
        output = self.linearH(weighted_y_Vs)

        if not self.self_attention:
            output = tf.transpose(tf.reshape(output,[nSet_x, nSet_y, nItemMax_x, self.head_size]),[1,0,2,3])

        else:    
            output = tf.reshape(output,[nSet_x, nSet_y, nItemMax_x, self.head_size])


        return output
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
    def __init__(self, is_SoftMax=0 , isCNN=True, max_item_num=5, item_perm_order=0, is_set_perm=1, is_set_norm=False, is_cross_norm=True, is_final_linear=True, num_layers=1, num_heads=2, head_mode='setRepVec_pivot', backbone_mode=1, baseChn=32, rep_vec_num=1, max_channel_ratio=2, is_neg_down_sample=False):
        super(SMN, self).__init__()
        self.isCNN = isCNN
        self.num_layers = num_layers
        self.backbone_mode = backbone_mode #0:Attention,1:Mixer
        self.head_mode = head_mode
        self.rep_vec_num = rep_vec_num
        self.baseChn = baseChn
        self.is_final_linear = is_final_linear
        self.is_neg_down_sample = is_neg_down_sample

        if self.head_mode.find('setRepVec') > -1:
            max_item_num += 1

        #---------------------
        # cnn
        self.CNN = []
        self.fc_cnn_proj = Dense(baseChn*max_channel_ratio, activation=tfa.activations.gelu, use_bias=False, name='setmatching_cnn')
        #---------------------
        
        #---------------------
        # encoder
        self.set_emb = self.add_weight(name='set_emb',shape=(1,1,self.rep_vec_num,baseChn*max_channel_ratio),trainable=True)
        self.self_mixer = [mixerLayer(item_num=max_item_num, item_dim=baseChn*max_channel_ratio, hidden_dim=num_heads, is_SoftMax=is_SoftMax) for i in range(num_layers)]
        self.self_attentions = [set_attention(head_size=baseChn*max_channel_ratio, num_heads=num_heads, self_attention=True) for i in range(num_layers)]
        self.layer_norms_enc1 = [layer_normalization(is_set_norm=is_set_norm) for i in range(num_layers)]
        self.layer_norms_enc2 = [layer_normalization(is_set_norm=is_set_norm) for i in range(num_layers)]
        self.fcs_enc = [Dense(baseChn*max_channel_ratio, activation=tfa.activations.gelu, use_bias=False, name='setmatching') for i in range(num_layers)]        
        #---------------------

        #---------------------
        # decoder
        self.cross_mixer = [mixerLayer(item_num=max_item_num*2, item_dim=baseChn*max_channel_ratio, hidden_dim=num_heads, is_SoftMax=is_SoftMax, is_set_perm=is_set_perm) for i in range(num_layers)]
        self.cross_attentions = [set_attention(head_size=baseChn*max_channel_ratio, num_heads=num_heads) for i in range(num_layers)]
        self.layer_norms_dec1 = [layer_normalization(is_set_norm=is_set_norm, is_cross_norm=is_cross_norm) for i in range(num_layers)]
        self.layer_norms_dec2 = [layer_normalization(is_set_norm=is_set_norm, is_cross_norm=is_cross_norm) for i in range(num_layers)]
        self.fcs_dec = [Dense(baseChn*max_channel_ratio, activation=tfa.activations.gelu, use_bias=False, name='setmatching') for i in range(num_layers)]
        #---------------------
     
        #---------------------
        # head network
        self.norm = LayerNormalization()
        self.cross_set_score = cross_set_score(head_size=baseChn*max_channel_ratio, num_heads=num_heads)
        self.pma = set_attention(head_size=baseChn*max_channel_ratio, num_heads=num_heads)  # poolingMA
        self.fc_final1 = Dense(baseChn, name='setmatching_fc1')
        self.fc_final2 = Dense(1, activation='sigmoid', name='setmatching_fc2')
        self.fc_proj = Dense(1, use_bias=False, name='projection')  # linear projection
        #self.MLP = MLP(hidden_dim=num_heads, out_dim=1, item_num=max_item_num)  # MLP
        self.MLP_phi1 = MLP(hidden_dim=baseChn*max_channel_ratio*2, out_dim=self.baseChn*max_channel_ratio)  # phi1
        self.MLP_phi2 = MLP(hidden_dim=baseChn*max_channel_ratio*2, out_dim=num_heads, is_SoftMax=item_perm_order)  # phi2
        #self.MLP_phi2 = MLP(hidden_dim=self.hidden_dim, out_dim=1, item_perm_order=self.item_perm_order)  # phi2
        self.MLP_3 = MLP(hidden_dim=baseChn*max_channel_ratio/2, out_dim=1)  # MLP
        self.MLP_phi3_liner = Dense(1, activation="linear",name='phi3')
        #---------------------

    # compute score of set-pair using dot product
    def dot_set_score(self, x):
        nSet_x, nSet_y, dim = x.shape
        """
       
        score = tf.stack([[tf.tensordot(x[i,j],x[j,i],1) for i in range(nSet_x)] for j in range(nSet_y)])
        score = tf.expand_dims(score,-1)/tf.cast(dim,float)

        return score
        """
        score = tf.einsum('ijk,jik->ij', x, x) #アインシュタイン記法
        score = tf.expand_dims(score,-1)/tf.cast(dim,float)
        return score

    def call(self, x):
        x, x_size = x

        debug = {}
        shape = tf.shape(x)
        nSet = shape[0]
        nItemMax = shape[1]

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

        # add_embedding
        x_orig = x
        if self.head_mode.find('setRepVec') > -1:
            set_emb_tile = tf.tile(self.set_emb, [nSet,nSet,1,1])
            x = tf.concat([set_emb_tile,x], axis=2)
            x_size += 1
            nItemMax += 1
        
        debug['x_encoder_layer_0'] = x

        x_2enc = x


        #---------------------
        #pdb.set_trace()
        # encoder
        if self.backbone_mode==1:   # self-mixer           
            for i in range(self.num_layers):
                x = self.self_mixer[i](x, x_size=x_size)
                debug[f'x_encoder_layer_{i+1}'] = x
        else:               # self-attention
            for i in range(self.num_layers):

                if self.head_mode.find('setRepVec') > -1:
                    self.self_attentions[i].rep_vec_num = self.rep_vec_num

                z = self.layer_norms_enc1[i](x,x_size)

                # input: (nSet, nSet, nItemMax, D), output:(nSet, nSet, nItemMax, D)
                z = self.self_attentions[i](z,z)
                x += z

                z = self.layer_norms_enc2[i](x,x_size)
                z = self.fcs_enc[i](z)
                x += z

                debug[f'x_encoder_layer_{i+1}'] = x
        x_enc = x
        #---------------------

        #---------------------
        # decoder
        debug[f'x_decoder_layer_0'] = x
        if self.backbone_mode==1:   # cross-mixer
            x = tf.concat([x,tf.transpose(x,[1,0,2,3])],axis=2)

            for i in range(self.num_layers):
                x = self.cross_mixer[i](x, x_size=x_size)
                debug[f'x_decoder_layer_{i+1}'] = x[:,:,:nItemMax]

            x = x[:,:,:nItemMax]
        else:               # cross-attention
            for i in range(self.num_layers):

                if self.head_mode.find('setRepVec') > -1:
                    self.cross_attentions[i].rep_vec_num = self.rep_vec_num            

                if self.head_mode == 'setRepVec_pivot': # Bi-PMA + pivot-cross
                    self.cross_attentions[i].pivot_cross = True

                z = self.layer_norms_dec1[i](x,x_size)

                # input: (nSet, nSet, nItemMax, D), output:(nSet, nSet, nItemMax, D)
                z = self.cross_attentions[i](z,z)
                x += z
        
                z = self.layer_norms_dec2[i](x,x_size)
                z = self.fcs_dec[i](z)
                x += z

                debug[f'x_decoder_layer_{i+1}'] = x
        x_dec = x
        #---------------------

        #---------------------
        #pdb.set_trace()
        # calculation of score
        if self.head_mode=='CSS':
            score = self.cross_set_score(x,x_size)   #(nSet,nSet,1)

        elif self.head_mode.find('setRepVec') > -1:    # representative vec
            x_rep = x[:,:,:self.rep_vec_num,:] #(nSet,nSet,nItemMax+1,D) -> (nSet,nSet,D)
            shape = x_rep.shape
            x_rep = tf.reshape(x_rep,[shape[0],shape[1],-1])

            score = self.dot_set_score(x_rep) #(nSet,nSet,D) -> (nSet, nSet)
        
        elif self.head_mode=='sumPooling':
            # zero-padding mask
            mask = tf.tile(tf.cast(tf.reduce_sum(tf.cast(x!=0,float),axis=-1,keepdims=1)!=0,float),[1,1,1,x.shape[-1]])
            x = self.norm(x)
            x = x*mask
            x_rep = tf.reduce_sum(x,axis=2)   #(nSet,nSet,nItemMax,D) -> (nSet,nSet,D)

            score = self.dot_set_score(x_rep)

        elif self.head_mode=='MLP':
            x_proj = self.MLP(tf.transpose(x,[0,1,3,2])) #(nSet,nSet,nItemMax,D) -> (nSet,nSet,D,nItemMax) -> FC -> (nSet,nSet,D,1)
            x_rep = x_proj[:,:,:,0] #(nSet,nSet,D,1) -> (nSet,nSet,D)
            score = self.dot_set_score(x_rep)

        elif self.head_mode=='dumlp':
            #pdb.set_trace()
            mask = tf.tile(tf.cast(tf.reduce_sum(tf.cast(x!=0,float),axis=-1,keepdims=1)!=0,float),[1,1,1,x.shape[-1]])
            x = self.norm(x)
            x = x*mask
            phi1 = self.MLP_phi1(x) #(nSet,nSet,nItemMax,D) -> (nSet,nSet,nItemMax,D_1)
            phi2 = self.MLP_phi2(x) #(nSet,nSet,nItemMax,D) -> (nSet,nSet,nItemMax,D_2) 
            #phi1 = self.norm_2(phi1+x)*mask
            x = tf.matmul(tf.transpose(phi1,[0,1,3,2]),phi2) #(nSet,nSet,nItemMax,D_1),(nSet,nSet,nItemMax,D_2) -> (nSet,nSet,D_1,D_2) 
            #x = tf.reshape(x,[x.shape[0],x.shape[1],1,x.shape[2]*x.shape[3]]) #(nSet,nSet,D_1,D_2) -> (nSet,nSet,1,D_1*D_2)
            #x = self.norm_2(x)
            #x = self.MLP_3(x) #(nSet,nSet,D_1,D_2) -> (nSet,nSet,D_1,1)
            x = tfa.activations.gelu(x)
            x = self.MLP_phi3_liner(x)
            x_proj = tf.transpose(x,[0,1,3,2])
            #x_proj = self.norm_2(x_proj)
            x_rep = x_proj[:,:,0,:] #(nSet,nSet,1,D) -> (nSet,nSet,D)
            score = self.dot_set_score(x_rep)         

        elif self.head_mode=='linearProj':
            x_proj = self.fc_proj(tf.transpose(x,[0,1,3,2])) #(nSet,nSet,nItemMax,D) -> (nSet,nSet,D,nItemMax) -> FC -> (nSet,nSet,D,1)
            x_rep = x_proj[:,:,:,0] #(nSet,nSet,D,1) -> (nSet,nSet,D)
            score = self.dot_set_score(x_rep)

        elif self.head_mode=='maxPooling':
            # zero-padding mask
            shape = tf.shape(x)
            mask = tf.tile(tf.reduce_sum(x,axis=-1,keepdims=1)!=0,[1,1,1,shape[-1]])

            x_inf = tf.where(mask,x,tf.ones_like(x)*-np.inf)
            x_rep = tf.reduce_max(x,axis=2)   #(nSet,nSet,nItemMax,D) -> (nSet,nSet,D)

            score = self.dot_set_score(x_rep)

   
        elif self.head_mode=='poolingMA':  # pooling by multihead attention            
            # create Seed Vector
            set_emb_tile = tf.tile(self.set_emb, [nSet,nSet,1,1])
            
            # PMA
            x_pma = self.pma(set_emb_tile,x) #(nSet,nSet,rep_vec_num,D), (nSet,nSet,nItemMax,D) -> (nSet,nSet,rep_vec_num,D)
            x_rep = x_pma[:,:,0,:]

            # calculate score
            score = self.dot_set_score(x_rep) #(nSet,nSet,D) -> (nSet,nSet)
            if np.isnan(np.max(score)):
                pdb.set_trace()

        debug['score'] = score
        
        # linearly convert matching-score to class-score
        size_d = tf.shape(score)[2]

        if self.is_final_linear:
            predSMN = self.fc_final2(tf.reshape(score,[-1,size_d]))
        else:
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
        batch_data = data[0]
        x, x_size = batch_data
        
        start_time = time.time()  # start_time

        # predict
        predCNN, predSMN, debug = self((x, x_size), training=False)

        end_time = time.time()  # end_time
        elapsed_time = end_time - start_time  # 所要時間

        return predCNN, predSMN, debug, elapsed_time
    """
    def predict_step(self,data):
        batch_data = data[0]
        x, x_size = batch_data
        
        # predict
        predCNN, predSMN, debug = self((x, x_size), training=False)

        return predCNN, predSMN, debug
        """
#----------------------------

