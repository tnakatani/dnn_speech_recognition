# DNN Speech Recognizer Project

_Note: This project was completed as part 3 of the Udacity NLP course._

In this notebook, you will build a deep neural network that functions as part of an end-to-end automatic speech recognition (ASR) pipeline!

We begin by investigating the [LibriSpeech dataset](http://www.openslr.org/12/) that will be used to train and evaluate your models. Your algorithm will first convert any raw audio to feature representations that are commonly used for ASR. You will then move on to building neural networks that can map these audio features to transcribed text. After learning about the basic types of layers that are often used for deep learning-based approaches to ASR, you will engage in your own investigations by creating and testing your own state-of-the-art models. Throughout the notebook, we provide recommended research papers for additional reading and links to GitHub repositories with interesting implementations.

## Tasks
The tasks for this project are outlined in the `vui_notebook.ipynb` in three steps. Follow all the instructions, which include implementing code in `sample_models.py`, answering questions, and providing results. The following list is a summary of the required tasks.


### Step 1 - Feature Extraction

- Execute all code cells to extract features from raw audio

### Step 2 - Acoustic Model

- Implement the code for Models 1, 2, 3, and 4 in sample_models.py
- Train Models 0, 1, 2, 3, 4 in the notebook
- Execute the comparison code in the notebook
- Answer Question 1 in the notebook regarding the comparison
- Implement the code for the Final Model in sample_models.py
- Train the Final Model in the notebook
- Answer Question 2 in the notebook regarding your final model

### Step 3 - Decoder

- Execute the prediction code in the notebook

--- 


# Project Notes

- The code was written and run on the Udacity workspace with GPU mode enabled.
- Module versions were as follows:
```
python:     3.6.3
keras:      2.0.9
tensorflow: 1.3.0
numpy:      1.12.1
```

## Model Evaluation


### Model Summaries

Model summaries for each implementation (Model 1~4 + Final Model) are listed below.  Code implementations can be viewed in [sample_models.py](./sample_models.py)

### Model 1 - RNN + TimeDistributed Dense
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
the_input (InputLayer)       (None, None, 13)          0         
_________________________________________________________________
rnn (GRU)                    (None, None, 200)         128400    
_________________________________________________________________
batch_normalization_5 (Batch (None, None, 200)         800       
_________________________________________________________________
time_distributed_5 (TimeDist (None, None, 29)          5829      
_________________________________________________________________
softmax (Activation)         (None, None, 29)          0         
=================================================================
Total params: 135,029
Trainable params: 134,629
Non-trainable params: 400
_________________________________________________________________
None
```

#### Model 2 - CNN + RNN + TimeDistributed Dense
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
the_input (InputLayer)       (None, None, 13)          0         
_________________________________________________________________
conv1d (Conv1D)              (None, None, 200)         28800     
_________________________________________________________________
bn_conv_1d_1 (BatchNormaliza (None, None, 200)         800       
_________________________________________________________________
rnn (SimpleRNN)              (None, None, 200)         80200     
_________________________________________________________________
bn_conv_1d_2 (BatchNormaliza (None, None, 200)         800       
_________________________________________________________________
time_distributed_1 (TimeDist (None, None, 29)          5829      
_________________________________________________________________
softmax (Activation)         (None, None, 29)          0         
=================================================================
Total params: 116,429
Trainable params: 115,629
Non-trainable params: 800
_________________________________________________________________
None
```

#### Model 3 - Deeper RNN + TimeDistributed Dense
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
the_input (InputLayer)       (None, None, 13)          0         
_________________________________________________________________
hidden_1 (GRU)               (None, None, 200)         128400    
_________________________________________________________________
bn_1 (BatchNormalization)    (None, None, 200)         800       
_________________________________________________________________
hidden_2 (GRU)               (None, None, 200)         240600    
_________________________________________________________________
bn_2 (BatchNormalization)    (None, None, 200)         800       
_________________________________________________________________
time_distributed_6 (TimeDist (None, None, 29)          5829      
_________________________________________________________________
softmax (Activation)         (None, None, 29)          0         
=================================================================
Total params: 376,429
Trainable params: 375,629
Non-trainable params: 800
_________________________________________________________________
None
```

#### Model 4 - Bidirectional RNN + TimeDistributed Dense
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
the_input (InputLayer)       (None, None, 13)          0         
_________________________________________________________________
bidirectional_1 (Bidirection (None, None, 400)         342400    
_________________________________________________________________
bnorm (BatchNormalization)   (None, None, 400)         1600      
_________________________________________________________________
time_distributed_1 (TimeDist (None, None, 29)          11629     
_________________________________________________________________
softmax (Activation)         (None, None, 29)          0         
=================================================================
Total params: 355,629
Trainable params: 354,829
Non-trainable params: 800
_________________________________________________________________
None
```

#### Final Model - Bidirectional + Deeper RNN + TimeDistributed
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
the_input (InputLayer)       (None, None, 13)          0         
_________________________________________________________________
bidir_1 (Bidirectional)      (None, None, 400)         256800    
_________________________________________________________________
bnorm_1 (BatchNormalization) (None, None, 400)         1600      
_________________________________________________________________
bidir_2 (Bidirectional)      (None, None, 400)         721200    
_________________________________________________________________
bnorm_2 (BatchNormalization) (None, None, 400)         1600      
_________________________________________________________________
bidir_3 (Bidirectional)      (None, None, 400)         721200    
_________________________________________________________________
time_distributed_2 (TimeDist (None, None, 29)          11629     
_________________________________________________________________
softmax (Activation)         (None, None, 29)          0         
=================================================================
Total params: 1,714,029
Trainable params: 1,712,429
Non-trainable params: 1,600
_________________________________________________________________
None
```


### Training / Validation Loss Results

| Model   | Loss (Train) | Loss (Validation) | 
|:--------|:-------------|:------------------|
| Model 0 | 777.9210  | 756.2926 |
| Model 1 | 113.2084  | 130.5225 | 
| Model 2 | 101.4944  | 130.7747 | 
| Model 3 | 101.0904  | 128.3843 | 
| Model 4 | 232.9241  | 222.5477 | 
| Final Model | 106.7501  | 103.9373 | 

#### Model 0: One RNN (GRU) layer.  
- Training loss flatlined by the second epoch at 778 with a performance much worse than the other models.  This model is underfitting by a large margin.

#### Model 1: RNN + Batch Normalization + TimeDistributed Dense layer. 
- Considerable improvement from Model 0.  This is attributed to the TimeDistributed Dense layer, as task requires a temporal element (processing streaming audio).  The TimeDistributed Layer allows the model to apply a layer to every temporal slice of an input.  
- Model 1 also has 36x the number of trainable parameters (3,741 vs. 134,629) compared to Model 0, which contributes to the model accuracy.

#### Model 2: CNN + RNN + TimeDistributed Dense.  
- Model 2 has 20,000 less trainable parameters compared to Model 1 (115,629 vs 134,629) but training loss improved by 10.3% relative by epoch 20.  
- On the other hand, Model 2's validation loss at epoch 20 worsened by 19% relative compared to Model 1 (130.5225 vs 130.7747).

#### Model 3: Deeper RNN + TimeDistributed Dense. 
- Model 3 exhibited the best training and validation loss out of all the models.
- Model 3 has 2.8x the number of parameters as Model 2 (375,629 vs 115,629), which likely accounts for the better performance.
- The increased number of parameters affected the trainig time, averaging 380 seconds per epoch.  This is approximately 3x the length of the Model 3, which had the shortest training time per epoch (~115 seconds per epoch) out of all the models.

#### Model 4: Bidirectional RNN + TimeDistributed Dense
- Even with the number parameters comparable to Model 3, validation loss worsened by 73% relative. Training loss also flatlined by epoch 3.
- Model 4 had the highest training time per epoch averaging approximately 470 seconds per epoch.  This is most likely explained by the bidrectional design of this model where 2 RNNs layers are running.
- The results of this model was surprising, as I thought bidrectional layers would be performant on speech recognition.  I would like to dive deeper into understanding why this model was not as performant as I expected.  Some hypotheses:
  - Perhaps applying more than 1 bidirectional layer would have improved the performance.
  - Since the RNN layer is bidirectional in this model, is the number of parameters need to be double the amount Model 3 to expect comparable performance?


#### Final Model: Deep Bidirectional RNN + Batch Normalization + TimeDistributed Dense

- Out of all the preliminary models, the multilayered RNN approached performed the best, hence I chose a deeper RNN model structure.
- I was still interested in using a bidirectional RNN since speech recognition is a task where making use of future context is useful. However, instead of using an LSTM layer in my previous model, I chose a GRU layer for efficiency.
- Each bidirectional layer is configured with a dropout parameter to prevent overfitting.
- Batch normalization was added after each layer. This normalizes the input layer which can accelerate the training efficiency of the model.
- TimeDistributed Dense layer was added after the RNN layers. This layer allows us to apply the previous RNN layer to every temporal slice of an input.- Training time was considerably longer than the other models, due to the dual implementation of a bidirectional RNN and deep layers.  Training time per epoch was approximately 2x that of the Model 4 (1100s/epoch vs 475/epoch)


### Notes from Project Review

From the reviewer on the poor performance of the bidirectional model:
> Excellent analysis! Probably one of the best! Here, the reason why Bidirectional Model didn't well can be attributed to various reasons, like initialization of random weights, sometimes the randoms weights initialized in the beginning cannot be very favorable, so just try to re-train the model. Another being is the insufficient parameters. Try to increase parameters and add more layers. Additionally, also try GRU and check how it works.

GRU vs CuDNNGRU:
> In order to better utilize GPU, use the CuDNNGRU Layer instead of a simple GRU Layer. Using GPU specific functions reduces the training time significantly and thus enabling you to try even more architectures. Note:- In this workspace, CuDNN layers won't work when using Bidirectional Layers. However, they may work in other workspaces like Kaggle, Colab etc., as they have access to higher versions of Tensorflow and Keras.

Critique on  `final_model` implementation:

```
def final_model(input_dim, units, dropout, recur_layers, output_dim=29):
    """ Build a deep network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    rnn = GRU(units, activation='relu', dropout=dropout,
              return_sequences=True)
    for l in range(recur_layers):
        if l == 0:
            bidir_rnn = Bidirectional(rnn, name=f'bidir_{l + 1}')(input_data)
            bn_rnn = BatchNormalization(name=f'bnorm_{l + 1}')(bidir_rnn)
        else:
            bidir_rnn = Bidirectional(rnn, name=f'bidir_{l + 1}')(bn_rnn)
            bn_rnn = BatchNormalization(name=f'bnorm_{l + 1}')(bidir_rnn)
    # Dense layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)  # error here
    # Softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    # Specify model.output_length
    model.output_length = lambda x: x
    print(model.summary())
    return model
```

> You should pass the bn_rnn layer as an input layer as it is the immediate previous layer's output.

This fix has been resolved

