"""

Brains of the operation, runs tensorflow to determine a phonetic
spelling for any made up word


"""
#SAmple: https://github.com/Piasy/Udacity-DLND/blob/master/language-translation/dlnd_language_translation.ipynb
from distutils.version import LooseVersion
import warnings
import tensorflow as tf
from tensorflow.python.layers.core import Dense
import numpy as np
tf.reset_default_graph() #Clears the default graph stack and resets the global default graph.
sess = tf.InteractiveSession() #initializes a tensorflow session
data_file = open('cmudict.dict')


data=[]
l1=-1
l2=-1
for line in data_file:
    tok = line.split()  
    word = tok[0]
    if '(' in word:
        word=word[:word.index('(')]
    if len(word) > l1:
        l1 = len(word)
    phones = tok[1:]
    if len(phones) > l2:
        l2=len(phones)
    data.append(list([word,phones]))
print("Enc:",l1)
print("Dec:",l2)
data_file.close()

print(data[0])

f1 = open('src.txt',"w")
f2 = open('tar.txt',"w")
src__txt_to_int={}
src__int_to_txt={}
cnt=1
for pair in data:
    for char in pair[0]:
        f1.write(char+" ")
        if char not in src__txt_to_int:
            src__txt_to_int[char]=cnt
            src__int_to_txt[cnt]=char
            cnt+=1
    f1.write("\n")
    for phone in pair[1]:
        f2.write(phone+" ")
    f2.write("\n")
src__txt_to_int['<PAD>']=0
src__int_to_txt[0]='<PAD>'
src__txt_to_int['<UNK>']=cnt
src__int_to_txt[cnt]='<UNK>'
print("src__txt_to_int:",src__txt_to_int)
print("src__int_to_txt:",src__int_to_txt)


f1.close()
f2.close()


phonef = open('cmudict.symbols','r')
tar__txt_to_int={}
tar__int_to_txt={}
cnt=1
for phone in phonef:
    phone=phone.split()
    tar__txt_to_int[phone[0]]=cnt
    tar__int_to_txt[cnt]=phone[0]
    cnt+=1
tar__txt_to_int['<EOS>']=cnt
tar__int_to_txt[cnt]='<EOS>'
cnt+=1
tar__txt_to_int['<GO>']=cnt
tar__int_to_txt[cnt]='<GO>'
cnt+=1
tar__txt_to_int['<UNK>']=cnt
tar__int_to_txt[cnt]='<UNK>'
tar__txt_to_int['<PAD>']=0
tar__int_to_txt[0]='<PAD>'
phonef.close()
print("txt->int:",tar__txt_to_int)
print("int->txt:",tar__int_to_txt)


#Write new files in a way that tf can understand...
f1ints = open('int_src.txt','w')
f1 = open('src.txt',"r")
src_f=[]
tar_f=[]
for line in f1:
    temp=[]
    toks = line.split()
    for tok in toks:
        f1ints.write(str(src__txt_to_int[tok])+" ")
        temp.append(src__txt_to_int[tok])
    src_f.append(temp)
    f1ints.write("\n")
f2ints = open('int_tar.txt','w')

f2 = open('tar.txt',"r")
for line in f2:
    temp=[]
    toks = line.split()
    for tok in toks:
        if tok == '#':
            break
        f2ints.write(str(tar__txt_to_int[tok])+" ")
        temp.append(tar__txt_to_int[tok])
    temp.append(tar__txt_to_int['<EOS>'])
    f2ints.write(str(tar__txt_to_int['<EOS>'])+'\n')
    tar_f.append(temp)
f1.close()
f2.close()
f1ints.close()
f2ints.close()


'''
Preprocess finished...

Split cmudict.dict into source sequence (words are letters in this sequence(makes sense!?))
and target sequences (these are the phones) in two files:
src sequences: src.txt
tar sequences: tar.txt
'''
src_path='src.txt'
tar_path='tar.txt'

'''

!!!IMPORTANT INFORMATION!!!
datapoints with multi prounciations: src1 sequence = scr2 sequence but matching tar1 sequence != tar2 sequence
How does this affect learning?

datapoints with forgin pronunications simmilar
How does this affect learning?

should we just remove this points altogether?
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


Source:
text_to_int dict: src__txt_to_int
int_to_text dict: src__int_to_txt

Target:
text_to_int dict: tar__txt_to_int
int_to_tect dict: tar__int_to_txt


Used dictionaries to map sequences in a way that tf can understand:

Source_id'd: int_src.txt
Target_id'd: int_tar.txt
'''
print(src_f[:10])

'''
Now lets save this in a pickle file...
'''
#first get id's in a list:
src_ids=[]
tar_ids=[]
f1 = open('int_src.txt','r')
f2 = open('int_tar.txt','r')
for line in f1:
    toks = line.split()
    src_ids.append(toks)
for line in f2:
    toks = line.split()
    tar_ids.append(toks)
f1.close()
f2.close()
import pickle
with open('preprocess.p', 'wb') as out_file:
    pickle.dump((
            (src_f, tar_f),
            (src__txt_to_int, tar__txt_to_int),
            (src__int_to_txt, tar__int_to_txt)), out_file)

#From Sample XXXX
def model_inputs():
    
    """
    Create TF Placeholders for:
    input, 
    targets, 
    learning rate, 
    lengths of source 
    target sequences
    return Tuple (input, targets, learning rate, keep probability, target sequence length,
    max target sequence length, source sequence length)
    """
    
    # TODO: Implement Function
    inputs = tf.placeholder(tf.int32,[None,None],name="input")
    targets = tf.placeholder(tf.int32,[None,None])
    learning_rate = tf.placeholder(tf.float32)
    keep_prob = tf.placeholder(tf.float32,name="keep_prob")
    target_seq = tf.placeholder(tf.int32,[None],name="target_sequence_length")
    max_target = tf.reduce_max(target_seq,name="max_target_len")                               
    source_seq = tf.placeholder(tf.int32,[None],name="source_sequence_length")
    return (inputs, targets, learning_rate, keep_prob, target_seq, max_target, source_seq)


#From Sample XXXX
def process_decoder_input(target_data, tar__txt_to_int, batch_size):
    """
    Preprocess target data for encoding
    :param target_data: Target Placehoder
    :param target_vocab_to_int: Dictionary to go from the target words to an id
    :param batch_size: Batch Size
    :return: Preprocessed target data
    """
    # TODO: Implement Function
    cut_off = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
    dec_input = tf.concat([tf.fill([batch_size, 1], tar__txt_to_int['<GO>']), cut_off], 1)
    return dec_input

#From Sample XXXX
def encoding_layer(rnn_inputs, rnn_size, num_layers, keep_prob, 
                   source_sequence_length, source_vocab_size, 
                   encoding_embedding_size):
    """
    Create encoding layer
    :param rnn_inputs: Inputs for the RNN
    :param rnn_size: RNN Size
    :param num_layers: Number of layers
    :param keep_prob: Dropout keep probability
    :param source_sequence_length: a list of the lengths of each sequence in the batch
    :param source_vocab_size: vocabulary size of source data
    :param encoding_embedding_size: embedding size of source data
    :return: tuple (RNN output, RNN state)
    """
    # Encoder embedding
    enc_embed_input = tf.contrib.layers.embed_sequence(rnn_inputs, source_vocab_size, encoding_embedding_size)

    # RNN cell
    def make_cell(rnn_size,keep_prob):
        enc_cell = tf.contrib.rnn.LSTMCell(rnn_size,
                                           initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
        drop = tf.contrib.rnn.DropoutWrapper(enc_cell, output_keep_prob = keep_prob)
        return drop

    enc_cell = tf.contrib.rnn.MultiRNNCell([make_cell(rnn_size,keep_prob) for _ in range(num_layers)])
    
    enc_output, enc_state = tf.nn.dynamic_rnn(enc_cell, enc_embed_input, sequence_length=source_sequence_length, dtype=tf.float32)
    
    return enc_output, enc_state

#From Sample XXXX
def decoding_layer_train(encoder_state, dec_cell, dec_embed_input, 
                         target_sequence_length, max_summary_length, 
                         output_layer, keep_prob):
    """
    Create a decoding layer for training
    :param encoder_state: Encoder State
    :param dec_cell: Decoder RNN Cell
    :param dec_embed_input: Decoder embedded input
    :param target_sequence_length: The lengths of each sequence in the target batch
    :param max_summary_length: The length of the longest sequence in the batch
    :param output_layer: Function to apply the output layer
    :param keep_prob: Dropout keep probability
    :return: BasicDecoderOutput containing training logits and sample_id
    """
    # TODO: Implement Function
    training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=dec_embed_input,
                                                            sequence_length=target_sequence_length,
                                                            time_major=False)
    
    training_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                                           training_helper,
                                                           encoder_state,
                                                           output_layer) 
    
    BasicDecoderOutput = tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                                       impute_finished=True,
                                                                       maximum_iterations=max_summary_length)[0]
    

    
    return BasicDecoderOutput

#From Sample XXXX
def decoding_layer_infer(encoder_state, dec_cell, dec_embeddings, start_of_sequence_id,
                         end_of_sequence_id, max_target_sequence_length,
                         vocab_size, output_layer, batch_size, keep_prob):
    """
    Create a decoding layer for inference
    :param encoder_state: Encoder state
    :param dec_cell: Decoder RNN Cell
    :param dec_embeddings: Decoder embeddings
    :param start_of_sequence_id: GO ID
    :param end_of_sequence_id: EOS Id
    :param max_target_sequence_length: Maximum length of target sequences
    :param vocab_size: Size of decoder/target vocabulary
    :param decoding_scope: TenorFlow Variable Scope for decoding
    :param output_layer: Function to apply the output layer
    :param batch_size: Batch size
    :param keep_prob: Dropout keep probability
    :return: BasicDecoderOutput containing inference logits and sample_id
    """
    
    with tf.variable_scope("decode", reuse=True):
        start_tokens = tf.tile(tf.constant([start_of_sequence_id], dtype=tf.int32), [batch_size], name='start_tokens')
    
    
    # TODO: Implement Function
    inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(dec_embeddings,
                                                                start_tokens,
                                                                end_of_sequence_id)

    # Basic decoder
    inference_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                                    inference_helper,
                                                    encoder_state,
                                                    output_layer)

    # Perform dynamic decoding using the decoder
    BasicDecoderOutput = tf.contrib.seq2seq.dynamic_decode(inference_decoder,
                                                        impute_finished=True,
                                                        maximum_iterations=max_target_sequence_length)[0]
    return BasicDecoderOutput

#From Sample XXXX
def decoding_layer(dec_input, encoder_state,
                   target_sequence_length, max_target_sequence_length,
                   rnn_size,
                   num_layers, target_vocab_to_int, target_vocab_size,
                   batch_size, keep_prob, decoding_embedding_size):
    """
    Create decoding layer
    :param dec_input: Decoder input
    :param encoder_state: Encoder state
    :param target_sequence_length: The lengths of each sequence in the target batch
    :param max_target_sequence_length: Maximum length of target sequences
    :param rnn_size: RNN Size
    :param num_layers: Number of layers
    :param target_vocab_to_int: Dictionary to go from the target words to an id
    :param target_vocab_size: Size of target vocabulary
    :param batch_size: The size of the batch
    :param keep_prob: Dropout keep probability
    :param decoding_embedding_size: Decoding embedding size
    :return: Tuple of (Training BasicDecoderOutput, Inference BasicDecoderOutput)
    """
    # TODO: Implement Function
    dec_embeddings = tf.Variable(tf.random_uniform([target_vocab_size, decoding_embedding_size]))
    dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, dec_input)

    # 2. Construct the decoder cell
    def make_cell(rnn_size):
        dec_cell = tf.contrib.rnn.LSTMCell(rnn_size,
                                           initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
        return dec_cell

    rnnCell = tf.contrib.rnn.MultiRNNCell([make_cell(rnn_size) for _ in range(num_layers)])
    
    output_layer = Dense(target_vocab_size,
                         kernel_initializer = tf.truncated_normal_initializer(mean = 0.0, stddev=0.1))
    
    Training_BasicDecoderOutput = decoding_layer_train(encoder_state, rnnCell, dec_embed_input, target_sequence_length, max_target_sequence_length, output_layer, keep_prob)
    
    Inference_BasicDecoderOutput = decoding_layer_infer(encoder_state, rnnCell, dec_embeddings, target_vocab_to_int['<GO>'], target_vocab_to_int['<EOS>'], max_target_sequence_length, target_vocab_size, output_layer, batch_size, keep_prob)
    
    return Training_BasicDecoderOutput, Inference_BasicDecoderOutput

#From Sample XXXX
def seq2seq_model(input_data, target_data, keep_prob, batch_size,
                  source_sequence_length, target_sequence_length,
                  max_target_sentence_length,
                  source_vocab_size, target_vocab_size,
                  enc_embedding_size, dec_embedding_size,
                  rnn_size, num_layers, target_vocab_to_int):
    """
    Build the Sequence-to-Sequence part of the neural network
    :param input_data: Input placeholder
    :param target_data: Target placeholder
    :param keep_prob: Dropout keep probability placeholder
    :param batch_size: Batch Size
    :param source_sequence_length: Sequence Lengths of source sequences in the batch
    :param target_sequence_length: Sequence Lengths of target sequences in the batch
    :param source_vocab_size: Source vocabulary size
    :param target_vocab_size: Target vocabulary size
    :param enc_embedding_size: Decoder embedding size
    :param dec_embedding_size: Encoder embedding size
    :param rnn_size: RNN Size
    :param num_layers: Number of layers
    :param target_vocab_to_int: Dictionary to go from the target words to an id
    :return: Tuple of (Training BasicDecoderOutput, Inference BasicDecoderOutput)
    """
    # TODO: Implement Function
    
    _, enc_state= encoding_layer(input_data, rnn_size, num_layers, keep_prob,  source_sequence_length, source_vocab_size, enc_embedding_size)
    
    dec_input = process_decoder_input(target_data, target_vocab_to_int, batch_size)
    
    Training_BasicDecoderOutput, Inference_BasicDecoderOutput = decoding_layer(dec_input, enc_state, target_sequence_length, max_target_sentence_length, rnn_size, num_layers, target_vocab_to_int, target_vocab_size, batch_size, keep_prob, dec_embedding_size)
    
    return Training_BasicDecoderOutput, Inference_BasicDecoderOutput

# Number of Epochs
epochs = 5
# Batch Size
batch_size = 64
# RNN Size
rnn_size = 128
# Number of Layers
num_layers = 2
# Embedding Size
encoding_embedding_size = 30
decoding_embedding_size = 30
# Learning Rate
learning_rate = 0.001
# Dropout Keep Probability
keep_probability = 0.5
display_step = 10

#From Sample XXXX
save_path = 'checkpoints/dev'
with open('preprocess.p', mode='rb') as in_file:
    LOADED = pickle.load(in_file)
(source_int_text, target_int_text), (source_vocab_to_int, target_vocab_to_int), _ = LOADED
max_target_sentence_length = max([len(sentence) for sentence in source_int_text])

train_graph = tf.Graph()
with train_graph.as_default():
    input_data, targets, lr, keep_prob, target_sequence_length, max_target_sequence_length, source_sequence_length = model_inputs()

    #sequence_length = tf.placeholder_with_default(max_target_sentence_length, None, name='sequence_length')
    input_shape = tf.shape(input_data)

    train_logits, inference_logits = seq2seq_model(tf.reverse(input_data, [-1]),
                                                   targets,
                                                   keep_prob,
                                                   batch_size,
                                                   source_sequence_length,
                                                   target_sequence_length,
                                                   max_target_sequence_length,
                                                   len(source_vocab_to_int),
                                                   len(target_vocab_to_int),
                                                   encoding_embedding_size,
                                                   decoding_embedding_size,
                                                   rnn_size,
                                                   num_layers,
                                                   target_vocab_to_int)


    training_logits = tf.identity(train_logits.rnn_output, name='logits')
    inference_logits = tf.identity(inference_logits.sample_id, name='predictions')

    masks = tf.sequence_mask(target_sequence_length, max_target_sequence_length, dtype=tf.float32, name='masks')

    with tf.name_scope("optimization"):
        # Loss function
        cost = tf.contrib.seq2seq.sequence_loss(
            training_logits,
            targets,
            masks)

        # Optimizer
        optimizer = tf.train.AdamOptimizer(lr)

        # Gradient Clipping
        gradients = optimizer.compute_gradients(cost)
        capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients)


# In[19]:


#From Sample XXXX

#XXXXXX MODIFIED FOR OUT PROBLEM XXXXXXX#
def pad_sentence_batch(sentence_batch, pad_int):
    """Pad char seq with <PAD> so that each char sequence of a batch has the same length"""
    max_seq = max([len(seq) for seq in sentence_batch])
    padded_batch=[]
    for seq in sentence_batch:
        temp=[]
        for char in seq:
            temp.append(char)
        for x in range(len(temp),max_seq):
            temp.append(pad_int)
        padded_batch.append(temp)
    return padded_batch


def get_batches(sources, targets, batch_size, source_pad_int, target_pad_int):
    """Batch targets, sources, and the lengths of their sentences together"""
    
    for batch_i in range(0, len(sources)//batch_size):
        
        start_i = batch_i * batch_size

        # Slice the right amount for the batch
        sources_batch = sources[start_i:start_i + batch_size]
        targets_batch = targets[start_i:start_i + batch_size]

        # Pad
        pad_sources_batch = np.array(pad_sentence_batch(sources_batch, source_pad_int))
        pad_targets_batch = np.array(pad_sentence_batch(targets_batch, target_pad_int))

        # Need the lengths for the _lengths parameters
        pad_targets_lengths = []
        for target in pad_targets_batch:
            pad_targets_lengths.append(len(target))

        pad_source_lengths = []
        for source in pad_sources_batch:
            pad_source_lengths.append(len(source))

        yield pad_sources_batch, pad_targets_batch, pad_source_lengths, pad_targets_lengths


# In[20]:



"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
def get_accuracy(target, logits):
    """
    Calculate accuracy
    """
    max_seq = max(target.shape[1], logits.shape[1])
    if max_seq - target.shape[1]:
        target = np.pad(
            target,
            [(0,0),(0,max_seq - target.shape[1])],
            'constant')
    if max_seq - logits.shape[1]:
        logits = np.pad(
            logits,
            [(0,0),(0,max_seq - logits.shape[1])],
            'constant')

    return np.mean(np.equal(target, logits))

# Split data to training and validation sets
train_source = source_int_text[batch_size:]
train_target = target_int_text[batch_size:]
valid_source = source_int_text[:batch_size]
valid_target = target_int_text[:batch_size]
(valid_sources_batch, valid_targets_batch, valid_sources_lengths, valid_targets_lengths ) = next(get_batches(valid_source,valid_target,batch_size,source_vocab_to_int['<PAD>'],target_vocab_to_int['<PAD>']))


with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())

    for epoch_i in range(epochs):
        for batch_i, (source_batch, target_batch, sources_lengths, targets_lengths) in enumerate(
                get_batches(train_source, train_target, batch_size,
                            source_vocab_to_int['<PAD>'],
                            target_vocab_to_int['<PAD>'])):

            _, loss = sess.run(
                [train_op, cost],
                {input_data: source_batch,
                 targets: target_batch,
                 lr: learning_rate,
                 target_sequence_length: targets_lengths,
                 source_sequence_length: sources_lengths,
                 keep_prob: keep_probability})


            if batch_i % display_step == 0 and batch_i > 0:


                batch_train_logits = sess.run(
                    inference_logits,
                    {input_data: source_batch,
                     source_sequence_length: sources_lengths,
                     target_sequence_length: targets_lengths,
                     keep_prob: 1.0})


                batch_valid_logits = sess.run(
                    inference_logits,
                    {input_data: valid_sources_batch,
                     source_sequence_length: valid_sources_lengths,
                     target_sequence_length: valid_targets_lengths,
                     keep_prob: 1.0})

                train_acc = get_accuracy(target_batch, batch_train_logits)

                valid_acc = get_accuracy(valid_targets_batch, batch_valid_logits)

                print('Epoch {:>3} Batch {:>4}/{} - Train Accuracy: {:>6.4f}, Validation Accuracy: {:>6.4f}, Loss: {:>6.4f}'
                      .format(epoch_i, batch_i, len(source_int_text) // batch_size, train_acc, valid_acc, loss))

    # Save Model
    saver = tf.train.Saver()
    saver.save(sess, save_path)
    print('Model Trained and Saved')
print('Done!')

#save model
pickle.dump(save_path, open('params.p','wb'))

