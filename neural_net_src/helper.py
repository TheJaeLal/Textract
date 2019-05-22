import numpy as np

from neural_net_src.train_config import vocab_size

def load_data(filename):
    with open(filename,'r') as f:
        content = f.read()

    lines = content

    #print(lines)

    char_set = set()
    vocabulary = []
    
    for char in lines:
        if char not in char_set:
            char_set.add(char)
            vocabulary.append(char)    
        
    #for i,char in enumerate(char_set):
        #print("{0}:{1}".format(i,char))
    return lines,vocabulary

def encode_int(text,vocabulary):
	#Takes in a string of "text" and vocabulary (list of all possible output characters)
	#Returns a list of integers encoding every character w.r.t its position in vocabulary
	list_chars = list(text)
	#Returns a list of indices
	return [vocabulary.index(char) for char in list_chars]


def one_hot(text,vocabulary):

    z = encode_int(text,vocabulary)
    
    #Takes in a list of ints:
    
    #Returns a 2d Array
    print("New one_hot")
    one_hot_array = np.zeros([len(z),vocab_size]).astype(np.float32)
    for i in range(len(z)):
        el = z[i]
        one_hot_array[i][el] = 1
    
    return one_hot_array

#ToDO: Add batch_size feature.... (multiple samples in single batch)
def create_chunks(input_sequence,output_sequence,vocab_size,sequence_size=25,batch_size=1):
    
    chunks = []
    #Break down data into chunks .. of sequence_size
    for sample_start in range(0,len(input_sequence),sequence_size):
        input_sample = one_hot(input_sequence[sample_start:sample_start+sequence_size],vocab_size)
        output_sample = one_hot(output_sequence[sample_start:sample_start+sequence_size],vocab_size)
        chunks.append( (input_sample,output_sample) )
        
    return chunks


def interpret(y,vocabulary):    
    interpretation = []
    for i in range(y[0].shape[0]):
        index = np.argmax(y[0][i])
        #print(index)
        interpretation.append(vocabulary[index])
    cont = "".join(interpretation)
    #print(cont)
    return cont

def stack_batches(batch_size,chunks):
    pass
    #return input_batches

def sparse_tuple_form(sequences, dtype=np.int32):
    """Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int32)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int32)

    return indices, values, shape

def create_batches(batch_size,training_list,image_arrays,image_labels,vocabulary):
    
    return _batch_x(batch_size,training_list,image_arrays),_batch_y(batch_size,training_list,image_labels,vocabulary)

def _batch_x(batch_size,training_list,image_arrays):
    
    #-------Batch images-------#
    batches_x = []
    #Stack images together to form batches
    for i in range(0,len(training_list),batch_size):
        #Initialize the current_batch with first image
#         current_batch = (image_arrays[training_list[i]] - image_arrays[training_list[i]].mean())
        current_batch = image_arrays[training_list[i]]
        for b in range(1,batch_size):
            #Get the next_image from training_list
#             next_image = (image_arrays[training_list[i+b]] - image_arrays[training_list[i+b]].mean())
            next_image = image_arrays[training_list[i+b]]
            #Add it to the current batch, such that the depth increases
            #i.e current_batch --> (height,width,1), next_image --> (height,width)
            # then current_batch --> (height,width,2) and so, on until you get
            #current_batch --> (height,width,batch_size)
            current_batch = np.dstack((current_batch,next_image))
        batches_x.append(current_batch)
        
    return batches_x

def _batch_y(train_label,vocabulary):
    
    #Get the current batch, list of sequences(lists)
    list_of_labels = [encode_int(train_label[b],vocabulary) for b in range(len(train_label))]

    #Convert the list of sequences(lists) to sparse_tuple_form and add them to batches_y
    return sparse_tuple_form(list_of_labels)
    
def create_batches_test(batch_size,training_list,image_arrays):
        
    return _batch_x(batch_size,training_list,image_arrays)