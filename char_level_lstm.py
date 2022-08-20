import dynet as dy
import random
import numpy as np

pc = dy.ParameterCollection()

characters = list("abcdefghijklmnopqrstuvwxyz ")+["<EOS>"]
VOCAB_SIZE = len(characters)

int2char = list(characters)
char2int = {c:i for i,c in enumerate(characters)}

def get_gradient(index: int):
    return pc.parameters_list()[index].grad_as_array()

def is_zeros(t: np.ndarray):
    return np.allclose(t, np.zeros(t.shape))


# return compute loss of RNN for one sentence
def do_one_sentence(rnn, params, sentence):
    # setup the sentence
    dy.renew_cg()
    s0 = rnn.initial_state()

    R = params["R"]
    bias = params["bias"]
    lookup = params["lookup"]
    sentence = ["<EOS>"] + list(sentence) + ["<EOS>"]
    sentence = [char2int[c] for c in sentence]
    s = s0
    loss = []
    for char,next_char in zip(sentence,sentence[1:]):
        s = s.add_input(lookup[char])
        probs = dy.softmax(R*s.output() + bias)
        loss.append( -dy.log(dy.pick(probs,next_char)) )
    loss = dy.esum(loss)
    return loss

def generate(rnn, params):
    def sample(probs):
        rnd = random.random()
        for i,p in enumerate(probs):
            rnd -= p
            if rnd <= 0: break
        return i

    # setup the sentence
    dy.renew_cg()
    s0 = rnn.initial_state()

    R = params["R"]
    bias = params["bias"]
    lookup = params["lookup"]

    s = s0.add_input(lookup[char2int["<EOS>"]])
    out=[]
    while True:
        probs = dy.softmax(R*s.output() + bias)
        probs = probs.vec_value()
        next_char = sample(probs)
        out.append(int2char[next_char])
        if out[-1] == "<EOS>": break
        s = s.add_input(lookup[next_char])
    return "".join(out[:-1]) # strip the <EOS>

# train, and generate every 5 samples
def train(rnn, params, sentence):
    trainer = dy.SimpleSGDTrainer(pc)
    for i in range(200):
        loss = do_one_sentence(rnn, params, sentence)
        loss_value = loss.value()
        loss.backward()
        trainer.update()
        if i % 5 == 0:
            print("%.10f" % loss_value, end='\t')
            print(generate(rnn, params))
        for j in range(len(pc.parameters_list())):
            if not is_zeros(get_gradient(j)):
                print("\nHELLO \n")