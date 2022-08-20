import dynet as dy
from char_level_lstm import train, VOCAB_SIZE, pc

LAYERS = 1
INPUT_DIM = 50
HIDDEN_DIM = 50


def main():

    srnn = dy.SimpleRNNBuilder(LAYERS, INPUT_DIM, HIDDEN_DIM, pc)
    lstm = dy.LSTMBuilder(LAYERS, INPUT_DIM, HIDDEN_DIM, pc)

    # add parameters for the hidden->output part for both lstm and srnn
    params_lstm = {}
    params_srnn = {}
    for params in [params_lstm, params_srnn]:
        params["lookup"] = pc.add_lookup_parameters((VOCAB_SIZE, INPUT_DIM))
        params["R"] = pc.add_parameters((VOCAB_SIZE, HIDDEN_DIM))
        params["bias"] = pc.add_parameters((VOCAB_SIZE,))

    sentence = "a quick brown fox jumped over the lazy dog"

    train(srnn, params_srnn, sentence)

    train(lstm, params_lstm, sentence)


if __name__ == '__main__':
    main()

    # dy.renew_cg()
