import dynet as dy
import numpy as np

v1 = dy.inputVector([1, 2, 3, 4])
v2 = dy.inputVector([5, 6, 7, 8])  # v1 & v2 are expressions

v3 = v1 + v2
print(v1.value())
print(v2.value())
print(v3.value())

e1 = dy.vecInput(4)
e1.set([1, 2, 3, 4])
restr_index = 3

def sum_logs(lst):
    return sum([dy.exp(e) for e in lst])

def assert_softmax(source_lst, restr_index, logsoftmax_lst):
    if restr_index is not None:
        source_lst = source_lst[:restr_index]
        logsoftmax_lst = logsoftmax_lst[:restr_index]

    assert np.allclose(source_lst.npvalue(), (logsoftmax_lst + dy.log(sum(dy.exp(source_lst)))).npvalue())

e_log_softmax = dy.log_softmax(e1)
print(f'log_softmax result: {np.round(e_log_softmax.value(), 4)}')
print(f'Sum(e_log_softmax) = {sum_logs(e_log_softmax).value():.2}')
assert_softmax(e1, None, e_log_softmax)

e_log_softmax2 = dy.log_softmax(e1, restrict=list(range(restr_index)))
print(f'log_softmax2 result: {np.round(e_log_softmax2.value(), 4)}')
print(f'Sum(e_log_softmax) in 0 <= i <= {restr_index} = {sum_logs(e_log_softmax2[:restr_index]).value():.2}')
assert_softmax(e1, restr_index, e_log_softmax2)

print(type(e_log_softmax.npvalue()))
print(type(e_log_softmax2.npvalue()))
