import dynet as dy
import numpy as np

# # create a parameter collection and add the parameters.
# m = dy.ParameterCollection()
# W = m.add_parameters((8,2))
# V = m.add_parameters((1,8))
# b = m.add_parameters((8))
#
# dy.renew_cg() # new computation graph. not strictly needed here, but good practice.
#
# x = dy.vecInput(2) # an input vector of size 2. Also an expression.
# y = dy.scalarInput(0) # this will hold the correct answer
#
# trainer = dy.SimpleSGDTrainer(m)
#
# output = dy.logistic(V*(dy.tanh((W*x)+b)))
# # we want to be able to define a loss, so we need an input expression to work against.
# loss = dy.binary_log_loss(output, y)


#
# # we can now query our network
# x.set([0,0])
# output.value()
#
# x.set([1,0])
# y.set(0)
# print(loss.value())
#
# y.set(1)
# print(loss.value())
#
#
# x.set([1,0])
# y.set(1)
# loss_value = loss.value() # this performs a forward through the network.
# print("the loss before step is:",loss_value)
#
# # now do an optimization step
# loss.backward()  # compute the gradients
# trainer.update()
#
# # see how it affected the loss:
# loss_value = loss.value(recalculate=True) # recalculate=True means "don't use precomputed value"
# print("the loss after step is:",loss_value)


# define the parameters
m = dy.ParameterCollection()
W = m.add_parameters((8,2))
V = m.add_parameters((1,8))
b = m.add_parameters((8))

# renew the computation graph
dy.renew_cg()

def get_gradient(index: int):
    return m.parameters_list()[index].grad_as_array()

def is_zeros(t: np.ndarray):
    return np.allclose(t, np.zeros(t.shape))


# create the network
x = dy.vecInput(2) # an input vector of size 2.
output = dy.logistic(V*(dy.tanh((W*x)+b)))
# define the loss with respect to an output y.
y = dy.scalarInput(0) # this will hold the correct answer
loss = dy.binary_log_loss(output, y)

# create training instances
def create_xor_instances(num_rounds=2000):
    questions = []
    answers = []
    for round in range(num_rounds):
        for x1 in 0,1:
            for x2 in 0,1:
                answer = 0 if x1==x2 else 1
                questions.append((x1,x2))
                answers.append(answer)
    return questions, answers

questions, answers = create_xor_instances()

# train the network
trainer = dy.SimpleSGDTrainer(m)

total_loss = 0
seen_instances = 0
for question, answer in zip(questions, answers):
    x.set(question)
    y.set(answer)
    seen_instances += 1
    total_loss += loss.value()
    loss.backward()
    trainer.update()
    if seen_instances > 1 and seen_instances % 100 == 0:
        print("average loss is:", np.round(total_loss / seen_instances, 4))
    for j in range(len(m.parameters_list())):
        if not is_zeros(get_gradient(j)):
            print("\nHELLO \n")