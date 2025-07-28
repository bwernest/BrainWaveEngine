"""___Modules_______________________________________________________________"""

# Python
import numpy as np
import copy
from time import perf_counter as clock

# BrainWaveEngine
from .activation import Activation
from .loss import Loss
from .network import Network

"""___Optimizer_____________________________________________________________"""

class Optimizer():

    def __init__(self) -> None:
        self.brains = []
        self.last_loss = 0

    def ConLin(self,
               network: Network,
               sample: list[float],
               solution: list[float],
               activation_sup: str = "identity",
               loss_function: str = "CCE",
               h1: float = 0.1, h2: float = 0.02,
               n_batch: int = 3,
               debug: bool = False
               ) -> None:
        """___Reference___"""
        activation = Activation()
        loss_fct = Loss()
        network.forward(sample)
        activation.forward(network.output, activation_sup)
        loss_fct.forward(activation.output, solution, loss_function)
        fx0 = loss_fct.output

        w_list0 = network.get_weights()
        b_list0 = network.get_biases()

        self.loss_history = np.array([[0, fx0]])
        if debug:
            print('Lowest loss : ' + str(fx0))

        nwei = len(w_list0)
        nbia = len(b_list0)
        nval = nwei + nbia

        """___Loop___"""
        for batch in range(n_batch):
            if debug:
                print('***_Batch_n°' + str(batch + 1) + '_**************')

            """___Df_calculation___"""
            Df = np.zeros(nval)
            for i in range(nwei):
                w_list = copy.deepcopy(w_list0)
                w_list[i][0] += h1

                network.insert_weights(w_list)
                network.forward(sample)
                activation.forward(network.output, activation_sup)
                loss_fct.forward(activation.output, solution, loss_function)
                Df[i] = (loss_fct.output - fx0) / h1

            for i in range(nbia):
                b_list = copy.deepcopy(b_list0)
                b_list[i][0] += h1

                network.insert_biases(b_list)
                network.forward(sample)
                activation.forward(network.output, activation_sup)
                loss_fct.forward(activation.output, solution, loss_function)
                Df[i + nwei] = (loss_fct.output - fx0) / h1

            """___Save___"""
            w_list0[:, 0] -= Df[:nwei] * h2
            b_list0[:, 0] -= Df[nwei:] * h2

            network.insert_weights(w_list0)
            network.insert_biases(b_list0)

            """___Reference___"""
            network.forward(sample)
            activation.forward(network.output, activation_sup)
            loss_fct.forward(activation.output, solution, loss_function)
            fx0 = loss_fct.output
            self.loss_history = np.append(self.loss_history, [[batch + 1, fx0]], axis=0)
            if debug:
                print(f"Lowest loss : {fx0}")

    def Homemade(self,
                 network: Network,
                 sample: list[float],
                 solution: list[float],
                 activation_sup: str = "identity",
                 loss_function: str = "CCE",
                 learning_rate: float = 1,
                 n_batch: int = 3,
                 n_step: int = 50,
                 debug: bool = False
                 ) -> None:
        '''___Setup___'''
        activation = Activation()
        loss_fct = Loss()
        weights_list = network.get_weights()
        biases_list = network.get_biases()
        iteration = 0

        '''___Reference___'''
        network.forward(sample)
        activation.forward(network.output, activation_sup)
        loss_fct.forward(activation.output, solution, loss_function)
        loss = loss_fct.output
        lowest_loss = loss
        self.loss_history = np.array([[0, lowest_loss]])
        best_weights_list = copy.deepcopy(weights_list)
        best_biases_list = copy.deepcopy(biases_list)

        for batch in range(n_batch):
            if debug:
                print('***_Batch_n°' + str(batch + 1) + '_**************')
            improvement = 0
            for w in range(len(weights_list)):
                x, y, z = weights_list[w, 1:]
                step = learning_rate
                fail = 0
                for _ in range(n_step):
                    iteration += 1
                    weights_list[w][0] += step
                    network.insert_weights(weights_list)
                    network.forward(sample)
                    activation.forward(network.output, activation_sup)
                    loss_fct.forward(activation.output, solution, loss_function)
                    loss = loss_fct.output

                    if loss < lowest_loss:
                        lowest_loss = loss
                        self.loss_history = np.append(self.loss_history, [[iteration, lowest_loss]], axis=0)
                        improvement += 1
                        best_weights_list = copy.deepcopy(weights_list)
                        if debug:
                            print('Lowest loss w =', lowest_loss)
                    else:
                        fail += 1
                        weights_list = copy.deepcopy(best_weights_list)
                        step *= -1
                        weights_list[w][0] += step
                        if fail == 2:
                            fail = 0
                            step /= 4
                weights_list = copy.deepcopy(best_weights_list)

            for b in range(len(biases_list)):
                x, y = biases_list[b, 1:]
                step = learning_rate
                fail = 0
                for _ in range(n_step):
                    iteration += 1
                    biases_list[b][0] += step
                    network.insert_biases(biases_list)
                    network.forward(sample)
                    activation.forward(network.output, activation_sup)
                    loss_fct.forward(activation.output, solution, loss_function)
                    loss = loss_fct.output

                    if loss < lowest_loss:
                        lowest_loss = loss
                        self.loss_history = np.append(self.loss_history, [[iteration, lowest_loss]], axis=0)
                        improvement += 1
                        best_biases_list = copy.deepcopy(biases_list)
                        if debug:
                            print('Lowest loss b =', lowest_loss)
                    else:
                        fail += 1
                        biases_list = copy.deepcopy(best_biases_list)
                        step *= -1
                        biases_list[b][0] += step
                        if fail == 2:
                            fail = 0
                            step /= 4
                biases_list = copy.deepcopy(best_biases_list)

            if improvement == 0:
                learning_rate /= 4
                if learning_rate < 1e-4:
                    network.trained = True
                    if debug:
                        print('Max potential reached')
                    break

        network.insert_weights(best_weights_list)
        network.insert_biases(best_biases_list)
        self.last_loss = lowest_loss

    def Drunk(self,
              network: Network,
              sample: list[float],
              solution: list[float],
              activation_sup: str = "identity",
              loss_function: str = "CCE",
              learning_rate: float = 1,
              n_batch: int = 3,
              n_step: int = 50,
              debug: bool = False
              ) -> None:

        activation = Activation()
        loss_fct = Loss()

        best_weights_list = copy.deepcopy(network.get_weights())
        best_biases_list = copy.deepcopy(network.get_biases())

        weights_list = copy.deepcopy(best_weights_list)
        biases_list = copy.deepcopy(best_biases_list)

        lenw, lenb = len(weights_list), len(biases_list)

        network.forward(sample)
        activation.forward(network.output, activation_sup)
        loss_fct.forward(activation.output, solution, loss_function)
        lowest_loss = loss_fct.output

        for batch in range(n_batch):
            if debug:
                print('***_Batch_n°' + str(batch + 1) + '_**************')

            for _ in range(n_step):
                weights_list[:, 0] += learning_rate * np.random.random(lenw)
                biases_list[:, 0] += learning_rate * np.random.random(lenb)

                network.insert_weights(weights_list)
                network.insert_biases(biases_list)

                network.forward(sample)
                activation.forward(network.output, activation_sup)
                loss_fct.forward(activation.output, solution, loss_function)

                if loss_fct.output < lowest_loss:
                    lowest_loss = loss_fct.output
                    best_weights_list = copy.deepcopy(weights_list)
                    best_biases_list = copy.deepcopy(biases_list)
                    if debug:
                        print('Lowest loss =', lowest_loss)

                else:
                    weights_list = copy.deepcopy(best_weights_list)
                    biases_list = copy.deepcopy(best_biases_list)

            learning_rate /= 10
        network.insert_weights(best_weights_list)
        network.insert_biases(best_biases_list)

    def BaffWill_training_v1(self,
                             AI: object,
                             loss_fct,
                             children: int = 10,
                             parties: int = 10
                             ) -> None:
        """
        Méthode sans gradient d'algo génétique du livre d'opti de structures.
        """
        start = clock()
        n_children = children

        lowest_loss = loss_fct(AI)
        reach = 10
        layers_list = AI.network.layers_list
        children_network = [AI.network]
        children_loss = [lowest_loss]
        self.loss_history = [[0], [-1 * lowest_loss]]

        w_list = AI.network.get_weights()
        b_list = AI.network.get_biases()
        a_list = AI.network.get_activations()

        lenW = len(w_list)
        lenB = len(b_list)

        while reach > 0.01:
            self.loss_history[0].append(len(self.loss_history[0]))
            for _ in range(n_children):
                child = Network()
                child.default(layers_list)
                w_child = child.get_weights()
                b_child = child.get_biases()

                for w in range(lenW):
                    w_child[w][0] = w_list[w][0] + 2 * reach * (np.random.random() - 0.5)
                for b in range(lenB):
                    b_child[b][0] = b_list[b][0] + 2 * reach * (np.random.random() - 0.5)

                child.insert_weights(w_child)
                child.insert_biases(b_child)
                child.insert_activations(a_list)

                children_network.append(child)
                AI.network = child
                children_loss.append(loss_fct(AI, match=parties))

            best = np.argmin(children_loss)
            if children_loss[best] <= lowest_loss:
                lowest_loss = children_loss[best]
                w_list = children_network[best].get_weights()
                b_list = children_network[best].get_biases()
                AI.network = children_network[best]

            children_network = [AI.network]
            children_loss = [lowest_loss]

            self.loss_history[1].append(-1 * lowest_loss)
            reach /= 2
        # print('Finished in ' + str(round(clock()-start, 2)) + ' sec')

    """_Randomize again random layers_"""

    def Knockout(self,
                 n_networks: int,
                 example: Network,
                 sample: list[float],
                 solution: list[float],
                 activation_sup: str = "identity",
                 loss_function: str = "CCE",
                 learning_rate: float = 1,
                 knock: int = 5,
                 n_batch: int = 3,
                 n_step: int = 50,
                 debug: bool = False
                 ) -> None:
        net = Network()
        net.copy(example)

        weights_list = copy.deepcopy(net.get_weights())
        biases_list = copy.deepcopy(net.get_biases())
        weights_len = len(weights_list)
        biases_len = len(biases_list)

        brains = []

        for _ in range(n_networks):
            weights_list[:, 0] = np.random.random(weights_len)
            biases_list[:, 0] = np.random.random(biases_len)

            brains.append([copy.deepcopy(weights_list), copy.deepcopy(biases_list), False])
        # print(brains)

        while len(brains) != 1:
            loss_list = []

            for b in range(len(brains)):
                net.trained = brains[b][2]
                if net.trained:
                    print('ON SKIP')

                if not net.trained:
                    net.insert_weights(brains[b][0])
                    net.insert_biases(brains[b][1])
                    self.Homemade(net, sample, solution, activation_sup=activation_sup, loss_function=loss_function,
                                  learning_rate=learning_rate, n_batch=n_batch, n_step=n_step, debug=debug)

                    # print('On ajoute ça :', self.last_loss)
                    loss_list.append(self.last_loss)
                    brains[b][0] = net.get_weights()
                    brains[b][1] = net.get_biases()
                    brains[b][2] = net.trained

            # print('Loss list :', loss_list)
            if len(brains) <= knock:
                knock = len(brains) - 1
            for _ in range(knock):
                print('Brains left :', len(brains))
                worst = np.argmax(loss_list)
                print('Worst :', worst, 'with loss =', str(round(loss_list[worst], 3)), '\n')
                brains[worst] = brains[-1]
                loss_list[worst] = loss_list[-1]
                brains = brains[:-1]
                loss_list = loss_list[:-1]
        self.last_loss = loss_list[0]
        net.insert_weights(brains[0][0])
        net.insert_biases(brains[0][1])
        self.output = net
