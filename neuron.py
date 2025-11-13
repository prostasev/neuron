from random import randint  # ,randrange
from math import ceil, exp


class Layer:
    def __init__(self, layer_size, prev_layer, parent_network):
        self.prev_layer = prev_layer
        self.network = parent_network
        self.neurons = [Neuron(self, prev_layer) for _ in range(layer_size)]

    def no_inputs(self):
        return bool(self.prev_layer)
        # if self.prev_layer is None:
        #     return True
        # return False

    def print(self):
        netw = self.network
        netw_size = len(netw.layers)
        l_num = netw.layers.index(self) + 1  # list.index(element) поиск индекса элемента в списке
        l_type = 'In' if l_num == 1 else 'Out' if l_num == netw_size else 'Hidden'
        print(f'Layer_{l_num}, Type: {l_type}, Neurons - {len(self.neurons)}')
        for neuron in self.neurons:
            neuron.print()

    def set_input_data(self, val_list):
        if len(val_list) != len(self.neurons):
            return
        for i in range(len(val_list)):
            self.neurons[i].set_value(val_list[i])


class Neuron:
    # В конструктор нейрона мы можем передать два параметра: слой,
    # на котором этот нейрон находится и, если это не входной слой нейросети,
    # ссылку на предыдущий слой. В конструкторе, для каждого нейрона предыдущего слоя
    # мы создадим вход (массив inputs), который свяжет нейроны и будет иметь случайный вес,
    # и запишем все входы в массив inputs. Если же это входной слой сети, то массив inputs
    # будет состоять из единственного числового значения, того, которое мы передадим на вход.
    # inputs - массив входов!
    # конструктор создает вход, кот. свяжет нейроны и имеет случ вес
    # все входы запис. в масс. inputs
    @staticmethod
    def my_round(num: float) -> float:
        s_num = str(num)
        if '.' in s_num:
            pos = s_num.index('.') + 1
            s_num = s_num[:pos + 3]
            num = float(s_num)
        return num

    def __init__(self, layer: Layer, previous_layer: Layer):
        self.value = 0
        self._layer = layer
        self.inputs = [Input(prev_neuron, randint(0, 10) / 10) for prev_neuron in
                       previous_layer.neurons] if previous_layer else []  # Генератор списка + однострочное условие
        # random.randint(0, 10) / 10   случайное число от 0.0 до 1.0
        # последовательно создавать входы для каждого нейрона из списка neurons и назначать им рандомный вес 0..1
        # new Layer(layerSize, this.layers[this.layers.length - 1], this)
        # this - экземпляр NN
        # layerSize - количество нейронов в слое - его размерность
        # this.layers[this.layers.length - 1] = previous_layer : Layer
        # my_listcomp = [chr(i) for i in range(97, 123)]
        # if previous_layer != None
        self.get_value()

    def is_no_inputs(self):
        return not self.inputs

    def print(self):
        layer = self._layer
        n_num = layer.neurons.index(self) + 1
        n_size = len(self.inputs)
        print(f' Neuron_{n_num}, Value - {self.my_round(self.value)}, Inputs - {n_size}')
        for n_input in self.inputs:
            i_num = self.inputs.index(n_input) + 1
            print(f'\tInput_{i_num}, Weight - {n_input}')

    def get_value(self):
        """
        Пересчитывает и возвращает значение нейрона, если он имеет входы, иначе просто возвращает значение
        :return:
        """
        network = self._layer.network
        if not self.is_no_inputs():
            self.set_value(network.activate_func(self.get_input_sum()))
        return self.value

    def get_input_sum(self) -> float:
        inputs = self.inputs
        total_sum = sum(curr_input.prev_neuron.get_value() * curr_input.weight for curr_input in inputs)
        return total_sum

    def set_error(self, val):
        if self.is_no_inputs():
            return
        w_delta = val * self._layer.network.derivate_func(self.get_input_sum())
        for curr_input in self.inputs:
            curr_input.weight -= curr_input.prev_neuron.get_value() * w_delta * self._layer.network.learning_rate
            curr_input.prev_neuron.set_error(curr_input.weight * w_delta)

    def set_value(self, val):
        self.value = val


# У каждого входа есть числовой вес и ссылка на нейрон предыдущего слоя
class Input:
    def __init__(self, prev_neuron, weight):
        self.prev_neuron = prev_neuron
        self.weight = weight

    def __str__(self):
        return str(Neuron.my_round(self.weight))


class NeuroNetwork:
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + exp(
            -x))  # exp() – возвращает экспоненциальное значение x: e^x.
        # Аргумент x может быть целого или вещественного типа

    @staticmethod
    def sigmoid_derivative(x):
        return NeuroNetwork.sigmoid(x) * (1 - NeuroNetwork.sigmoid(x))

    def __init__(self, input_l_size, output_l_size, hidden_layers_count=1, learning_rate=0.5):
        self.activate_func = NeuroNetwork.sigmoid
        self.derivate_func = NeuroNetwork.sigmoid_derivative
        self.learning_rate = learning_rate
        self.selected_layer = None  # указатель на слой
        self.l_count = hidden_layers_count + 2  # 2 = 1 input + 1 output
        hidden_l_size = min(input_l_size * 2 - 1, ceil(
            input_l_size * 2 / 3 + output_l_size))  # формула расчета размера скрытых слоев,
        # на основе размера входного и выходного слоев
        # (math.ceil() - округление числа с точкой до большего целого)
        self.layers = []
        self.layers = [self.add_layer(i, input_l_size, output_l_size, hidden_l_size) for i in
                       range(self.l_count)]  # range of i = [0..l_count)
        self.selected_layer = None  # "чистим указатель"

    # refactor -> rename
    def add_layer(self, i, in_size, out_size, hl_size):  # self - текущий экземпляр NeuroNetwork
        count = i + 1  # range of i = [0..l_count)
        if 1 < count < self.l_count:  # hidden
            self.selected_layer = Layer(hl_size, self.selected_layer,
                                        self)  # создаем новый слой на основе слоя с указателем,
            # и ставим указатель на созданный слой
            return self.selected_layer
        if count == 1:  # input
            self.selected_layer = Layer(in_size, None, self)  # ставим указатель на первый слой
            return self.selected_layer
        # else: count == l_count -> output
        self.selected_layer = Layer(out_size, self.selected_layer, self)
        return self.selected_layer

    def print(self):
        info = self.__sum_info()
        print('about NN:')
        print(f'Layers - {info["l"]}, Neurons - {info["n"]}, Inputs - {info["i"]}')
        layers = self.layers
        for layer in layers:
            layer.print()

    def __sum_info(self):
        layers = self.layers
        inf = {'l': len(layers), 'n': 0, 'i': 0}
        neurons = 0
        inputs = 0
        for layer in layers:
            for neuron in layer.neurons:
                neurons += 1
                inputs += len(neuron.inputs)
        inf['n'] = neurons
        inf['i'] = inputs
        return inf

    def train(self, dataset, iters=1000):
        print(f'\nTRAINING STARTED({iters} iterations)...')
        for _ in range(iters):
            self.train_once(dataset)
        print(f'\nTRAINING COMPLETED!\n')

    def train_once(self, dataset):
        """
        В нем мы устанавливаем входные данные, получаем предсказание сети, вычисляем ошибку и сообщаем ее,
        инициируя пересчет весов нейронных связей. :param dataset:
        """
        if not isinstance(dataset, list):
            return  # TODO: нужна ли эта проверка?
        for case in dataset:
            datacase = {'in_data': case[0], 'res': case[1]}
            self.set_input_data(datacase['in_data'])
            curr_res = self.get_prediction()
            for i in range(len(curr_res)):
                self.layers[self.l_count - 1].neurons[i].set_error(
                    curr_res[i] - datacase['res'])  # self.layers[self.l_count - 1] = out layer

    def get_prediction(self):
        """
        Происходит пересчет значений с нейронов выходного слоя и возвращение их в виде списка
        :return: out_data
        """
        layers = self.layers
        output_layer = layers[len(layers) - 1]
        out_data = [neuron.get_value() for neuron in output_layer.neurons]
        return out_data

    def set_input_data(self, val_list):
        self.layers[0].set_input_data(val_list)

    def test(self, data, op_name):
        print('\nTESTING DATA:')
        for case in data:
            self.set_input_data(case)
            res = self.get_prediction()
            print(f'{case[0]} {op_name} {case[1]} ~ {Neuron.my_round(res[0])}')


new_n_n = NeuroNetwork(2, 1)
new_n_n.print()
dataset_or = [[[0, 0], 0], [[0, 1], 1], [[1, 0], 1], [[1, 1], 1]]
new_n_n.train(dataset_or, 100000)
new_n_n.print()
test_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
new_n_n.test(test_data, 'OR')
