import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch import optim
import neurogym as ngym
from sklearn.decomposition import PCA

class ContinuousTimeModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, tau: float, dt: float) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.tau = tau
        self.dt = dt
        self.constant = self.dt / self.tau

        self.input_layer = nn.Linear(self.input_size, self.hidden_size)
        self.hidden_layer = nn.Linear(self.hidden_size, self.hidden_size)

    def recur(self, input, hidden):
        return hidden * (1 - self.constant) + self.constant * torch.relu(
            self.input_layer(input) + self.hidden_layer(hidden))

    def forward(self, input, hidden=None):

        if hidden is None:
            hidden = torch.zeros(input.shape[1], self.hidden_size)

        output = []

        print(input.shape)

        for step in range(input.size(0)):
            hidden_output = self.recur(input[step], hidden)
            output.append(hidden_output)

        output = torch.stack(output, dim=0)

        return output, hidden


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, tau, dt):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.tau = tau
        self.dt = dt
        self.rnn = ContinuousTimeModel(self.input_size, self.hidden_size, self.tau, self.dt)
        self.fc = nn.Linear(hidden_size, self.output_size)

    def forward(self, input):
        rnn_output, hidden = self.rnn(input)
        logits = self.fc(rnn_output)
        return logits, rnn_output


class ModelTrainer:
    import torch
    import numpy as np
    import neurogym as ngym

    def __init__(self, tau: object, dt: object, x_0: object, W: object, seed: object) -> object:
        self.seed = seed
        self.torch.manual_seed(seed)
        self.np.random.seed(seed)
        self.x_0 = x_0
        self.dt = dt
        self.tau = tau
        self.W = W
        self.hidden_activations = []
        self.input_activations = []
        self.train_loader = None

    def get_train_loader(self):

        if self.train_loader == None:
            print('WARNING: loader is None')

        return self.train_loader

    def prepare_data_ngym(self, plot = True):

        # Environment
        task = 'PerceptualDecisionMaking-v0'
        kwargs = {'dt': 100}
        seq_len = 100

        # Make supervised dataset
        dataset = ngym.Dataset(task, env_kwargs=kwargs, batch_size=16,
                               seq_len=seq_len)

        # A sample environment from dataset
        env = dataset.env
        # Visualize the environment with 2 sample trials
        _ = ngym.utils.plot_env(env, num_trials=2)

        # Network input and output size
        input_size = env.observation_space.shape[0]
        output_size = env.action_space.n
        return dataset

    def prepare_data_sim(self, plot=False):
        # Predefined paramters
        ar_n = 3  # Order of the AR(n) data
        ar_coeff = [0.7, -0.3, -0.1]  # Coefficients b_3, b_2, b_1
        noise_level = 0.1  # Noise added to the AR(n) data
        length = 200  # Number of data points to generate

        # Random initial values
        ar_data = list(np.random.randn(ar_n))

        # Generate the rest of the values
        for i in range(length - ar_n):
            next_val = (np.array(ar_coeff) @ np.array(ar_data[-3:])) + np.random.randn() * noise_level
            ar_data.append(next_val)

        if plot:
            # Plot the time series
            fig = plt.figure(figsize=(12, 5))
            plt.plot(ar_data)
            plt.show()

        print(np.array(ar_data).shape)

        samples = list()
        length = 20
        n = 200
        # step over the 5,000 in jumps of 200
        for i in range(0, n, length):
            sample = ar_data[i:i + length]
            samples.append(sample)
        print(len(samples))

        # convert list of arrays into 2d array
        data = np.array(samples).reshape((10, 20, 1))  ##bug here?
        print(data.shape)
        return torch.tensor(data).float()

    def test_data_size(self, dataset):
        batch_size = 10
        seq_len = 20  # sequence length
        input_size = 1  # input dimension

        # Make network
        rnn = RNN(input_size=input_size, hidden_size=100, output_size=1, tau=250, dt=25)

        # Run the sequence through the network
        out, rnn_output = rnn(dataset)

        print('Input of shape (SeqLen, BatchSize, InputDim)=', dataset.shape)
        print('Output of shape (SeqLen, BatchSize, Neuron)=', out.shape)

    def get_model(self):
        return RNN(input_size=1, hidden_size=100, output_size=1, tau=250, dt=25)

    def train_model(self, model, data):
        self.train_loader = torch.utils.data.DataLoader(dataset=data, batch_size=2, shuffle=False)
        optimizer = optim.SGD(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()


        for epoch in range(10):
            for i, input in enumerate(self.train_loader):
                model.train()

                model_input = input[0, :, :].reshape((1, 20, 1))
                labels = input[1, :, :].reshape((1, 20, 1))

                optimizer.zero_grad()

                self.input_activations.append(model_input.detach().numpy().reshape((20,)))

                outputs, rnn_output = model(model_input)

                self.hidden_activations.append(rnn_output.detach().numpy().reshape((20, 100)))

                loss = criterion(outputs, labels)

                loss.backward()

                optimizer.step()

                print('loss:' + str(loss))

    def get_inputs_and_activations(self):
        return np.array(self.input_activations), np.array(self.hidden_activations)

    def run(self):
        dataset = self.prepare_data_sim()
        model = self.get_model()
        self.train_model(model, dataset)
        self.test_data_size(dataset)
        return model

class ModelAnalyzer:
    def __init__(self, model, loader):
        self.model = model
        self.loader = loader

    #def plot_dynamical_system_analysis(self):

    def plot_pca(self):
        perf = 0
        num_trial = 10
        activity_dict = {}
        trial_infos = {}
        for i, input in enumerate(self.loader):

            model_input = input[0, :, :].reshape((1, 20, 1))
            labels = input[1, :, :].reshape((1, 20, 1))

            action_pred, rnn_activity = self.model(model_input)
            print('rnn_activity shape: ')
            print(rnn_activity.shape)
            rnn_activity = rnn_activity[:, 0, :].detach().numpy().reshape((-1, 100))
            activity_dict[i] = rnn_activity
            trial_infos[i] = labels[:, 0, :]
            print(i)

        activity_list = []

        for i in range(4):
            print(activity_dict[i])
            activity_list.append(activity_dict[i])


        # Concatenate activity for PCA
        activity = np.concatenate(activity_list, axis=0)
        print('Shape of the neural activity: (Time points, Neurons): ', activity.shape)
        pca = PCA(n_components=2)
        pca.fit(activity)  # activity (Time points, Neurons)
        activity_pc = pca.transform(activity)  # transform to low-dimension
        print('Shape of the projected activity: (Time points, PCs): ', activity_pc.shape)

        # Print trial informations
        for i in range(5):
            print('Trial ', i, trial_infos[i])

        # Plot all trials in ax1, plot fewer trials in ax2
        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(6, 3))

        for i in range(4):
            # Transform and plot each trial
            activity_pc = pca.transform(activity_dict[i])  # (Time points, PCs)

            trial = trial_infos[i]
            color = 'red'

            _ = ax1.plot(activity_pc[:, 0], activity_pc[:, 1], 'o-', color=color)
            if i < 3:
                _ = ax2.plot(activity_pc[:,
                             0], activity_pc[:, 1], 'o-', color=color)

            # Plot the beginning of a trial with a special symbol
            _ = ax1.plot(activity_pc[0, 0], activity_pc[0, 1], '^', color='black')

        ax1.set_title('{:d} Trials'.format(100))
        ax2.set_title('{:d} Trials'.format(3))
        ax1.set_xlabel('PC 1')
        ax1.set_ylabel('PC 2')
        fig.show()

    def plot_inputs_and_activations(self, inputs, activations):
        print(inputs.shape)
        print(activations.shape)

        fig, a = plt.subplots(10, 10)

        x_idx = 0
        y_idx = 0

        for i in range(100):
            x_array = []
            y_array = []

            for j in range(20):
                x_array.append(inputs[49, j]*10)
                y_array.append(activations[49, j, i]*10)
                #print('for unit number: ' + str(i))
                #print('************************************')
                #print(inputs[49, j])
                #print(activations[49, j, i])
                #print('------------------------------------')

            if i % 10 == 0 and i != 0:
                x_idx = 0
                y_idx = y_idx + 1
                a[x_idx][y_idx].title.set_text(str(i))

            a[x_idx][y_idx].plot(x_array, y_array)

            if i == 0:
                a[x_idx][y_idx].title.set_text(str(i))

            print('for unit number: ' + str(i))
            print('::::::::::::::::::::::::::::::::::::::::::::')
            print(max(x_array))
            print(min(x_array))
            print(max(y_array))
            print(min(y_array))
            print('::::::::::::::::::::::::::::::::::::::::::::')
            x_idx = x_idx + 1

        plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
        plt.show()

if __name__ == '__main__':
    model_trainer = ModelTrainer(tau=250.0, dt=25.0, x_0=1.0, W=2.0, seed=10)
    model = model_trainer.run()
    inputs, activations = model_trainer.get_inputs_and_activations()
    loader = model_trainer.get_train_loader()

    model_analyzer = ModelAnalyzer(model, loader)
    model_analyzer.plot_inputs_and_activations(inputs, activations)
    model_analyzer.plot_pca()
