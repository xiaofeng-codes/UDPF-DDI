import torch
import torch.nn as nn


class FuzzyLayer(nn.Module):

    def __init__(self, input_vector_size, fuzz_vector_size, num_class, fuzzy_layer_input_dim=1,
                 fuzzy_layer_output_dim=1, dropout_rate=0.5):
        super(FuzzyLayer, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.input_vector_size = input_vector_size
        self.fuzz_vector_size = fuzz_vector_size
        self.num_class = num_class

        self.fuzzy_layer_input_dim = fuzzy_layer_input_dim
        self.fuzzy_layer_output_dim = fuzzy_layer_output_dim

        fuzzy_degree_weights = torch.Tensor(1, self.fuzz_vector_size)
        self.fuzzy_degree = nn.Parameter(fuzzy_degree_weights)
        sigma_weights = torch.Tensor(1, self.fuzz_vector_size)
        self.sigma = nn.Parameter(sigma_weights)

        nn.init.xavier_uniform_(self.fuzzy_degree)
        nn.init.ones_(self.sigma)

    def forward(self, batch):

        input = batch

        fuzz_input = input

        input_expanded = fuzz_input.unsqueeze(-1)

        fuzzy_degree_expanded = self.fuzzy_degree.expand(fuzz_input.size(0), -1).unsqueeze(-1)

        sigma_expanded = self.sigma.expand(fuzz_input.size(0), -1).unsqueeze(-1) + 1e-5

        diff = input_expanded - fuzzy_degree_expanded
        diff_squared = torch.square(diff)

        denominator = sigma_expanded ** 2

        exponent = -torch.sum(diff_squared / denominator, dim=-1)
        if torch.isnan(exponent).any() or torch.isinf(exponent).any():
            print(exponent)

        fuzzy_out = torch.exp(exponent)

        batch = fuzzy_out
        return batch
