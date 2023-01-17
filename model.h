#ifndef MODEL_H
#define MODEL_H

#include <torch/torch.h>
#include <QDebug>

class Actor : public torch::nn::Module
{
public:
    Actor(int state_dim, int action_dim, int hidden_unites)
        : fully_connected_layer_1(torch::nn::Linear(state_dim, hidden_unites)),
          batch_normal_layer_1(torch::nn::BatchNorm1d(hidden_unites)),
          fully_connected_layer_2(torch::nn::Linear(hidden_unites, hidden_unites)),
          batch_normal_layer_2(torch::nn::BatchNorm1d(hidden_unites)),
          fully_connected_layer_3(torch::nn::Linear(hidden_unites, hidden_unites)),
          batch_normal_layer_3(torch::nn::BatchNorm1d(hidden_unites)),
          fully_connected_layer_4(torch::nn::Linear(hidden_unites, action_dim))
    {
        register_module("fully_connected_layer_1", fully_connected_layer_1);
        register_module("batch_normal_layer_1", batch_normal_layer_1);
        register_module("fully_connected_layer_2", fully_connected_layer_2);
        register_module("batch_normal_layer_2", batch_normal_layer_2);
        register_module("fully_connected_layer_3", fully_connected_layer_3);
        register_module("batch_normal_layer_3", batch_normal_layer_3);
        register_module("fully_connected_layer_4", fully_connected_layer_4);
        resetParameters();
    };

    torch::Tensor forward(torch::Tensor state) {
        torch::Tensor x;
        x = fully_connected_layer_1->forward(state);
        x = batch_normal_layer_1->forward(x);
        x = torch::tanh(x);

        x = fully_connected_layer_2->forward(x);
        x = batch_normal_layer_2->forward(x);
        x = torch::tanh(x);

        x = fully_connected_layer_3->forward(x);
        x = batch_normal_layer_3->forward(x);
        x = torch::tanh(x);

        x = fully_connected_layer_4->forward(x);
        x = torch::sigmoid(x);

        torch::Tensor action = x.squeeze(0);

        return action;
    };

private:
    torch::nn::Linear fully_connected_layer_1;
    torch::nn::BatchNorm1d batch_normal_layer_1;
    torch::nn::Linear fully_connected_layer_2;
    torch::nn::BatchNorm1d batch_normal_layer_2;
    torch::nn::Linear fully_connected_layer_3;
    torch::nn::BatchNorm1d batch_normal_layer_3;
    torch::nn::Linear fully_connected_layer_4;

    void resetParameters() {
        this->fully_connected_layer_1->weight.data().normal_(0, 0.1);
        this->fully_connected_layer_2->weight.data().normal_(0, 0.1);
        this->fully_connected_layer_3->weight.data().normal_(0, 0.1);
        this->fully_connected_layer_4->weight.data().normal_(0, 0.1);
    };

};


class Critic : public torch::nn::Module
{
public:
    Critic(int state_dim, int action_dim, int hidden_unites)
        : fully_connected_layer_1(torch::nn::Linear(state_dim, hidden_unites)),
          batch_normal_layer_1(torch::nn::BatchNorm1d(hidden_unites)),
          fully_connected_layer_2(torch::nn::Linear(hidden_unites+action_dim, hidden_unites)),
          batch_normal_layer_2(torch::nn::BatchNorm1d(hidden_unites)),
          fully_connected_layer_3(torch::nn::Linear(hidden_unites, Q_VALUE_SIZE))
    {
        register_module("fully_connected_layer_1", fully_connected_layer_1);
        register_module("batch_normal_layer_1", batch_normal_layer_1);
        register_module("fully_connected_layer_2", fully_connected_layer_2);
        register_module("batch_normal_layer_2", batch_normal_layer_2);
        register_module("fully_connected_layer_3", fully_connected_layer_3);
        resetParameters();
    };

    torch::Tensor forward(torch::Tensor state, torch::Tensor action) {
        torch::Tensor x;
        x = fully_connected_layer_1->forward(state);
        x = batch_normal_layer_1->forward(x);
        x = torch::relu(x);

        x = torch::cat({x, action}, 1);
        x = fully_connected_layer_2->forward(x);
        x = batch_normal_layer_2->forward(x);
        x = torch::relu(x);

        torch::Tensor q_value = fully_connected_layer_3->forward(x);

        return q_value;
    };

private:
    torch::nn::Linear fully_connected_layer_1;
    torch::nn::BatchNorm1d batch_normal_layer_1;
    torch::nn::Linear fully_connected_layer_2;
    torch::nn::BatchNorm1d batch_normal_layer_2;
    torch::nn::Linear fully_connected_layer_3;

    const int Q_VALUE_SIZE = 1;

    void resetParameters() {
        this->fully_connected_layer_1->weight.data().normal_(0, 0.1);
        this->fully_connected_layer_2->weight.data().normal_(0, 0.1);
        this->fully_connected_layer_3->weight.data().normal_(0, 0.1);
    };

};

#endif // MODEL_H
