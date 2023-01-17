#ifndef REINFORCEMENT_LEARNING_H
#define REINFORCEMENT_LEARNING_H

#include <torch/torch.h>
#include <vector>
#include <fstream>
#include <memory>
#include <random>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <QDebug>

#include "model.h"
#include "utilities.h"

class DDPG {
public:
    DDPG(int action_dim, int state_dim)
        : action_dim(action_dim),
          state_dim(state_dim),
          device(torch::kCUDA)
    {
        memory = std::make_shared<ExperienceMemory>(action_dim, state_dim);
        if(torch::cuda::is_available()) {
            device_type = torch::kCUDA;
            qDebug() << "using GPU";
        } else {
            device_type = torch::kCPU;
        }
        device = torch::Device(device_type);

        actor_online = std::make_shared<Actor>(state_dim, action_dim, HIDDEN_UNITES);
        actor_target = std::make_shared<Actor>(state_dim, action_dim, HIDDEN_UNITES);
        actor_model_file.open("actor.pt");
        if(actor_model_file) {
            loadActor();
        }
        actor_online->to(device);
        actor_target->to(device);
        actor_optimizer = std::make_shared<torch::optim::Adam>(actor_online->parameters(), ACTOR_LEARNING_RATE);

        critic_online = std::make_shared<Critic>(state_dim, action_dim, HIDDEN_UNITES);
        critic_target = std::make_shared<Critic>(state_dim, action_dim, HIDDEN_UNITES);
        critic_model_file.open("critic.pt");
        if(critic_model_file) {
            loadCritic();
        }
        critic_online->to(device);
        critic_target->to(device);
        critic_optimizer = std::make_shared<torch::optim::Adam>(critic_online->parameters(), CRITIC_LEARNING_RATE);

        noise = std::make_shared<OUNoise>(action_dim);
    };

    std::vector<float> selectAction(std::vector<float> &current_state_vector, bool add_noise=true) {
        current_state = torch::from_blob(current_state_vector.data(), {state_dim}).to(device);
        actor_online->eval();
        torch::NoGradGuard no_grad;
        action = actor_online->forward(current_state.unsqueeze(0)).to(device);
        if(add_noise)
            action = action + torch::from_blob(noise->sampleNoise().data(), {action_dim}).to(device);
        actor_online->train();
        // TODO: clip action.
        std::vector<float> action_vector(action.data_ptr<float>(), action.data_ptr<float>()+action.numel());
        return action_vector;
    };

    void learn() {
        Experience experiences = memory->getExperiences(BATCH_SIZE);
        current_states = experiences.current_state;
        actions = experiences.action;
        rewards = experiences.reward;
        next_states = experiences.next_state;
        // Critic Update.
        next_actions = actor_target->forward(next_states).to(device);
        torch::NoGradGuard no_grad;
        q_reward_targets_next = critic_target->forward(next_states, next_actions).to(device);
        q_reward_targets = rewards + GAMMA * q_reward_targets_next;
        critic_loss = torch::mse_loss(q_reward_targets, q_reward_expected);
        critic_optimizer->zero_grad();
        critic_loss.backward();
        torch::nn::utils::clip_grad_norm_(critic_online->parameters(), 1);
        critic_optimizer->step();
        // Actor update
        action_predictions = actor_online->forward(current_states).to(device);
        actor_loss = -critic_online->forward(current_states, action_predictions).mean().to(device);
        actor_optimizer->zero_grad();
        actor_loss.backward();
        actor_optimizer->step();
        // Soft update
        softUpdate(critic_target, critic_online);
        softUpdate(actor_target, actor_online);
    };

    bool isMemoryFull() {
        return memory->isMemoryFull();
    };

    void storeExperience(std::vector<float> &current_state_vector, std::vector<float> &action_vector, float reward, std::vector<float> &next_state_vector) {
        torch::Tensor current_state_tensor = torch::from_blob(current_state_vector.data(), {state_dim}).to(device);
        torch::Tensor action_tensor = torch::from_blob(action_vector.data(), {action_dim}).to(device);
        torch::Tensor reward_tensor = torch::mul(torch::ones(1), reward).to(device);
        torch::Tensor next_state_tensor = torch::from_blob(next_state_vector.data(), {state_dim}).to(device);

        memory->storeExperience(current_state_tensor, action_tensor, reward_tensor, next_state_tensor);
    };

    void saveActor() {
        torch::save(std::dynamic_pointer_cast<torch::nn::Module>(actor_online), "actor.pt");
    };

    void saveCritic() {
        torch::save(std::dynamic_pointer_cast<torch::nn::Module>(critic_online), "critic.pt");
    };

    void loadActor() {
        torch::load(actor_online, "actor.pt");
        torch::load(actor_target, "actor.pt");
    };

    void loadCritic() {
        torch::load(critic_online, "critic.pt");
        torch::load(critic_target, "critic.pt");
    };

    ~DDPG() {
        actor_model_file.close();
        critic_model_file.close();
    };

private:
    int action_dim;
    int state_dim;
    std::shared_ptr<ExperienceMemory> memory;
    std::shared_ptr<Actor> actor_online;
    std::shared_ptr<Actor> actor_target;
    std::shared_ptr<Critic> critic_online;
    std::shared_ptr<Critic> critic_target;
    std::ifstream actor_model_file;
    std::ifstream critic_model_file;
    std::shared_ptr<OUNoise> noise;
    torch::DeviceType device_type;
    torch::Device device;
    std::shared_ptr<torch::optim::Adam> actor_optimizer;
    std::shared_ptr<torch::optim::Adam> critic_optimizer;

    torch::Tensor current_state;
    torch::Tensor action;

    torch::Tensor current_states;
    torch::Tensor actions;
    torch::Tensor rewards;
    torch::Tensor next_states;
    torch::Tensor next_actions;
    torch::Tensor q_reward_targets_next;
    torch::Tensor q_reward_targets;
    torch::Tensor q_reward_expected;
    torch::Tensor critic_loss;
    torch::Tensor action_predictions;
    torch::Tensor actor_loss;

    const int HIDDEN_UNITES = 50;
    const float ACTOR_LEARNING_RATE = 0.0001;
    const float CRITIC_LEARNING_RATE = 0.001;
    const float TAU = 0.001;
    const int BATCH_SIZE = 128;
    const float GAMMA = 0.98;

//    void clip(torch::Tensor &action) {
//        for(int i = 0; i < action_dim; i++) {
//            action[i] = std::max(0, std::min(action[i], 1));
//        }
//    };

    void softUpdate(std::shared_ptr<torch::nn::Module> target, std::shared_ptr<torch::nn::Module> local) {
        torch::NoGradGuard no_grad;
        for(int i = 0; i < target->parameters().size(); i++) {
            target->parameters()[i].copy_(TAU*local->parameters()[i] + (1-TAU)*target->parameters()[i]);
        }
    };
};

#endif // REINFORCEMENT_LEARNING_H
