#ifndef ENVIRONMENT_H
#define ENVIRONMENT_H

#include <memory>
#include <cmath>
#include <vector>
#include <QDebug>
#include <Eigen/Dense>
#include "reinforcement_learning.h"
#include "utilities.h"

class SerpenBot {
public:
    SerpenBot(int action_dim, int state_dim)
        : action_dim(action_dim),
          state_dim(state_dim)
    {
        agent = std::make_shared<DDPG>(action_dim, state_dim);
    };

    SerpenBot(int action_dim, int state_dim, State &init_state, State &goal_state)
        : action_dim(action_dim),
          state_dim(state_dim),
          current_state(init_state),
          goal_state(goal_state)
    {
        agent = std::make_shared<DDPG>(action_dim, state_dim);
    };

    void initState(State &init_state) {
        setCurrentState(init_state);
    };

    void setGoalState(State &goal_state) {
        this->goal_state = goal_state;
    };

    State &getGoalState() {
        return goal_state;
    };

    void setCurrentState(State &current_state) {
        this->current_state = current_state;
    };

    State &getCurrentState() {
        return current_state;
    };

    float calculateReward(State &goal_state, State &next_state) {
        this->goal_state = goal_state;
        return calculateReward(next_state);
    };

    double calculateReward(State &next_state) {
        this->reward = -std::sqrt(std::pow(next_state.x-goal_state.x, 2)+std::pow(next_state.y-goal_state.y, 2)) - std::abs(next_state.angle-goal_state.angle);
        return this->reward;
    };

    Action &selectAction(State &current_state, bool add_noise=false) {
        this->current_state = current_state;
        std::vector<float> current_state_vector;
        current_state_vector.push_back(current_state.x);
        current_state_vector.push_back(current_state.y);
        current_state_vector.push_back(current_state.angle);
        std::vector<float> action_vector = agent->selectAction(current_state_vector, add_noise);
        action_probability.laser_current = action_vector[0];
        action_probability.laser_frequency = action_vector[1];
        action.laser_current = (MAX_LASER_CURRENT - MIN_LASER_CURRENT) * action_vector[0] + MIN_LASER_CURRENT;
        action.laser_frequency = (int)(MAX_LASER_FREQUENCY - MIN_LASER_FREQUENCY) * action_vector[1] + MIN_LASER_FREQUENCY;
        return action;
    };

    Action &control(State &goal_state, State &next_state) {
        this->goal_state = goal_state;
        return control(next_state);
    };

    Action &control(State &next_state) {
        current_state = next_state;
        return selectAction(next_state, false);
    };

    Action &learn(State &goal_state, State &next_state) {
        this->goal_state = goal_state;
        return learn(next_state);
    };

    Action &learn(State &next_state) {
        qDebug() << "next state x: " << next_state.x;
        qDebug() << "next state y: " << next_state.y;
        qDebug() << "next state angle: " << next_state.angle;
        selectAction(next_state, true);
        calculateReward(next_state);
        storeExperience(next_state);
        current_state = next_state;
        if(agent->isMemoryFull())
            agent->learn();
//        qDebug() << "action: " << action;
        return action;
    };

    void storeExperience(State &current_state, Action &action, float reward, State &next_state) {
        this->current_state = current_state;
        this->action = action;
        this->reward = reward;
        storeExperience(next_state);
    };

    void storeExperience(State &next_state) {
        std::vector<float> current_state_vector;
        current_state_vector.push_back(current_state.x);
        current_state_vector.push_back(current_state.y);
        current_state_vector.push_back(current_state.angle);

        // Store action probability.
        std::vector<float> action_vector;
        action_vector.push_back(action_probability.laser_current);
        action_vector.push_back(action_probability.laser_frequency);

        std::vector<float> next_state_vector;
        next_state_vector.push_back(next_state.x);
        next_state_vector.push_back(next_state.y);
        next_state_vector.push_back(next_state.angle);

        agent->storeExperience(current_state_vector, action_vector, reward, next_state_vector);
    };

    State move(Action &action) {
        // Kinematic model of the robot.
        Eigen::Vector3f next_state_vector;
        Eigen::Matrix3f rotation_matrix;
        rotation_matrix << std::cos(current_state.angle), std::sin(current_state.angle), 0,
                          -std::sin(current_state.angle), std::cos(current_state.angle), 0,
                                                       0,                             0, 1;
        Eigen::MatrixXf jacobian_matrix(3, 2);
        jacobian_matrix <<          0.5f,           0.5f,
                                    0.0f,           0.0f,
                           1/ROBOT_WIDTH, -1/ROBOT_WIDTH;
        Eigen::Vector2f legs_velocity_vector;
        legs_velocity_vector << actuatorSystemFunc(action.laser_current, action.laser_frequency, LEFT_ACTUATOR_RESONANCE_FREQUENCY, LEFT_GAIN_K, LEFT_DAMPING_RATIO_ZETA),
                                actuatorSystemFunc(action.laser_current, action.laser_frequency, RIGHT_ACTUATOR_RESONANCE_FREQUENCY, RIGHT_GAIN_K, RIGHT_DAMPING_RATIO_ZETA);
        Eigen::Vector3f current_state_vector;
        current_state_vector << current_state.x, current_state.y, current_state.angle;
        next_state_vector = rotation_matrix * jacobian_matrix * legs_velocity_vector + current_state_vector;
        State next_state;
        next_state.x = next_state_vector.coeff(0);
        next_state.y = next_state_vector.coeff(1);
        next_state.angle = next_state_vector.coeff(2);
        return next_state;
    };

    bool isDone(State &goal_state, State &next_state) {
        this->goal_state = goal_state;
        return isDone(next_state);
    };

    bool isDone(State &next_state) {
        float error = std::sqrt(std::pow(next_state.x-goal_state.x, 2)+std::pow(next_state.y-goal_state.y, 2)) + std::abs(next_state.angle-goal_state.angle);
        if(error < ERROR_TOLERANCE)
            return true;
        return false;
    };

    void saveAgent() {
        agent->saveActor();
        agent->saveCritic();
    };

private:
    int action_dim;
    int state_dim;
    const float ERROR_TOLERANCE = 0.1;
    const float MAX_LASER_CURRENT = 5.5;
    const float MIN_LASER_CURRENT = 3.5;
    const float MAX_LASER_FREQUENCY = 5000;
    const float MIN_LASER_FREQUENCY = 600;

    const int LEFT_ACTUATOR_RESONANCE_FREQUENCY = 1800;
    const int RIGHT_ACTUATOR_RESONANCE_FREQUENCY = 2000;
    const float LEFT_GAIN_K = 1.0;
    const float RIGHT_GAIN_K = 0.9;
    const float LEFT_DAMPING_RATIO_ZETA = 0.5;
    const float RIGHT_DAMPING_RATIO_ZETA = 0.5;

    const int ROBOT_WIDTH = 777;

    std::shared_ptr<DDPG> agent;
    State goal_state;
    State current_state;
    Action action;
    Action action_probability;
    float reward;

    float actuatorSystemFunc(int laser_current,
                             float laser_frequency,
                             int resonance_frequency,
                             float gain,
                             float damping_ratio) {
        float velocity;
        velocity = laser_current * gain * (1 / std::sqrt(std::pow(1-std::pow(laser_frequency/resonance_frequency,2),2)
                                                         + std::pow(2*damping_ratio*(laser_frequency/resonance_frequency),2)));
        return velocity;
    };
};

#endif // ENVIRONMENT_H
