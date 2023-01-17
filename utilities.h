#ifndef UTILITIES_H
#define UTILITIES_H

#include <torch/torch.h>
#include <vector>
#include <chrono>
#include <random>
#include <QQueue>
#include <QMutex>
#include <QDebug>

class OUNoise {
public:
    OUNoise(int action_dim=2, float mu=0.0, float theta=0.03, float sigma=0.07) {
        this->action_dim = action_dim;
        this->mu = mu;
        this->theta = theta;
        this->sigma = sigma;
        reset();
        engine = std::default_random_engine(seed);
        generator = std::normal_distribution<float>(0.0, 1.0);
    };

    std::vector<float> sampleNoise() {
        for(int i = 0; i < action_dim; i++) {
            ns_vector[i] = action_vector[i];
        }
        for(int i = 0; i < action_dim; i++) {
            dn_vector[i] = theta * (mu_vector[i] - ns_vector[i]) + sigma * generator(engine);
            action_vector[i] = ns_vector[i] + dn_vector[i];
        }
        return action_vector;
    };


private:
    int action_dim;
    float mu, theta, sigma;
    std::vector<float> mu_vector, action_vector, ns_vector, dn_vector;
    int seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine engine;
    std::normal_distribution<float> generator;
    void reset() {
        for(int i = 0; i < action_dim; i++) {
            mu_vector.push_back(mu);
            action_vector.push_back(mu);
            ns_vector.push_back(mu);
            dn_vector.push_back(mu);
        }
    };
};

class State {
public:
    float x;
    float y;
    float angle;
};

class Action {
public:
    float laser_current;
    float laser_frequency;
};

class Experience {
public:
    torch::Tensor current_state;
    torch::Tensor action;
    torch::Tensor reward;
    torch::Tensor next_state;
};

class ExperienceMemory {
public:
    ExperienceMemory(int action_dim=2, int state_dim=2) {
        this->action_dim = action_dim;
        this->state_dim = state_dim;
        this->current_index = 0;
        this->total_stored = 0;
        engine = std::default_random_engine(seed);
        generator = std::uniform_real_distribution<float>(0, MEMORY_CAPACITY);
    };

    void storeExperience(torch::Tensor &current_state, torch::Tensor &action, torch::Tensor &reward, torch::Tensor &next_state) {
        Experience experience;
        experience.current_state = current_state;
        experience.action = action;
        experience.reward = reward;
        experience.next_state = next_state;
        memory[current_index] = experience;
        current_index = (current_index + 1) % MEMORY_CAPACITY;
        total_stored = total_stored + 1;
    };

    Experience getExperiences(int batch_size) {
        Experience experiences;
        Experience experience;
        experience = getExperience();
        experiences = experience;
        for(int i = 0; i < batch_size-1; i++) {
            experience = getExperience();
            experiences.current_state = torch::cat({experiences.current_state, experience.current_state}, 0);
            experiences.action = torch::cat({experiences.action, experience.action}, 0);
            experiences.reward = torch::cat({experiences.reward, experience.reward}, 0);
            experiences.next_state = torch::cat({experiences.next_state, experience.next_state}, 0);
        }
        return experiences;
    };

    Experience getExperience() {
        Experience experience = memory[(int)generator(engine)];
        experience.current_state = torch::unsqueeze(experience.current_state, 0);
        experience.action = torch::unsqueeze(experience.action, 0);
        experience.reward = torch::unsqueeze(experience.reward, 0);
        experience.next_state = torch::unsqueeze(experience.next_state, 0);
        return experience;
    };

    bool isMemoryFull() {
        if(total_stored > MEMORY_CAPACITY)
            return true;
        return false;
    };

    static const int MEMORY_CAPACITY = 5000;
private:
    int action_dim;
    int state_dim;
    int current_index;
    int total_stored;
    std::array<Experience, MEMORY_CAPACITY> memory;
    int seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine engine;
    std::uniform_real_distribution<float> generator;
};

template <class T>
class AsyncVariable
{
public:
    AsyncVariable(){};
    AsyncVariable(const T &t) {
        variable_mutex.lock();
        this->t = t;
        variable_mutex.unlock();
    };

    AsyncVariable<T> &operator=(const AsyncVariable &other) {
        variable_mutex.lock();
        this->t = other.t;
        variable_mutex.unlock();
        return this->t;
    };

    void setValue(const T &value) {
        variable_mutex.lock();
        this->t = value;
        variable_mutex.unlock();
    };

    T &getValue() {
        variable_mutex.lock();
        T value = this->t;
        variable_mutex.unlock();
        return value;
    };

private:
    T t;
    QMutex variable_mutex;
};

template <class T>
class AsyncQueue : public QQueue<T>
{
public:
    AsyncQueue(){};
    T dequeue() {
        queue_mutex.lock();
        T element = QQueue<T>::dequeue();
        queue_mutex.unlock();
        return element;
    };

    void enqueue(const T &t) {
        queue_mutex.lock();
        QQueue<T>::enqueue(t);
        queue_mutex.unlock();
    };

    T &head() {
        queue_mutex.lock();
        T element = QQueue<T>::head();
        queue_mutex.unlock();
    };
    void swap(AsyncQueue<T> &other) {
        queue_mutex.lock();
        QQueue<T>::swap(other);
        queue_mutex.unlock();
    };

    bool isEmpty() {
        queue_mutex.lock();
        bool is_empty = QQueue<T>::isEmpty();
        queue_mutex.unlock();
        return is_empty;
    };

    void clear() {
        queue_mutex.lock();
        QQueue<T>::clear();
        queue_mutex.unlock();
    };

private:
    QMutex queue_mutex;
};

#endif // UTILITIES_H
