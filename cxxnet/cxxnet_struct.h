#ifndef CXXNET_STRUCT_H
#define CXXNET_STRUCT_H
#pragma once

namespace cxxnet {
// Only one pair of layer-edge-layer is stored in GPU at one time
class Layer {
public:
    Layer(...);
protected:
    // Data Batch
    mshadow::Tensor<mshadow::cpu, 4> host_data_;
    mshadow::Tensor<mshadow::gpu, 4> device_data_;
    // Bias ...
    // Target Batch
    mshadow::Tensor<mshadow::cpu, 4> host_target_;
    mshadow::Tensor<mshadow::gpu, 4> host_target_;
    // Loss
    mshadow::Tensor<mshadow::cpu, 4> host_loss_;
    mshadow::Tensor<mshadow::gpu, 4> device_loss_;
    // Mask
    mshadow::Tensor<mshadow::cpu, 4> host_mask_;
    mshadow::Tensor<mshadow::gpu, 4> device_mask_;
    // Connections
    std::vector<Edge*> income_edges_;
    std::vector<Edge*> outcome_edges_;

    virtual void Active() = 0;
    virtual void GetLoss() = 0;
    virtual void Forward() = 0;
    virtual void Backward() = 0;
    virtual void GetGradient() = 0;
    virtual void Norm() = 0;
}; // class Node


class Edge {
public:
    Edge(Node &, Node &);
protected:
    mshadow::Tensor<mshadow::cpu, 2> host_weight_;
    mshadow::Tensor<mshadow::gpu, 2> device_weight_;
    // Deriv
    mshadow::Tensor<mshadow::cpu, 4> host_deriv_;
    mshadow::Tensor<mshadow::gpu, 4> device_deriv_;

    virtual void Update() = 0;
}; //class Edge
}; // namespace cxxnet

#endif // CXXNET_STRUCT_H
