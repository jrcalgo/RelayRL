#include <torch/script.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

struct RelayrlMlpLayer {
    int64_t in_dim;
    int64_t out_dim;
    float* weight_out_in;
    float* bias;
};

namespace {

float* copy_floats(const at::Tensor& tensor) {
    const auto bytes = static_cast<size_t>(tensor.numel()) * sizeof(float);
    auto* data = static_cast<float*>(std::malloc(bytes));
    if (data == nullptr) {
        throw std::bad_alloc();
    }
    std::memcpy(data, tensor.data_ptr<float>(), bytes);
    return data;
}

}  // namespace

extern "C" int relayrl_pt_mlp_layers(
    void* module_ptr,
    RelayrlMlpLayer** out_layers,
    int* out_count,
    char** out_error) {
    *out_layers = nullptr;
    *out_count = 0;
    *out_error = nullptr;
    std::vector<RelayrlMlpLayer> layers;
    try {
        auto* module = reinterpret_cast<torch::jit::script::Module*>(module_ptr);
        auto graph = module->get_method("forward").graph();
        for (const auto& node : graph->nodes()) {
            if (node->kind() != torch::jit::aten::linear) {
                continue;
            }
            if (node->inputs().size() < 3) {
                throw std::runtime_error("loaded policy linear layer has no bias");
            }
            at::Tensor weight = node->input(1)
                                    ->node()
                                    ->t(torch::jit::attr::value)
                                    .to(at::kCPU)
                                    .contiguous();
            at::Tensor bias = node->input(2)
                                  ->node()
                                  ->t(torch::jit::attr::value)
                                  .to(at::kCPU)
                                  .contiguous();
            if (weight.scalar_type() != at::kFloat || bias.scalar_type() != at::kFloat) {
                throw std::runtime_error("loaded policy linear weights are not f32");
            }
            if (weight.dim() != 2 || bias.dim() != 1 || weight.size(0) != bias.size(0)) {
                throw std::runtime_error("loaded policy linear shapes are inconsistent");
            }
            RelayrlMlpLayer layer{};
            layer.out_dim = weight.size(0);
            layer.in_dim = weight.size(1);
            layer.weight_out_in = copy_floats(weight);
            layer.bias = copy_floats(bias);
            layers.push_back(layer);
        }
        if (layers.empty()) {
            throw std::runtime_error("loaded policy graph has no linear layers");
        }
        auto* copied = static_cast<RelayrlMlpLayer*>(
            std::malloc(layers.size() * sizeof(RelayrlMlpLayer)));
        if (copied == nullptr) {
            throw std::bad_alloc();
        }
        std::memcpy(copied, layers.data(), layers.size() * sizeof(RelayrlMlpLayer));
        *out_layers = copied;
        *out_count = static_cast<int>(layers.size());
        return 0;
    } catch (const std::exception& error) {
        for (const auto& layer : layers) {
            std::free(layer.weight_out_in);
            std::free(layer.bias);
        }
        *out_error = ::strdup(error.what());
        return 1;
    }
}

extern "C" void relayrl_pt_mlp_layers_free(RelayrlMlpLayer* layers, int count) {
    if (layers == nullptr) {
        return;
    }
    for (int index = 0; index < count; ++index) {
        std::free(layers[index].weight_out_in);
        std::free(layers[index].bias);
    }
    std::free(layers);
}

extern "C" void relayrl_pt_string_free(char* text) {
    std::free(text);
}
