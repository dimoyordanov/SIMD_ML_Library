#include "weight_io.h"

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>

namespace ml {

namespace {
    constexpr char     kMagic[4]  = {'M', 'L', 'W', 'T'};
    constexpr uint32_t kVersion   = 1;
}

void save_weights(const Model& m, const std::string& path) {
    std::ofstream f{path, std::ios::binary};
    if (!f) throw std::runtime_error{"Cannot open for writing: " + path};

    f.write(kMagic, 4);
    f.write(reinterpret_cast<const char*>(&kVersion), sizeof(kVersion));

    const auto num_layers = static_cast<uint32_t>(m.layers().size());
    f.write(reinterpret_cast<const char*>(&num_layers), sizeof(num_layers));

    for (const auto& layer : m.layers()) {
        const auto in  = static_cast<uint32_t>(layer->in_size());
        const auto out = static_cast<uint32_t>(layer->out_size());
        f.write(reinterpret_cast<const char*>(&in),  sizeof(in));
        f.write(reinterpret_cast<const char*>(&out), sizeof(out));
        f.write(reinterpret_cast<const char*>(layer->weights().data()),
                static_cast<std::streamsize>(layer->weights().size() * sizeof(float)));
        f.write(reinterpret_cast<const char*>(layer->bias().data()),
                static_cast<std::streamsize>(layer->bias().size() * sizeof(float)));
    }
}

void load_weights(Model& m, const std::string& path) {
    std::ifstream f{path, std::ios::binary};
    if (!f) throw std::runtime_error{"Cannot open for reading: " + path};

    char magic[4]{};
    f.read(magic, 4);
    for (int i = 0; i < 4; ++i)
        if (magic[i] != kMagic[i])
            throw std::runtime_error{"Invalid file format: bad magic bytes"};

    uint32_t version{};
    f.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (version != kVersion)
        throw std::runtime_error{"Unsupported version: " + std::to_string(version)};

    uint32_t num_layers{};
    f.read(reinterpret_cast<char*>(&num_layers), sizeof(num_layers));
    if (num_layers != static_cast<uint32_t>(m.layers().size()))
        throw std::runtime_error{
            "Layer count mismatch: file has " + std::to_string(num_layers) +
            " layers, model has " + std::to_string(m.layers().size())};

    for (auto& layer : m.layers()) {
        uint32_t in{}, out{};
        f.read(reinterpret_cast<char*>(&in),  sizeof(in));
        f.read(reinterpret_cast<char*>(&out), sizeof(out));

        if (static_cast<int>(in)  != layer->in_size() ||
            static_cast<int>(out) != layer->out_size())
            throw std::runtime_error{"Layer shape mismatch"};

        std::vector<float> weights(static_cast<std::size_t>(in * out));
        std::vector<float> bias(static_cast<std::size_t>(out));

        f.read(reinterpret_cast<char*>(weights.data()),
               static_cast<std::streamsize>(weights.size() * sizeof(float)));
        f.read(reinterpret_cast<char*>(bias.data()),
               static_cast<std::streamsize>(bias.size() * sizeof(float)));

        layer->set_weights(std::move(weights));
        layer->set_bias(std::move(bias));
    }
}

}  // namespace ml
