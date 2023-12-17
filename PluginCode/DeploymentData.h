//
// Created by u153171 on 11/30/2023.
//

#pragma once

struct DPLData
{
    // groove information
    torch::Tensor groove_hits = torch::zeros({1, 32, 1}, torch::kFloat32);
    torch::Tensor groove_velocities = torch::zeros({1, 32, 1}, torch::kFloat32);
    torch::Tensor groove_offsets = torch::zeros({1, 32, 1}, torch::kFloat32);

    // following will be used for the inference
    // contains the above information in a single tensor of shape (1, 32, 27)
    torch::Tensor groove_hvo = torch::zeros({1, 32, 27}, torch::kFloat32);

    // Latent vector
    torch::Tensor latent_vector = torch::zeros({1, 128}, torch::kFloat32);
};

