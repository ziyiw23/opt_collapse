// conformations.h - Header for protein embedding generation

#pragma once

#include <vector>
#include <string>

// Forward declarations
struct Vec3;
struct Atom;
struct Residue;
struct Environment;
class KDTree;

// Main function declarations
std::vector<Environment> generate_residue_environments(
    const std::vector<Atom>& all_atoms,
    const std::vector<Residue>& residues,
    float env_radius);

// Helper function declarations
Vec3 sample_functional_center(
    const std::vector<Atom>& residue_atoms,
    const std::string& residue_letter);