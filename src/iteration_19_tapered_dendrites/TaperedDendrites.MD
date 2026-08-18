# Tapered Dendrites Project

## Scientific Question

The fixed up-state value is hypothesized to result from a relatively uniform balance of excitatory (AMPA) and inhibitory (GABA) receptors such that the effective reversal potential is always the same. This project investigates the influence of the spatial distribution of stochastic E/I balance along a dendritic tree.

## Problem Statement

Standard cable theory models dendrites as uniform cylinders, but this approach has a major drawback: synaptic inputs modeled as delta functions show the same strength along the entire dendritic tree. In real dendrites, distal inputs produce stronger local depolarization than proximal ones due to decreasing diameter.

## Solution Approach

To address this limitation, we model the dendrite as a tapered cone rather than a uniform cylinder. This geometric approach captures the biophysical reality that distal dendritic compartments have higher input resistance, leading to stronger local depolarization for the same synaptic current.

## Current Implementation

The simulation is implemented in:
```
src/iteration_19_tapered_dendrites/
```

Key files include:
- `TaperedDendrites.py`: Core mathematical functions and unit tests
- `TaperedDendrites_improved.py`: TDD-improved version with comprehensive test suite
- `TaperredDendritesPDE.py`: Main PDE solver implementation
- `conical_data.py`: Geometric parameters for conical dendrites
- `DataTestCases.py`: Test data generation and validation
- Various test case files for different stimulation paradigms

## Mathematical Framework

The core model uses the Green's function approach (Eq. 32 from the referenced literature):

```
G_inf(x, x0, t, tau, Phi, chi, r_a, beta, lambda_of_x) = 
    prefactor * exp(exponent)
```

Where:
- `prefactor` involves spatial integration of the inverse space constant
- `exponent` combines temporal decay and spatial dispersion terms
- `lambda_of_x` represents the position-dependent space constant
- Other parameters represent membrane properties and channel densities

## Testing Strategy - Test-Driven Development (TDD)

This project employs rigorous Test-Driven Development (TDD) principles:

1. **Red**: Write failing tests that capture expected biological and mathematical behaviors
2. **Green**: Implement minimal code to make tests pass
3. **Refactor**: Improve code structure while maintaining test coverage
4. **Validate**: Compare against known analytical solutions and experimental data

Our TDD approach includes:
- Testing for physically impossible conditions (negative time → zero response)
- Validating positive responses for biologically plausible inputs
- Checking symmetry properties in uniform dendrites
- Verifying distance-dependent signal attenuation
- Comparing tapered vs. uniform dendrite behaviors
- Validating mathematical correctness of spatial integrals

## TDD Refactoring Completed

We have successfully refactored the original `TaperedDendrites.py` into `TaperedDendrites_improved.py` with:

- **Complete test suite**: 6 comprehensive unit tests covering edge cases and expected behaviors
- **Clear documentation**: Detailed docstrings explaining parameters, return values, and biological meaning
- **Parameter flexibility**: Functions now accept configurable parameters for easier experimentation
- **Error handling**: Proper handling of edge cases like non-positive time values
- **Test independence**: Each test validates a specific, verifiable property of the system

## Next Steps

1. **Review and validate**: Examine the improved implementation against original requirements
2. **Extend test coverage**: Add tests for more complex scenarios and parameter combinations
3. **Performance optimization**: Profile and optimize computationally intensive sections
4. **Integration testing**: Connect the improved core functions to the PDE solver
5. **Experimental validation**: Compare simulations with available experimental data
6. **Active extensions**: Incorporate voltage-gated channels and synaptic dynamics

## References

[To be filled with relevant neuroscience modeling literature]