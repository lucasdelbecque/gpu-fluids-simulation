// Types.swift
// Shared data structures between Swift (CPU) and Metal (GPU).
//
// Every struct that crosses the CPU/GPU boundary must have the same memory
// layout on both sides.  We use float4/uint4 padding to guarantee 16-byte
// alignment on every row.

import simd

// MARK: - Particle

/// One SPH particle (48 bytes, 16-byte aligned).
struct Particle {
    var positionDensity:  SIMD4<Float>   // xyz = position,  w = density ρ
    var velocityPressure: SIMD4<Float>   // xyz = velocity,  w = pressure P
    var force:            SIMD4<Float>   // xyz = force,     w = unused
}

// MARK: - Simulation Parameters

/// All physical constants and grid dimensions for every compute kernel.
/// 7 rows × 16 bytes = 112 bytes total.
struct SimParams {
    // row 1
    var numParticles: UInt32
    var h:            Float     // Smoothing radius (m)
    var mass:         Float     // Particle mass (kg)
    var restDensity:  Float     // ρ₀ (kg/m³)

    // row 2
    var gasConstant:  Float     // k  — pressure stiffness
    var viscosity:    Float     // μ  — dynamic viscosity
    var dt:           Float     // Time step (s)
    var _pad0:        Float = 0

    // row 3
    var boundMinX:    Float
    var boundMinY:    Float
    var boundMinZ:    Float
    var cellSize:     Float

    // row 4
    var boundMaxX:    Float
    var boundMaxY:    Float
    var boundMaxZ:    Float
    var gridDimX:     UInt32

    // row 5
    var gridDimY:     UInt32
    var gridDimZ:     UInt32
    var maxPerCell:   UInt32
    var _pad1:        UInt32 = 0

    // row 6 — gravity vector in box-local space (rotates with tilt each frame)
    var gravX:        Float
    var gravY:        Float
    var gravZ:        Float
    var _pad2:        Float = 0
}

// MARK: - Camera Uniforms

/// Sent to every vertex shader each frame.
/// viewProj (64) + model (64) + 3×float4 (48) = 176 bytes.
struct CameraUniforms {
    var viewProj:    simd_float4x4  // Combined view-projection matrix
    var model:       simd_float4x4  // Box-to-world transform (rotation around box centre)
    var cameraRight: SIMD4<Float>   // World-space camera right  (w = 0)
    var cameraUp:    SIMD4<Float>   // World-space camera up     (w = 0)
    var cameraPos:   SIMD4<Float>   // World-space camera origin (w = 0)
}
