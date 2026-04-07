// ShaderSources.swift
// All Metal shader code embedded as a Swift string.
//
// Sections:
//   1. Shared types       (Particle, SimParams, CameraUniforms)
//   2. SPH kernel math    (poly6, spikyGrad, viscLaplacian)
//   3. Grid helpers
//   4. Compute kernels    (clearGrid, buildGrid, density, forces, integrate)
//   5. Render shaders     (sphere impostor, bounding-box wireframe)

let metalShaderSource = """
#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

// ─────────────────────────────────────────────────────────────────────────────
// 1.  SHARED TYPES  — must mirror Types.swift byte-for-byte
// ─────────────────────────────────────────────────────────────────────────────

struct Particle {
    float4 positionDensity;   // xyz = position,  w = density
    float4 velocityPressure;  // xyz = velocity,  w = pressure
    float4 force;             // xyz = force,     w = unused
};

struct SimParams {
    uint  numParticles;
    float h, mass, restDensity;       // row 1

    float gasConstant, viscosity, dt; // row 2
    float _pad0;

    float boundMinX, boundMinY, boundMinZ, cellSize; // row 3
    float boundMaxX, boundMaxY, boundMaxZ;           // row 4
    uint  gridDimX;

    uint  gridDimY, gridDimZ, maxPerCell;            // row 5
    uint  _pad1;

    float gravX, gravY, gravZ;                       // row 6  — box-local gravity
    float _pad2;
};

struct CameraUniforms {
    float4x4 viewProj;        // clip  = viewProj * world
    float4x4 model;           // world = model    * boxLocal
    float4   cameraRight;
    float4   cameraUp;
    float4   cameraPos;
};

// ─────────────────────────────────────────────────────────────────────────────
// 2.  SPH KERNEL FUNCTIONS
// ─────────────────────────────────────────────────────────────────────────────

// Poly6 — density estimation
// W(r²,h) = 315/(64π h⁹) · (h²–r²)³     for r ≤ h
inline float poly6(float r2, float h) {
    float h2 = h * h;
    if (r2 >= h2) return 0.0;
    float t = h2 - r2;
    return (315.0 / (64.0 * M_PI_F)) * (t * t * t) / pow(h, 9.0);
}

// Spiky gradient — pressure forces
// ∇W(rij,r,h) = –45/(π h⁶) · (h–r)² · r̂    for r ≤ h
inline float3 spikyGrad(float3 rij, float r, float h) {
    if (r >= h || r < 1e-5) return float3(0.0);
    float t = h - r;
    return (-45.0 / (M_PI_F * pow(h, 6.0))) * t * t * (rij / r);
}

// Viscosity Laplacian
// ∇²W(r,h) = 45/(π h⁶) · (h–r)           for r ≤ h
inline float viscLaplacian(float r, float h) {
    if (r >= h) return 0.0;
    return (45.0 / (M_PI_F * pow(h, 6.0))) * (h - r);
}

// ─────────────────────────────────────────────────────────────────────────────
// 3.  GRID HELPERS
// ─────────────────────────────────────────────────────────────────────────────

inline int3 worldToCell(float3 pos, constant SimParams& p) {
    return int3((pos - float3(p.boundMinX, p.boundMinY, p.boundMinZ)) / p.cellSize);
}

inline uint cellIndex(int3 c, constant SimParams& p) {
    if (any(c < int3(0)) || any(c >= int3(p.gridDimX, p.gridDimY, p.gridDimZ)))
        return UINT_MAX;
    return (uint)c.x + (uint)c.y * p.gridDimX + (uint)c.z * p.gridDimX * p.gridDimY;
}

// ─────────────────────────────────────────────────────────────────────────────
// 4.  COMPUTE KERNELS
// ─────────────────────────────────────────────────────────────────────────────

// ── clearGrid ─────────────────────────────────────────────────────────────────
kernel void clearGrid(
    device atomic_uint* gridCount [[ buffer(0) ]],
    constant SimParams& params    [[ buffer(1) ]],
    uint tid [[ thread_position_in_grid ]]
) {
    if (tid >= params.gridDimX * params.gridDimY * params.gridDimZ) return;
    atomic_store_explicit(&gridCount[tid], 0u, memory_order_relaxed);
}

// ── buildGrid ─────────────────────────────────────────────────────────────────
kernel void buildGrid(
    device const Particle* particles     [[ buffer(0) ]],
    device atomic_uint*    gridCount     [[ buffer(1) ]],
    device uint*           gridParticles [[ buffer(2) ]],
    constant SimParams&    params        [[ buffer(3) ]],
    uint tid [[ thread_position_in_grid ]]
) {
    if (tid >= params.numParticles) return;
    int3 cell = worldToCell(particles[tid].positionDensity.xyz, params);
    cell = clamp(cell, int3(0), int3(params.gridDimX, params.gridDimY, params.gridDimZ) - 1);
    uint cIdx = (uint)cell.x + (uint)cell.y * params.gridDimX
              + (uint)cell.z * params.gridDimX * params.gridDimY;
    uint slot = atomic_fetch_add_explicit(&gridCount[cIdx], 1u, memory_order_relaxed);
    if (slot < params.maxPerCell)
        gridParticles[cIdx * params.maxPerCell + slot] = tid;
}

// ── computeDensityPressure ────────────────────────────────────────────────────
kernel void computeDensityPressure(
    device Particle*    particles     [[ buffer(0) ]],
    device const uint*  gridCount     [[ buffer(1) ]],
    device const uint*  gridParticles [[ buffer(2) ]],
    constant SimParams& params        [[ buffer(3) ]],
    uint tid [[ thread_position_in_grid ]]
) {
    if (tid >= params.numParticles) return;
    float3 pi   = particles[tid].positionDensity.xyz;
    int3   cell = worldToCell(pi, params);
    float  density = 0.0;
    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                uint cIdx = cellIndex(cell + int3(dx, dy, dz), params);
                if (cIdx == UINT_MAX) continue;
                uint count = min(gridCount[cIdx], params.maxPerCell);
                for (uint k = 0; k < count; k++) {
                    uint j   = gridParticles[cIdx * params.maxPerCell + k];
                    float3 r = pi - particles[j].positionDensity.xyz;
                    density += params.mass * poly6(dot(r, r), params.h);
                }
            }
        }
    }
    density = max(density, params.restDensity * 0.01);
    // Clamp pressure to >= 0: negative pressure (ρ < ρ₀) would create
    // artificial cohesion (tensile instability) at the free surface.
    float pressure = params.gasConstant * (density - params.restDensity);
    pressure = max(pressure, 0.0);
    particles[tid].positionDensity.w  = density;
    particles[tid].velocityPressure.w = pressure;
}

// ── computeForces ─────────────────────────────────────────────────────────────
kernel void computeForces(
    device Particle*    particles     [[ buffer(0) ]],
    device const uint*  gridCount     [[ buffer(1) ]],
    device const uint*  gridParticles [[ buffer(2) ]],
    constant SimParams& params        [[ buffer(3) ]],
    uint tid [[ thread_position_in_grid ]]
) {
    if (tid >= params.numParticles) return;
    float3 pi   = particles[tid].positionDensity.xyz;
    float3 vi   = particles[tid].velocityPressure.xyz;
    float  rhoi = particles[tid].positionDensity.w;
    float  Pi   = particles[tid].velocityPressure.w;
    int3   cell = worldToCell(pi, params);
    float3 fp = 0, fv = 0;
    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                uint cIdx = cellIndex(cell + int3(dx, dy, dz), params);
                if (cIdx == UINT_MAX) continue;
                uint count = min(gridCount[cIdx], params.maxPerCell);
                for (uint k = 0; k < count; k++) {
                    uint j = gridParticles[cIdx * params.maxPerCell + k];
                    if (j == tid) continue;
                    float3 rij  = pi - particles[j].positionDensity.xyz;
                    float  r    = length(rij);
                    if (r >= params.h || r < 1e-6) continue;
                    float  rhoj = particles[j].positionDensity.w;
                    float  Pj   = particles[j].velocityPressure.w;
                    float3 vj   = particles[j].velocityPressure.xyz;
                    // Symmetric pressure (conserves momentum)
                    fp += -params.mass * (Pi/(rhoi*rhoi) + Pj/(rhoj*rhoj))
                        * spikyGrad(rij, r, params.h);
                    // Viscosity
                    fv += params.mass * params.viscosity
                        * (vj - vi) / rhoj * viscLaplacian(r, params.h);
                }
            }
        }
    }
    // Gravity in box-local frame (updated every frame as the box tilts)
    float3 fg = float3(params.gravX, params.gravY, params.gravZ) * rhoi;
    particles[tid].force = float4(fp + fv + fg, 0.0);
}

// ── integrate ─────────────────────────────────────────────────────────────────
// 1. Symplectic Euler  (force → velocity → position)
// 2. XSPH velocity correction (Schechter & Bridson 2012)
//    v̂ᵢ = vᵢ + ε · Σⱼ (mⱼ/ρ̄ᵢⱼ) · (vⱼ − vᵢ) · W(rᵢⱼ, h)
//    ε = 0.15  — particles partially adopt the local flow velocity
//    → eliminates particle clumping, makes the bulk look like a continuous fluid
// 3. Reflective walls — low restitution (water barely bounces)
kernel void integrate(
    device Particle*    particles     [[ buffer(0) ]],
    constant SimParams& params        [[ buffer(1) ]],
    device const uint*  gridCount     [[ buffer(2) ]],
    device const uint*  gridParticles [[ buffer(3) ]],
    uint tid [[ thread_position_in_grid ]]
) {
    if (tid >= params.numParticles) return;

    float  rho = particles[tid].positionDensity.w;
    float3 acc = particles[tid].force.xyz / rho;

    // ── Step 1: Euler velocity update ───────────────────────────────────────
    float3 vel = particles[tid].velocityPressure.xyz + acc * params.dt;

    // ── Step 2: XSPH correction ─────────────────────────────────────────────
    // Blends each particle's velocity toward the local neighbourhood average.
    // This is a purely numerical regularisation — it does not add or remove
    // energy, it just smooths the velocity field so the bulk moves coherently.
    const float eps = 0.15;
    float3 pi   = particles[tid].positionDensity.xyz;
    int3   cell = worldToCell(pi, params);
    float3 xsph = float3(0.0);

    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                uint cIdx = cellIndex(cell + int3(dx, dy, dz), params);
                if (cIdx == UINT_MAX) continue;
                uint count = min(gridCount[cIdx], params.maxPerCell);
                for (uint k = 0; k < count; k++) {
                    uint j = gridParticles[cIdx * params.maxPerCell + k];
                    if (j == tid) continue;
                    float3 rij  = pi - particles[j].positionDensity.xyz;
                    float  r2   = dot(rij, rij);
                    float  rhoj = particles[j].positionDensity.w;
                    float  rhoBar = 0.5 * (rho + rhoj);
                    float3 vj   = particles[j].velocityPressure.xyz;
                    xsph += (params.mass / rhoBar) * (vj - vel) * poly6(r2, params.h);
                }
            }
        }
    }
    vel += eps * xsph;

    // ── Step 3: position update ─────────────────────────────────────────────
    float3 pos = pi + vel * params.dt;

    // ── Step 4: reflective walls — very low restitution (water ≈ 0.05) ─────
    // High restitution causes the "rubber-ball" effect seen before.
    // Floor is fully inelastic (water puddles, doesn't bounce up).
    // Side walls allow a tiny bounce so the particle escapes the wall zone.
    float3 lo = float3(params.boundMinX, params.boundMinY, params.boundMinZ);
    float3 hi = float3(params.boundMaxX, params.boundMaxY, params.boundMaxZ);
    if (pos.x < lo.x) { pos.x = lo.x;  vel.x =  abs(vel.x) * 0.05; }
    if (pos.x > hi.x) { pos.x = hi.x;  vel.x = -abs(vel.x) * 0.05; }
    if (pos.y < lo.y) { pos.y = lo.y;  vel.y =  abs(vel.y) * 0.02; }
    if (pos.y > hi.y) { pos.y = hi.y;  vel.y = -abs(vel.y) * 0.02; }
    if (pos.z < lo.z) { pos.z = lo.z;  vel.z =  abs(vel.z) * 0.05; }
    if (pos.z > hi.z) { pos.z = hi.z;  vel.z = -abs(vel.z) * 0.05; }

    particles[tid].positionDensity.xyz  = pos;
    particles[tid].velocityPressure.xyz = vel;
}

// ─────────────────────────────────────────────────────────────────────────────
// 5.  RENDER SHADERS
// ─────────────────────────────────────────────────────────────────────────────

// ── Plasma colormap ───────────────────────────────────────────────────────────
// Piecewise-linear approximation of matplotlib's "plasma" LUT.
// t=0 → dark purple (slow particle) | t=1 → bright yellow (fast particle)
inline float3 plasma(float t) {
    t = saturate(t);
    float3 c[5] = {
        float3(0.0508, 0.0298, 0.5280),   // 0.00 — indigo
        float3(0.4949, 0.0120, 0.6579),   // 0.25 — violet
        float3(0.7982, 0.2802, 0.4695),   // 0.50 — rose
        float3(0.9734, 0.5859, 0.2515),   // 0.75 — amber
        float3(0.9400, 0.9752, 0.1313)    // 1.00 — yellow
    };
    float s = t * 4.0;
    int   i = (int)min(s, 3.0);
    return mix(c[i], c[i+1], s - (float)i);
}

// ── Particle vertex — sphere impostor billboard ───────────────────────────────
struct VertexOut {
    float4 clipPos [[ position ]];
    float2 uv;
    float3 color;
};

vertex VertexOut particleVertex(
    uint                     vertexID   [[ vertex_id   ]],
    uint                     instanceID [[ instance_id ]],
    device const Particle*   particles  [[ buffer(0)   ]],
    constant CameraUniforms& cam        [[ buffer(1)   ]],
    constant float&          radius     [[ buffer(2)   ]]
) {
    const float2 corners[4] = {
        float2(-1,-1), float2(1,-1), float2(-1,1), float2(1,1)
    };
    float3 center = particles[instanceID].positionDensity.xyz;
    float2 corner = corners[vertexID];

    // Transform particle center from box space → world space
    float3 worldCenter = (cam.model * float4(center, 1.0)).xyz;

    // Billboard: expand along camera axes in world space
    float3 worldPos = worldCenter
                    + cam.cameraRight.xyz * corner.x * radius
                    + cam.cameraUp.xyz    * corner.y * radius;

    float speed = length(particles[instanceID].velocityPressure.xyz);
    // With k=20000, wave impacts reach ~3-4 m/s → use full color range
    float t     = saturate(speed / 3.5);

    VertexOut out;
    out.clipPos = cam.viewProj * float4(worldPos, 1.0);
    out.uv      = corner;
    out.color   = plasma(t);
    return out;
}

// ── Particle fragment — physically-inspired water sphere ──────────────────────
//
// Lighting pipeline:
//   • Reconstructed sphere normal from the UV impostor
//   • Phong diffuse (warm key + cool fill)
//   • Blinn-Phong specular — tight highlight (wet surface)
//   • Fresnel–Schlick — grazing angles appear more reflective
//   • Rim light — halo on silhouette from a backlight
//   • Soft alpha at edge (smooth-step) — particles blend naturally

fragment float4 particleFragment(VertexOut in [[ stage_in ]]) {
    float r2 = dot(in.uv, in.uv);
    // Soft edge: opaque core, fade over outer 20%
    float alpha = 1.0 - smoothstep(0.60, 1.0, r2);
    if (alpha < 0.005) discard_fragment();

    float3 N = normalize(float3(in.uv, sqrt(max(0.0, 1.0 - r2))));
    float3 V = float3(0, 0, 1);   // view direction in impostor space

    float3 L_key  = normalize(float3( 1.5,  2.0,  1.2));
    float3 L_fill = normalize(float3(-1.0,  0.5, -0.5));
    float3 L_back = normalize(float3(-0.4, -1.0,  0.7));

    float diff  = max(dot(N, L_key),  0.0) * 0.70
                + max(dot(N, L_fill), 0.0) * 0.18;

    float3 H    = normalize(L_key + V);
    float  spec = pow(max(dot(N, H), 0.0), 120.0) * 0.90;

    // Fresnel–Schlick: more reflective at grazing angles (water IOR ≈ 1.33)
    float cosT    = max(dot(N, V), 0.0);
    float fresnel = 0.02 + 0.98 * pow(1.0 - cosT, 5.0);

    float rim = pow(max(dot(N, L_back), 0.0), 4.0) * 0.45;

    float3 col   = in.color;
    float3 final = col * 0.06                                       // ambient
                 + col * diff                                       // diffuse
                 + float3(0.90, 0.95, 1.00) * spec                 // specular
                 + mix(col, float3(0.50, 0.78, 1.00), fresnel) * fresnel * 0.40
                 + float3(0.30, 0.55, 1.00) * rim;                 // rim

    return float4(final, alpha);
}

// ── Bounding-box wireframe ────────────────────────────────────────────────────
// 24 float4 vertices (12 edges × 2 endpoints) uploaded by CPU, drawn as lines.

vertex float4 boxVertex(
    uint                     vid  [[ vertex_id ]],
    device const float4*   verts  [[ buffer(0) ]],
    constant CameraUniforms& cam  [[ buffer(1) ]]
) {
    // Box vertices are in box-local space → apply model then projection
    return cam.viewProj * cam.model * verts[vid];
}

fragment float4 boxFragment() {
    // Pale steel-blue at low opacity — just visible, not distracting
    return float4(0.55, 0.75, 1.00, 0.18);
}
"""
