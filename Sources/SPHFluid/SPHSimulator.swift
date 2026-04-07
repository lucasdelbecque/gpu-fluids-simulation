// SPHSimulator.swift
// GPU-side SPH simulation — five compute passes per step.
//
// The simulation always runs in box-local space (fixed AABB).
// When the box tilts in world space, the caller updates the gravity vector
// via updateGravity() so the fluid responds correctly.

import Foundation
import Metal
import simd

private let kMaxPerCell: UInt32 = 32

final class SPHSimulator {

    let device:       MTLDevice
    let commandQueue: MTLCommandQueue

    private var clearGridPSO:  MTLComputePipelineState!
    private var buildGridPSO:  MTLComputePipelineState!
    private var densityPSO:    MTLComputePipelineState!
    private var forcesPSO:     MTLComputePipelineState!
    private var integratePSO:  MTLComputePipelineState!

    private(set) var particleBuffer: MTLBuffer!
    private      var gridCountBuf:   MTLBuffer!
    private      var gridBuf:        MTLBuffer!
    private(set) var paramsBuf:      MTLBuffer!

    private(set) var numParticles: Int
    private(set) var params:       SimParams

    // MARK: - Init

    init(device: MTLDevice, numParticles: Int = 10_000) {
        self.device       = device
        self.commandQueue = device.makeCommandQueue()!
        self.numParticles = numParticles

        // ── Physical calibration ──────────────────────────────────────────────
        // h = 0.08 m, spacing s = h/2 = 0.04 m (ratio 2.0, ~27 neighbours)
        //
        // mass = ρ₀ × s³ = 1000 × 0.04³ = 0.064 kg
        //   → each particle represents a 4×4×4 cm cube of water.
        //   → at rest, ρ_computed = Σⱼ m·W(rⱼ,h) ≈ ρ₀  (pressure-free).
        //
        // gasConstant k:  speed of sound c_s = √(k/ρ₀)
        //   We need c_s >> max fluid velocity (~3 m/s for wave impacts).
        //   k = 20 000 → c_s = √20 ≈ 4.5 m/s  (weakly compressible, good waves)
        //   CFL check: dt × c_s / h = 0.0025 × 4.5 / 0.08 = 0.14  ✓  (< 0.4)
        //
        // viscosity μ = 0.03  (water-like, allows waves to propagate freely)
        //
        // Domain: wide & shallow → good sloshing aspect ratio (W:H ≈ 2:1)
        //   22 × 0.08 = 1.76 m wide
        //   14 × 0.08 = 1.12 m tall  (extra headroom for waves to climb)
        //   16 × 0.08 = 1.28 m deep
        //
        // 1/3 fill verification:
        //   colsX = (1.76 - 0.08) / 0.04 = 42
        //   colsZ = (1.28 - 0.08) / 0.04 = 30
        //   yMax   = 1.12 / 3 = 0.373 m  → 8 layers max = 0.32 m ≤ 0.373 m ✓
        //   capacity = 42 × 30 × 8 = 10 080  →  10 000 particles fit cleanly.
        let h:     Float  = 0.080
        let dimX:  UInt32 = 22
        let dimY:  UInt32 = 14
        let dimZ:  UInt32 = 16

        params = SimParams(
            numParticles: UInt32(numParticles),
            h:            h,
            mass:         0.064,        // kg  — one cubic-lattice cell of water
            restDensity:  1000.0,       // kg/m³  (water)
            gasConstant:  20_000.0,     // ← KEY FIX: was 250, now ×80 stronger
            viscosity:    0.03,         // ← KEY FIX: was 0.25, now water-like
            dt:           0.0025,       // s  (CFL safe with k=20 000)
            boundMinX: 0, boundMinY: 0, boundMinZ: 0,
            cellSize:     h,
            boundMaxX: Float(dimX) * h,
            boundMaxY: Float(dimY) * h,
            boundMaxZ: Float(dimZ) * h,
            gridDimX:  dimX,
            gridDimY:  dimY,
            gridDimZ:  dimZ,
            maxPerCell: kMaxPerCell,
            gravX:  0, gravY: -9.81, gravZ: 0
        )

        setupPipelines()
        setupBuffers()
        resetParticles()
    }

    // MARK: - Gravity update (called every frame when box tilts)

    /// Updates the gravity vector in box-local space and pushes it to the GPU buffer.
    /// When the box rotates by θ around the world Z axis, call with:
    ///   gx = -9.81 * sin(θ),  gy = -9.81 * cos(θ),  gz = 0
    func updateGravity(x: Float, y: Float, z: Float) {
        params.gravX = x
        params.gravY = y
        params.gravZ = z
        // Patch only the gravity row in the existing buffer (offset of gravX in SimParams).
        // It is simpler and safer to re-copy the whole struct — it is only 112 bytes.
        memcpy(paramsBuf.contents(), &params, MemoryLayout<SimParams>.stride)
    }

    // MARK: - Metal setup

    private func setupPipelines() {
        let opts = MTLCompileOptions()
        opts.languageVersion = .version3_0
        guard let lib = try? device.makeLibrary(source: metalShaderSource, options: opts) else {
            fatalError("[SPHSimulator] Metal shader compilation failed — check ShaderSources.swift")
        }
        func pso(_ name: String) -> MTLComputePipelineState {
            guard let fn = lib.makeFunction(name: name) else {
                fatalError("[SPHSimulator] kernel '\(name)' not found")
            }
            return try! device.makeComputePipelineState(function: fn)
        }
        clearGridPSO = pso("clearGrid")
        buildGridPSO = pso("buildGrid")
        densityPSO   = pso("computeDensityPressure")
        forcesPSO    = pso("computeForces")
        integratePSO = pso("integrate")
    }

    private func setupBuffers() {
        let numCells = Int(params.gridDimX * params.gridDimY * params.gridDimZ)
        particleBuffer = device.makeBuffer(
            length:  numParticles * MemoryLayout<Particle>.stride,
            options: .storageModeShared)!
        gridCountBuf = device.makeBuffer(
            length:  numCells * MemoryLayout<UInt32>.stride,
            options: .storageModeShared)!
        gridBuf = device.makeBuffer(
            length:  numCells * Int(kMaxPerCell) * MemoryLayout<UInt32>.stride,
            options: .storageModeShared)!
        paramsBuf = device.makeBuffer(
            bytes:   &params,
            length:  MemoryLayout<SimParams>.stride,
            options: .storageModeShared)!
    }

    // MARK: - Particle initialisation

    /// Fills the bottom 1/3 of the tank with particles stacked in a regular
    /// grid (spacing = h/2).  A tiny random jitter breaks the crystalline
    /// packing so the fluid relaxes more naturally in the first few frames.
    func resetParticles() {
        let ptr = particleBuffer.contents()
            .bindMemory(to: Particle.self, capacity: numParticles)

        // spacing = h/2 matches the calibration used for mass (ratio 2.0).
        let s:      Float = params.h * 0.5      // 0.04 m — matches mass calibration
        let margin: Float = s                   // one spacing away from walls

        let x0 = params.boundMinX + margin
        let z0 = params.boundMinZ + margin
        let y0 = params.boundMinY + margin
        // 1/3 of tank height — verified at init time to fit numParticles cleanly
        let yMax = params.boundMinY + (params.boundMaxY - params.boundMinY) / 3.0

        // Columns that fit within walls
        let colsX = max(1, Int((params.boundMaxX - params.boundMinX - 2 * margin) / s))
        let colsZ = max(1, Int((params.boundMaxZ - params.boundMinZ - 2 * margin) / s))

        // Seed a simple LCG for reproducible jitter
        var rng: UInt32 = 0x1234_5678
        func nextJitter() -> Float {
            rng = rng &* 1664525 &+ 1013904223
            let f = Float(rng >> 9) / Float(1 << 23)  // [0,1)
            return (f - 0.5) * s * 0.15               // ±7.5% of spacing
        }

        for i in 0..<numParticles {
            let layer = i / (colsX * colsZ)
            let rem   = i % (colsX * colsZ)
            let ix    = rem % colsX
            let iz    = rem / colsX

            let x = (x0 + Float(ix) * s + nextJitter())
                .clamped(to: (params.boundMinX + margin * 0.5)...(params.boundMaxX - margin * 0.5))
            let y = (y0 + Float(layer) * s + nextJitter())
                .clamped(to: (params.boundMinY + margin * 0.5)...yMax)
            let z = (z0 + Float(iz) * s + nextJitter())
                .clamped(to: (params.boundMinZ + margin * 0.5)...(params.boundMaxZ - margin * 0.5))

            ptr[i] = Particle(
                positionDensity:  SIMD4(x, y, z, params.restDensity),
                velocityPressure: SIMD4(0, 0, 0, 0),
                force:            SIMD4(0, 0, 0, 0)
            )
        }
    }

    // MARK: - Simulation step

    func step() {
        guard
            let cmdBuf = commandQueue.makeCommandBuffer(),
            let enc    = cmdBuf.makeComputeCommandEncoder()
        else { return }

        let numCells = Int(params.gridDimX * params.gridDimY * params.gridDimZ)

        encode(enc, pso: clearGridPSO, count: numCells) {
            enc.setBuffer(self.gridCountBuf, offset: 0, index: 0)
            enc.setBuffer(self.paramsBuf,   offset: 0, index: 1)
        }
        encode(enc, pso: buildGridPSO, count: numParticles) {
            enc.setBuffer(self.particleBuffer, offset: 0, index: 0)
            enc.setBuffer(self.gridCountBuf,   offset: 0, index: 1)
            enc.setBuffer(self.gridBuf,        offset: 0, index: 2)
            enc.setBuffer(self.paramsBuf,      offset: 0, index: 3)
        }
        encode(enc, pso: densityPSO, count: numParticles) {
            enc.setBuffer(self.particleBuffer, offset: 0, index: 0)
            enc.setBuffer(self.gridCountBuf,   offset: 0, index: 1)
            enc.setBuffer(self.gridBuf,        offset: 0, index: 2)
            enc.setBuffer(self.paramsBuf,      offset: 0, index: 3)
        }
        encode(enc, pso: forcesPSO, count: numParticles) {
            enc.setBuffer(self.particleBuffer, offset: 0, index: 0)
            enc.setBuffer(self.gridCountBuf,   offset: 0, index: 1)
            enc.setBuffer(self.gridBuf,        offset: 0, index: 2)
            enc.setBuffer(self.paramsBuf,      offset: 0, index: 3)
        }
        encode(enc, pso: integratePSO, count: numParticles) {
            enc.setBuffer(self.particleBuffer, offset: 0, index: 0)
            enc.setBuffer(self.paramsBuf,      offset: 0, index: 1)
            enc.setBuffer(self.gridCountBuf,   offset: 0, index: 2)
            enc.setBuffer(self.gridBuf,        offset: 0, index: 3)
        }

        enc.endEncoding()
        cmdBuf.commit()
    }

    // MARK: - Helpers

    private func encode(
        _ enc:  MTLComputeCommandEncoder,
        pso:    MTLComputePipelineState,
        count:  Int,
        setup:  () -> Void
    ) {
        enc.setComputePipelineState(pso)
        setup()
        let w  = pso.threadExecutionWidth
        enc.dispatchThreadgroups(
            MTLSize(width: (count + w - 1) / w, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1)
        )
    }
}
