// Renderer.swift
// MTKView delegate — drives simulation and renders each frame.
//
// Sloshing mechanism:
//   The simulation runs in box-local space (fixed AABB).
//   Each frame the box tilt angle θ is updated (auto sinusoidal or manual).
//   → The gravity vector in box-local space rotates: g_box = R(–θ) · g_world
//     so the fluid responds as if the box has physically tilted.
//   → The model matrix M = Translate(centre) · Rz(θ) · Translate(–centre)
//     is uploaded to the vertex shaders so particles and box wireframe
//     appear correctly tilted in world space.
//
// Draw order:  1. bounding-box wireframe (blended lines)
//              2. particles              (instanced sphere impostors, alpha blend)

import Metal
import MetalKit
import simd

final class Renderer: NSObject, MTKViewDelegate {

    let device:       MTLDevice
    let commandQueue: MTLCommandQueue
    let sim:          SPHSimulator
    let camera:       OrbitCamera

    private var particlePSO:  MTLRenderPipelineState!
    private var boxPSO:       MTLRenderPipelineState!
    private var depthFull:    MTLDepthStencilState!   // test + write
    private var depthTest:    MTLDepthStencilState!   // test only (for blended particles)
    private var cameraBuffer: MTLBuffer!
    private var boxVertBuf:   MTLBuffer!

    // Visual radius slightly > half-spacing → spheres touch and slightly overlap,
    // giving a continuous-looking fluid mass rather than visible individual balls.
    private var particleRadius: Float = 0.024

    // ── Sub-steps ─────────────────────────────────────────────────────────
    // dt=0.0025 × 6 steps = 0.015 s/frame ≈ real-time at 60 fps.
    var stepsPerFrame: Int = 6

    // ── Tilt / sloshing state ─────────────────────────────────────────────
    /// true = box oscillates automatically; false = user controls it manually
    var autoOscillate:   Bool  = true
    /// Current tilt angle of the box (radians, rotation around world Z)
    private var tiltAngle:     Float  = 0
    /// Angular velocity (rad/s) — used in manual mode or for impulses
    private var tiltVelocity:  Float  = 0
    /// Sinusoidal oscillation amplitude (radians)
    var oscAmplitude:    Float  = 0.38    // ≈ 22° — dramatic enough to create runup
    /// Sinusoidal oscillation frequency (Hz)
    /// Natural sloshing frequency for L=1.76m, d=0.37m:
    ///   f = (1/2π)·√(π·g/L · tanh(π·d/L)) ≈ 0.50 Hz  (resonance → max wave height)
    var oscFrequency:    Float  = 0.50
    /// Wall-clock time accumulator (seconds)
    private var simTime: Double = 0

    // MARK: - Init

    init(device: MTLDevice, view: MTKView) {
        self.device       = device
        self.commandQueue = device.makeCommandQueue()!
        self.sim          = SPHSimulator(device: device)

        let p = sim.params
        let centre = SIMD3<Float>(
            (p.boundMinX + p.boundMaxX) * 0.5,
            (p.boundMinY + p.boundMaxY) * 0.5,
            (p.boundMinZ + p.boundMaxZ) * 0.5
        )
        self.camera = OrbitCamera(target: centre)

        super.init()

        let lib = compile()
        setupParticlePipeline(lib, view)
        setupBoxPipeline(lib, view)
        setupDepthStates()
        cameraBuffer = device.makeBuffer(
            length:  MemoryLayout<CameraUniforms>.stride,
            options: .storageModeShared)!
        buildBoxVertices()
    }

    // MARK: - Pipeline setup

    private func compile() -> MTLLibrary {
        let opts = MTLCompileOptions()
        opts.languageVersion = .version3_0
        guard let lib = try? device.makeLibrary(source: metalShaderSource, options: opts) else {
            fatalError("[Renderer] Metal shader compilation failed")
        }
        return lib
    }

    private func setupParticlePipeline(_ lib: MTLLibrary, _ view: MTKView) {
        let d = MTLRenderPipelineDescriptor()
        d.vertexFunction   = lib.makeFunction(name: "particleVertex")
        d.fragmentFunction = lib.makeFunction(name: "particleFragment")
        d.depthAttachmentPixelFormat      = view.depthStencilPixelFormat
        d.colorAttachments[0].pixelFormat = view.colorPixelFormat
        let ca = d.colorAttachments[0]!
        ca.isBlendingEnabled           = true
        ca.rgbBlendOperation           = .add
        ca.alphaBlendOperation         = .add
        ca.sourceRGBBlendFactor        = .sourceAlpha
        ca.destinationRGBBlendFactor   = .oneMinusSourceAlpha
        ca.sourceAlphaBlendFactor      = .one
        ca.destinationAlphaBlendFactor = .oneMinusSourceAlpha
        particlePSO = try! device.makeRenderPipelineState(descriptor: d)
    }

    private func setupBoxPipeline(_ lib: MTLLibrary, _ view: MTKView) {
        let d = MTLRenderPipelineDescriptor()
        d.vertexFunction   = lib.makeFunction(name: "boxVertex")
        d.fragmentFunction = lib.makeFunction(name: "boxFragment")
        d.depthAttachmentPixelFormat      = view.depthStencilPixelFormat
        d.colorAttachments[0].pixelFormat = view.colorPixelFormat
        let ca = d.colorAttachments[0]!
        ca.isBlendingEnabled           = true
        ca.rgbBlendOperation           = .add
        ca.alphaBlendOperation         = .add
        ca.sourceRGBBlendFactor        = .sourceAlpha
        ca.destinationRGBBlendFactor   = .oneMinusSourceAlpha
        ca.sourceAlphaBlendFactor      = .one
        ca.destinationAlphaBlendFactor = .oneMinusSourceAlpha
        boxPSO = try! device.makeRenderPipelineState(descriptor: d)
    }

    private func setupDepthStates() {
        let full = MTLDepthStencilDescriptor()
        full.depthCompareFunction = .less
        full.isDepthWriteEnabled  = true
        depthFull = device.makeDepthStencilState(descriptor: full)!

        let test = MTLDepthStencilDescriptor()
        test.depthCompareFunction = .less
        test.isDepthWriteEnabled  = false
        depthTest = device.makeDepthStencilState(descriptor: test)!
    }

    // MARK: - Bounding box geometry

    private func buildBoxVertices() {
        let p  = sim.params
        let lo = SIMD3<Float>(p.boundMinX, p.boundMinY, p.boundMinZ)
        let hi = SIMD3<Float>(p.boundMaxX, p.boundMaxY, p.boundMaxZ)

        // 8 corners of the AABB in box-local space
        let c: [SIMD3<Float>] = [
            SIMD3(lo.x, lo.y, lo.z), SIMD3(hi.x, lo.y, lo.z),
            SIMD3(hi.x, hi.y, lo.z), SIMD3(lo.x, hi.y, lo.z),
            SIMD3(lo.x, lo.y, hi.z), SIMD3(hi.x, lo.y, hi.z),
            SIMD3(hi.x, hi.y, hi.z), SIMD3(lo.x, hi.y, hi.z)
        ]
        let edges: [(Int,Int)] = [
            (0,1),(1,2),(2,3),(3,0),   // bottom
            (4,5),(5,6),(6,7),(7,4),   // top
            (0,4),(1,5),(2,6),(3,7)    // pillars
        ]
        var verts: [SIMD4<Float>] = []
        for (a, b) in edges {
            verts.append(SIMD4<Float>(c[a], 1))
            verts.append(SIMD4<Float>(c[b], 1))
        }
        boxVertBuf = device.makeBuffer(
            bytes:   verts,
            length:  verts.count * MemoryLayout<SIMD4<Float>>.stride,
            options: .storageModeShared)!
    }

    // MARK: - MTKViewDelegate

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        camera.aspectRatio = Float(size.width / max(size.height, 1))
    }

    func draw(in view: MTKView) {
        // ── 1. Update tilt ─────────────────────────────────────────────────
        let dt = Float(1.0 / Double(max(view.preferredFramesPerSecond, 1)))
        simTime += Double(dt)

        if autoOscillate {
            // Smooth sinusoidal rocking
            tiltAngle = oscAmplitude * sin(Float(2 * Double.pi) * oscFrequency * Float(simTime))
            // Allow impulse perturbations on top of the base oscillation
            tiltAngle    += tiltVelocity * dt
            tiltVelocity *= 0.88          // quickly damp the perturbation
        } else {
            // Free tilt with friction (spring towards zero)
            tiltVelocity -= tiltAngle * 1.2 * dt    // weak restoring force
            tiltVelocity *= 0.96                     // air friction
            tiltAngle    += tiltVelocity * dt
            tiltAngle     = tiltAngle.clamped(to: -Float.pi/2 ... Float.pi/2)
        }

        // ── 2. Push gravity into simulation ────────────────────────────────
        let g: Float = 9.81
        sim.updateGravity(
            x: -g * sin(tiltAngle),
            y: -g * cos(tiltAngle),
            z:  0
        )

        // ── 3. Run simulation sub-steps ────────────────────────────────────
        for _ in 0..<stepsPerFrame { sim.step() }

        // ── 4. Build model matrix (rotate around box centre) ───────────────
        let p      = sim.params
        let centre = SIMD3<Float>(
            (p.boundMinX + p.boundMaxX) * 0.5,
            (p.boundMinY + p.boundMaxY) * 0.5,
            (p.boundMinZ + p.boundMaxZ) * 0.5
        )
        let model  = simd_float4x4(translation:  centre)
                   * simd_float4x4(rotationZ: tiltAngle)
                   * simd_float4x4(translation: -centre)

        camera.uniforms(buffer: cameraBuffer, model: model)

        // ── 5. Render ──────────────────────────────────────────────────────
        guard
            let pass     = view.currentRenderPassDescriptor,
            let cmdBuf   = commandQueue.makeCommandBuffer(),
            let drawable = view.currentDrawable
        else { return }

        pass.colorAttachments[0].clearColor  = MTLClearColor(red: 0.03, green: 0.03, blue: 0.06, alpha: 1)
        pass.colorAttachments[0].loadAction  = .clear
        pass.colorAttachments[0].storeAction = .store
        pass.depthAttachment.loadAction      = .clear
        pass.depthAttachment.clearDepth      = 1.0

        guard let enc = cmdBuf.makeRenderCommandEncoder(descriptor: pass) else { return }

        // Bounding box (drawn first, depth-write on so particles behind it sort correctly)
        enc.setRenderPipelineState(boxPSO)
        enc.setDepthStencilState(depthFull)
        enc.setVertexBuffer(boxVertBuf,  offset: 0, index: 0)
        enc.setVertexBuffer(cameraBuffer, offset: 0, index: 1)
        enc.drawPrimitives(type: .line, vertexStart: 0, vertexCount: 24)

        // Particles
        enc.setRenderPipelineState(particlePSO)
        enc.setDepthStencilState(depthTest)
        enc.setVertexBuffer(sim.particleBuffer, offset: 0, index: 0)
        enc.setVertexBuffer(cameraBuffer,        offset: 0, index: 1)
        var r = particleRadius
        enc.setVertexBytes(&r, length: MemoryLayout<Float>.stride, index: 2)
        enc.drawPrimitives(type: .triangleStrip, vertexStart: 0,
                           vertexCount: 4, instanceCount: sim.numParticles)

        enc.endEncoding()
        cmdBuf.present(drawable)
        cmdBuf.commit()
    }

    // MARK: - Input

    func handleMouseDrag(dx: Float, dy: Float) { camera.orbit(dx: dx, dy: dy) }
    func handleScroll(delta: Float)             { camera.zoom(delta: delta)    }
    func resetSimulation()                      { sim.resetParticles()         }

    /// Apply a tilt impulse (angular velocity kick in rad/s).
    func applyTiltImpulse(_ impulse: Float)     { tiltVelocity += impulse     }

    func increaseAmplitude() {
        oscAmplitude = min(oscAmplitude + 0.05, Float.pi / 2.5)
        print("[slosh] amplitude = \(Int(oscAmplitude * 180 / .pi))°")
    }
    func decreaseAmplitude() {
        oscAmplitude = max(oscAmplitude - 0.05, 0.05)
        print("[slosh] amplitude = \(Int(oscAmplitude * 180 / .pi))°")
    }
    func increaseFrequency() {
        oscFrequency = min(oscFrequency + 0.05, 1.5)
        print("[slosh] frequency = \(String(format: "%.2f", oscFrequency)) Hz")
    }
    func decreaseFrequency() {
        oscFrequency = max(oscFrequency - 0.05, 0.05)
        print("[slosh] frequency = \(String(format: "%.2f", oscFrequency)) Hz")
    }
}
