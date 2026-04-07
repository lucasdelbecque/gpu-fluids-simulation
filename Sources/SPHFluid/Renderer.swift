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

    // ── 3-axis smooth random motion ───────────────────────────────────────
    // Each axis is driven by a sum of sinusoids at mutually irrational
    // frequencies → the motion never repeats, always feels organic.
    //
    // Current Euler angles (box-local, applied as Rx·Ry·Rz):
    private var angleX: Float = 0   // pitch  (front/back tilt)
    private var angleY: Float = 0   // yaw    (slow twist around vertical)
    private var angleZ: Float = 0   // roll   (left/right tilt — primary sloshing axis)
    private var simTime: Double = 0

    // Amplitude envelope — user-controllable
    var oscAmplitude: Float = 0.35  // ≈ 20°  (applied to all axes, Y gets ×0.3)

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
        // ── 1. Smooth random 3-axis motion ─────────────────────────────────
        let dt = Float(1.0 / Double(max(view.preferredFramesPerSecond, 1)))
        simTime += Double(dt)
        let t = Float(simTime)

        // Sum of sinusoids at irrational frequency ratios → aperiodic, smooth.
        // Frequencies chosen so no two axes share a common period.
        //   Z (roll)  : dominant sloshing axis — two slow components
        //   X (pitch) : secondary — offset phases, slightly higher freqs
        //   Y (yaw)   : very subtle slow twist — feels like a boat in open water
        let A = oscAmplitude
        angleZ = A       * (0.55 * sin(0.41 * t) + 0.30 * sin(0.73 * t + 1.1)
                          + 0.15 * sin(1.17 * t + 2.3))
        angleX = A * 0.6 * (0.50 * sin(0.53 * t + 0.7) + 0.35 * sin(0.89 * t + 2.8)
                          + 0.15 * sin(1.31 * t + 0.4))
        angleY = A * 0.2 * (0.60 * sin(0.19 * t + 1.5) + 0.40 * sin(0.37 * t + 3.1))

        // ── 2. Gravity in box-local space ──────────────────────────────────
        // g_world = (0, -9.81, 0).
        // g_local = Rᵀ · g_world  where R = Rx(ax)·Ry(ay)·Rz(az)
        // We compute this analytically for the three Euler angles.
        let g: Float = 9.81
        let cx = cos(angleX), sx = sin(angleX)
        let cy = cos(angleY), sy = sin(angleY)
        let cz = cos(angleZ), sz = sin(angleZ)

        // Full rotation matrix R = Rx·Ry·Rz  (column-major, so R * v rotates v)
        // Transposing gives R⁻¹ to bring world-g into box frame.
        // R column 1 (world Y basis expressed in box space, i.e. second column of R):
        //   Ry_world_in_box = Rᵀ · (0,1,0)  = second row of R
        let gLocalX = g * ( sy * cx)        // second row, col 0 of Rx·Ry·Rz
        let gLocalY = g * (-sx)              // second row, col 1
        let gLocalZ = g * ( cy * cx * sz - sy * sx * cz + cx * cy) // approximated below

        // Cleaner: just rotate the gravity vector through the inverse rotation.
        // Build the 3×3 rotation R = Rx(ax)·Ry(ay)·Rz(az) and apply Rᵀ to g_world.
        // Row vectors of R (= columns of Rᵀ):
        let r00 = cy*cz;           let r01 = cy*sz;           let r02 = -sy
        let r10 = sx*sy*cz-cx*sz;  let r11 = sx*sy*sz+cx*cz;  let r12 = sx*cy
        let r20 = cx*sy*cz+sx*sz;  let r21 = cx*sy*sz-sx*cz;  let r22 = cx*cy
        // g_world = (0, -g, 0)  →  g_local = Rᵀ · g_world = -g * (col1 of R)
        let gx = -g * r01   // = -g * (row0·ĵ) = -g * r01
        let gy = -g * r11
        let gz = -g * r21
        _ = (gLocalX, gLocalY, gLocalZ, r00, r02, r10, r12, r20, r22) // silence warnings

        sim.updateGravity(x: gx, y: gy, z: gz)

        // ── 3. Run simulation sub-steps ────────────────────────────────────
        for _ in 0..<stepsPerFrame { sim.step() }

        // ── 4. Build model matrix — rotate around box centre ───────────────
        let p      = sim.params
        let centre = SIMD3<Float>(
            (p.boundMinX + p.boundMaxX) * 0.5,
            (p.boundMinY + p.boundMaxY) * 0.5,
            (p.boundMinZ + p.boundMaxZ) * 0.5
        )
        // Apply rotations in same order as the gravity calc: Rx · Ry · Rz
        let rot   = simd_float4x4(rotationX: angleX)
                  * simd_float4x4(rotationY: angleY)
                  * simd_float4x4(rotationZ: angleZ)
        let model = simd_float4x4(translation: centre)
                  * rot
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

    func increaseAmplitude() {
        oscAmplitude = min(oscAmplitude + 0.05, Float.pi / 2.5)
        print("[slosh] amplitude = \(Int(oscAmplitude * 180 / .pi))°")
    }
    func decreaseAmplitude() {
        oscAmplitude = max(oscAmplitude - 0.05, 0.03)
        print("[slosh] amplitude = \(Int(oscAmplitude * 180 / .pi))°")
    }
}
