// Camera.swift
// Orbit camera controlled with mouse drag and scroll wheel.

import Foundation
import simd
import Metal

final class OrbitCamera {

    var azimuth:     Float = -Float.pi / 5
    var elevation:   Float =  Float.pi / 8
    var distance:    Float = 3.0
    var target:      SIMD3<Float>
    var fovY:        Float = 55 * Float.pi / 180
    var aspectRatio: Float = 1.0
    let near:        Float = 0.01
    let far:         Float = 30.0

    private let minEl: Float = -Float.pi / 2 + 0.05
    private let maxEl: Float =  Float.pi / 2 - 0.05

    init(target: SIMD3<Float>) { self.target = target }

    func orbit(dx: Float, dy: Float) {
        azimuth  += dx * 0.006
        elevation = (elevation - dy * 0.006).clamped(to: minEl...maxEl)
    }

    func zoom(delta: Float) {
        distance = (distance - delta * 0.12).clamped(to: 0.5...12.0)
    }

    var position: SIMD3<Float> {
        let ce = cos(elevation)
        return target + distance * SIMD3(ce * cos(azimuth), sin(elevation), ce * sin(azimuth))
    }

    func uniforms(buffer: MTLBuffer, model: simd_float4x4) {
        let pos     = position
        let forward = normalize(target - pos)
        let right   = normalize(cross(forward, SIMD3<Float>(0, 1, 0)))
        let up      = cross(right, forward)

        var u = CameraUniforms(
            viewProj:    perspective() * lookAt(eye: pos, center: target, up: up),
            model:       model,
            cameraRight: SIMD4<Float>(right, 0),
            cameraUp:    SIMD4<Float>(up,    0),
            cameraPos:   SIMD4<Float>(pos,   0)
        )
        memcpy(buffer.contents(), &u, MemoryLayout<CameraUniforms>.stride)
    }

    // MARK: - Math

    private func lookAt(eye: SIMD3<Float>, center: SIMD3<Float>, up: SIMD3<Float>) -> simd_float4x4 {
        let f = normalize(center - eye)
        let r = normalize(cross(f, up))
        let u = cross(r, f)
        return simd_float4x4(columns: (
            SIMD4<Float>( r.x,  u.x, -f.x, 0),
            SIMD4<Float>( r.y,  u.y, -f.y, 0),
            SIMD4<Float>( r.z,  u.z, -f.z, 0),
            SIMD4<Float>(-dot(r, eye), -dot(u, eye), dot(f, eye), 1)
        ))
    }

    private func perspective() -> simd_float4x4 {
        let y = 1 / tan(fovY * 0.5)
        let x = y / aspectRatio
        let z = far / (near - far)
        return simd_float4x4(columns: (
            SIMD4<Float>(x,  0,       0,  0),
            SIMD4<Float>(0,  y,       0,  0),
            SIMD4<Float>(0,  0,       z, -1),
            SIMD4<Float>(0,  0, z * near,  0)
        ))
    }
}

// MARK: - simd_float4x4 convenience

extension simd_float4x4 {
    init(rotationX a: Float) {
        let c = cos(a), s = sin(a)
        self = simd_float4x4(columns: (
            SIMD4<Float>(1,  0, 0, 0),
            SIMD4<Float>(0,  c, s, 0),
            SIMD4<Float>(0, -s, c, 0),
            SIMD4<Float>(0,  0, 0, 1)
        ))
    }
    init(rotationY a: Float) {
        let c = cos(a), s = sin(a)
        self = simd_float4x4(columns: (
            SIMD4<Float>(c, 0, -s, 0),
            SIMD4<Float>(0, 1,  0, 0),
            SIMD4<Float>(s, 0,  c, 0),
            SIMD4<Float>(0, 0,  0, 1)
        ))
    }
    init(rotationZ a: Float) {
        let c = cos(a), s = sin(a)
        self = simd_float4x4(columns: (
            SIMD4<Float>( c, s, 0, 0),
            SIMD4<Float>(-s, c, 0, 0),
            SIMD4<Float>( 0, 0, 1, 0),
            SIMD4<Float>( 0, 0, 0, 1)
        ))
    }
    init(translation t: SIMD3<Float>) {
        self = simd_float4x4(columns: (
            SIMD4<Float>(1, 0, 0, 0),
            SIMD4<Float>(0, 1, 0, 0),
            SIMD4<Float>(0, 0, 1, 0),
            SIMD4<Float>(t.x, t.y, t.z, 1)
        ))
    }
}

// MARK: - Comparable clamp helper

extension Comparable {
    func clamped(to range: ClosedRange<Self>) -> Self {
        min(max(self, range.lowerBound), range.upperBound)
    }
}
