// main.swift
// Entry point — AppKit window + MTKView + input forwarding.
//
// ┌─────────────────────────────────────────────────────────────┐
// │  CONTROLS                                                   │
// │  Mouse drag      Orbit camera                               │
// │  Scroll          Zoom                                       │
// │  ↑  /  ↓         Motion amplitude  ±5°                      │
// │  R               Reset particles                            │
// │  Q / Esc         Quit                                       │
// └─────────────────────────────────────────────────────────────┘

import AppKit
import MetalKit

// ─────────────────────────────────────────────────────────────────────────────
// FluidView — handles all mouse and keyboard events
// ─────────────────────────────────────────────────────────────────────────────

final class FluidView: MTKView {

    var renderer: Renderer!

    override var acceptsFirstResponder: Bool { true }

    // ── Mouse orbit ─────────────────────────────────────────────────────────

    private var lastDrag: CGPoint?

    override func mouseDown(with e: NSEvent) {
        lastDrag = convert(e.locationInWindow, from: nil)
    }
    override func mouseDragged(with e: NSEvent) {
        let loc = convert(e.locationInWindow, from: nil)
        if let l = lastDrag {
            renderer.handleMouseDrag(dx: Float(loc.x - l.x), dy: Float(loc.y - l.y))
        }
        lastDrag = loc
    }
    override func mouseUp(with e: NSEvent) { lastDrag = nil }

    // ── Scroll zoom ─────────────────────────────────────────────────────────

    override func scrollWheel(with e: NSEvent) {
        renderer.handleScroll(delta: Float(e.scrollingDeltaY))
    }

    // ── Keyboard ────────────────────────────────────────────────────────────

    override func keyDown(with e: NSEvent) {
        switch e.keyCode {
        case 126: renderer.increaseAmplitude()   // ↑
        case 125: renderer.decreaseAmplitude()   // ↓
        case 15:                                 // R — reset
            renderer.resetSimulation()
            print("[slosh] particles reset")
        case 12, 53:                             // Q / Esc — quit
            NSApplication.shared.terminate(nil)
        default:
            super.keyDown(with: e)
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Bootstrap
// ─────────────────────────────────────────────────────────────────────────────

let app = NSApplication.shared
app.setActivationPolicy(.regular)

guard let device = MTLCreateSystemDefaultDevice() else {
    fatalError("Metal is not available on this machine.")
}

print("""
╔══════════════════════════════════════════════════════╗
║    SPH Fluid Simulation — Sloshing Tank (Metal GPU)  ║
╠══════════════════════════════════════════════════════╣
║  Mouse drag   : orbit camera                         ║
║  Scroll       : zoom                                 ║
║  ↑ / ↓        : motion amplitude  ±5°                ║
║  R            : reset particles                      ║
║  Q / Esc      : quit                                 ║
╚══════════════════════════════════════════════════════╝
GPU : \(device.name)
""")

// ── Window ──────────────────────────────────────────────────────────────────

let windowRect = NSRect(x: 0, y: 0, width: 1200, height: 760)
let window = NSWindow(
    contentRect: windowRect,
    styleMask:   [.titled, .closable, .miniaturizable, .resizable],
    backing:     .buffered,
    defer:       false
)
window.title = "SPH Sloshing Tank — Metal GPU"
window.center()
window.makeKeyAndOrderFront(nil)

// ── MTKView ─────────────────────────────────────────────────────────────────

let view = FluidView(frame: windowRect, device: device)
view.colorPixelFormat         = .bgra8Unorm
view.depthStencilPixelFormat  = .depth32Float
view.clearColor               = MTLClearColor(red: 0.03, green: 0.03, blue: 0.06, alpha: 1)
view.preferredFramesPerSecond = 60
view.isPaused                 = false
view.enableSetNeedsDisplay    = false
window.contentView = view

// ── Renderer ────────────────────────────────────────────────────────────────

let renderer = Renderer(device: device, view: view)
view.renderer = renderer
view.delegate = renderer
view.becomeFirstResponder()

// ── Run ─────────────────────────────────────────────────────────────────────

app.activate(ignoringOtherApps: true)
app.run()
