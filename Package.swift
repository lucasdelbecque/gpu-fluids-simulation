// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "SPHFluid",
    platforms: [.macOS(.v13)],
    targets: [
        .executableTarget(
            name: "SPHFluid",
            path: "Sources/SPHFluid"
        )
    ]
)
