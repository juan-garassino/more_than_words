// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "Thornfield",
    platforms: [.iOS(.v17)],
    targets: [
        .executableTarget(
            name: "Thornfield",
            path: ".",
            exclude: ["Package.swift", "Tests"],
            sources: ["App", "Engine", "Models", "UI", "Cartridge"],
            resources: [.copy("Cartridge/Bundles/TheAmberCipher.cartridge")]
        )
    ]
)
