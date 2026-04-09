// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "LivingTales",
    platforms: [.iOS(.v17)],
    targets: [
        .executableTarget(
            name: "LivingTales",
            path: ".",
            exclude: ["Package.swift", "Tests"],
            sources: ["App", "Engine", "Models", "UI", "Cartridge"],
            resources: [.copy("Cartridge/Bundles/TheAmberCipher.cartridge")]
        )
    ]
)
