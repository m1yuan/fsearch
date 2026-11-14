// swift-tools-version: 6.2
// Embeddings sidecar for fsearch. macOS 27: Core AI first, Core ML fallback.
// Both paths ask for the Neural Engine and do not enable the GPU.

import PackageDescription

let package = Package(
    name: "ane-embed",
    platforms: [
        .macOS("27.0"),
    ],
    products: [
        .executable(name: "ane-embed", targets: ["AneEmbed"]),
    ],
    dependencies: [
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.4.0"),
    ],
    targets: [
        .executableTarget(
            name: "AneEmbed",
            dependencies: [
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            linkerSettings: [
                .linkedFramework("CoreML"),
                .linkedFramework("CoreAI"),
            ]
        ),
    ]
)
