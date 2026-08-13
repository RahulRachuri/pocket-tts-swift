// swift-tools-version: 6.0
//
// pocket-tts-swift: the production Swift host for the pocket-tts Core AI port.
//
// Build (Xcode 27 toolchain, macOS 27 SDK):
//     swift build -c release
// Run from the repo root (so the default weights/ and artifacts/ paths resolve):
//     swift run -c release pocket-tts-cli --text "Hello there." --voice alba --out out.wav
//
// `CoreAI` is a system framework in the macOS 27 SDK — no package dependency, just
// a linker setting. No third-party dependencies at all: argument parsing, protobuf
// (for the sentencepiece model), and safetensors are hand-rolled in PocketTTSKit,
// which keeps the build offline-safe and the surface auditable.
import PackageDescription

let package = Package(
    name: "pocket-tts-swift",
    // iOS is a first-class consumer: the device bench harness depends on PocketTTSKit as
    // a local package. CoreAI ships in both the macOS 27 and iOS 27 SDKs (device-only on
    // iOS — the SDK has no simulator slice, so the library links only for real devices).
    platforms: [.macOS("27.0"), .iOS("27.0")],
    products: [
        .library(name: "PocketTTSKit", targets: ["PocketTTSKit"]),
        .executable(name: "pocket-tts-cli", targets: ["pocket-tts-cli"]),
    ],
    targets: [
        .target(
            name: "PocketTTSKit",
            linkerSettings: [.linkedFramework("CoreAI")]
        ),
        .executableTarget(
            name: "pocket-tts-cli",
            dependencies: ["PocketTTSKit"],
            linkerSettings: [.linkedFramework("CoreAI")]
        ),
    ]
)
