import Foundation

let executable = URL(fileURLWithPath: CommandLine.arguments[0])
let repository = executable
    .deletingLastPathComponent()  // MacOS
    .deletingLastPathComponent()  // Contents
    .deletingLastPathComponent()  // Forge LLM.app
    .deletingLastPathComponent()  // repository
    .path

let process = Process()
process.executableURL = URL(fileURLWithPath: "\(repository)/.venv/bin/python")
process.arguments = ["\(repository)/tools/forge_chat.py"]
var environment = ProcessInfo.processInfo.environment
environment["PYTHONPATH"] = "\(repository)/python"
process.environment = environment

do {
    try process.run()
    process.waitUntilExit()
    exit(process.terminationStatus)
} catch {
    fputs("Unable to start Forge LLM: \(error)\n", stderr)
    exit(1)
}
