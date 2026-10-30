// Instinct for macOS: a native window around the local agent server.
//
// The app starts `python -m instinct.harness.cli --app` from the checkout it
// was built from, hands it a fresh session token through the environment, and
// shows the UI in a WKWebView once the server reports its port. Quitting the
// app closes the server's stdin, which stops it and its model workers.

import Cocoa
import CryptoKit
import WebKit

// The page's background in each appearance (--page in app.css).
let background = NSColor(name: nil) { appearance in
    appearance.bestMatch(from: [.darkAqua, .aqua]) == .darkAqua
        ? NSColor(srgbRed: 0x13 / 255, green: 0x12 / 255, blue: 0x19 / 255, alpha: 1)
        : NSColor(srgbRed: 0xf8 / 255, green: 0xf7 / 255, blue: 0xfc / 255, alpha: 1)
}

// MARK: - Backend process

final class Backend {
    let root: URL
    private(set) var port: Int?
    private(set) var token = ""
    private var process: Process?
    private var stdin: Pipe?
    private var stopping = false
    private var buffer = Data()
    var onReady: ((Int) -> Void)?
    var onExit: ((String) -> Void)?

    static let logURL: URL = {
        let logs = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Logs/Instinct", isDirectory: true)
        try? FileManager.default.createDirectory(at: logs, withIntermediateDirectories: true)
        return logs.appendingPathComponent("backend.log")
    }()

    init(root: URL) { self.root = root }

    var python: URL { root.appendingPathComponent(".venv/bin/python") }
    var config: URL { root.appendingPathComponent("examples/instinct.json") }
    var modelsReady: Bool {
        FileManager.default.fileExists(atPath: root.appendingPathComponent("models/gemma-4-26b-a4b-mlx/config.json").path)
    }

    func start(workspace: URL, demo: Bool = false, state: URL? = nil) throws {
        guard FileManager.default.isExecutableFile(atPath: python.path) else {
            throw AppError("No Python environment at \(python.path). Run `macos/build.sh` from the repository after installing it.")
        }
        stopping = false
        port = nil
        token = Self.randomToken()
        // Keys and network settings belong to the app, not to a workspace; a
        // test's state folder keeps its own.
        let shared = state ?? Self.supportDirectory
        let keys = shared.appendingPathComponent("api-keys.json")
        let network = shared.appendingPathComponent("network.json")
        let state = state ?? Self.stateDirectory(for: workspace)
        try FileManager.default.createDirectory(at: state, withIntermediateDirectories: true)

        let process = Process()
        process.executableURL = python
        process.currentDirectoryURL = root
        var arguments = ["-m", "instinct.harness.cli", "--app", "--port", "0",
                         "--workspace", workspace.path, "--state-dir", state.path,
                         "--keys-file", keys.path, "--network-file", network.path]
        arguments += modelsReady && !demo ? ["--config", config.path] : ["--demo"]
        process.arguments = arguments
        var environment = ProcessInfo.processInfo.environment
        environment["INSTINCT_APP_TOKEN"] = token
        environment["PYTHONPATH"] = root.appendingPathComponent("python").path
        environment["PYTHONUNBUFFERED"] = "1"
        process.environment = environment

        let input = Pipe(), output = Pipe()
        FileManager.default.createFile(atPath: Self.logURL.path, contents: nil)
        let log = try FileHandle(forWritingTo: Self.logURL)
        process.standardInput = input
        process.standardOutput = output
        process.standardError = log
        output.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            DispatchQueue.main.async { self?.consume(data) }
        }
        process.terminationHandler = { [weak self] finished in
            DispatchQueue.main.async {
                guard let self, !self.stopping else { return }
                self.onExit?("The Instinct server stopped (exit \(finished.terminationStatus)).\n\n" + Self.logTail())
            }
        }
        try process.run()
        self.process = process
        self.stdin = input
    }

    private func consume(_ data: Data) {
        buffer.append(data)
        while let newline = buffer.firstIndex(of: 0x0A) {
            let line = buffer.prefix(upTo: newline)
            buffer.removeSubrange(...newline)
            guard port == nil,
                  let message = try? JSONSerialization.jsonObject(with: line) as? [String: Any],
                  message["event"] as? String == "ready",
                  let value = message["port"] as? Int else { continue }
            port = value
            onReady?(value)
        }
    }

    func stop() {
        guard let process else { return }
        stopping = true
        // Closing stdin asks the server to shut down cleanly; SIGTERM backs it up.
        try? stdin?.fileHandleForWriting.close()
        if process.isRunning { process.terminate() }
        let deadline = Date().addingTimeInterval(8)
        while process.isRunning && Date() < deadline { usleep(50_000) }
        if process.isRunning { kill(process.processIdentifier, SIGKILL) }
        self.process = nil
    }

    static var supportDirectory: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Instinct", isDirectory: true)
    }

    static func stateDirectory(for workspace: URL) -> URL {
        let digest = SHA256.hash(data: Data(workspace.standardizedFileURL.path.utf8))
        let id = digest.prefix(6).map { String(format: "%02x", $0) }.joined()
        return supportDirectory.appendingPathComponent("workspaces/\(workspace.lastPathComponent)-\(id)", isDirectory: true)
    }

    static func randomToken() -> String {
        var bytes = [UInt8](repeating: 0, count: 32)
        _ = SecRandomCopyBytes(kSecRandomDefault, bytes.count, &bytes)
        return Data(bytes).base64EncodedString()
            .replacingOccurrences(of: "+", with: "-").replacingOccurrences(of: "/", with: "_")
            .replacingOccurrences(of: "=", with: "")
    }

    static func logTail() -> String {
        guard let text = try? String(contentsOf: logURL, encoding: .utf8) else { return "" }
        return text.split(separator: "\n").suffix(12).joined(separator: "\n")
    }
}

struct AppError: LocalizedError {
    let message: String
    init(_ message: String) { self.message = message }
    var errorDescription: String? { message }
}

// MARK: - Startup screen

final class StartupView: NSView {
    private let status = NSTextField(labelWithString: "")
    private let spinner = NSProgressIndicator()
    private let buttons = NSStackView()
    var onRetry: (() -> Void)?

    override init(frame: NSRect) {
        super.init(frame: frame)
        let title = NSTextField(labelWithString: "Instinct")
        title.font = .systemFont(ofSize: 26, weight: .semibold)
        title.textColor = .labelColor
        status.font = .systemFont(ofSize: 13)
        status.textColor = .secondaryLabelColor
        status.alignment = .center
        status.maximumNumberOfLines = 14
        status.preferredMaxLayoutWidth = 520
        spinner.style = .spinning
        spinner.controlSize = .small
        let retry = NSButton(title: "Try Again", target: self, action: #selector(retry))
        let log = NSButton(title: "Open Log", target: self, action: #selector(openLog))
        buttons.addArrangedSubview(retry)
        buttons.addArrangedSubview(log)
        buttons.isHidden = true
        let stack = NSStackView(views: [title, spinner, status, buttons])
        stack.orientation = .vertical
        stack.spacing = 14
        stack.translatesAutoresizingMaskIntoConstraints = false
        addSubview(stack)
        NSLayoutConstraint.activate([
            stack.centerXAnchor.constraint(equalTo: centerXAnchor),
            stack.centerYAnchor.constraint(equalTo: centerYAnchor),
            stack.widthAnchor.constraint(lessThanOrEqualToConstant: 560),
        ])
    }

    required init?(coder: NSCoder) { fatalError() }

    override func draw(_ dirtyRect: NSRect) {
        background.setFill()  // Resolved per appearance on every draw.
        dirtyRect.fill()
    }

    func show(_ message: String, failed: Bool = false) {
        status.stringValue = message
        buttons.isHidden = !failed
        spinner.isHidden = failed
        failed ? spinner.stopAnimation(nil) : spinner.startAnimation(nil)
    }

    @objc private func retry() { onRetry?() }
    @objc private func openLog() { NSWorkspace.shared.open(Backend.logURL) }
}

// MARK: - UI tests

/// Set INSTINCT_TEST_SCRIPT to run a JavaScript test inside the app's own web
/// view: the app opens INSTINCT_TEST_WORKSPACE (demo mode unless
/// INSTINCT_TEST_MODELS=1), runs the script as an async function body once the
/// page loads, writes its result to INSTINCT_TEST_OUTPUT/result.json and
/// quits. The script can call `await snapshot(name)` to save the window,
/// title bar included, as a PNG. The window stays transparent and ignores
/// clicks, so a test run doesn't get in the way. macos/test.sh drives this.
struct UITest {
    let script: String
    let output: URL
    let workspace: URL
    let models: Bool

    static func fromEnvironment() -> UITest? {
        let environment = ProcessInfo.processInfo.environment
        guard let path = environment["INSTINCT_TEST_SCRIPT"],
              let script = try? String(contentsOfFile: path, encoding: .utf8) else { return nil }
        let output = URL(fileURLWithPath: environment["INSTINCT_TEST_OUTPUT"] ?? NSTemporaryDirectory())
        let workspace = URL(fileURLWithPath: environment["INSTINCT_TEST_WORKSPACE"] ?? NSTemporaryDirectory())
        try? FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        return UITest(script: script, output: output, workspace: workspace, models: environment["INSTINCT_TEST_MODELS"] == "1")
    }

    // Helpers every test script can use.
    static let prelude = """
        const sleep = (ms) => new Promise((done) => setTimeout(done, ms));
        const until = async (test, ms = 15000) => {
          for (const end = Date.now() + ms; Date.now() < end; await sleep(100)) { const value = await test(); if (value) return value; }
          throw new Error("timed out: " + test.toString().slice(0, 120));
        };
        const snapshot = (name) => window.webkit.messageHandlers.instinctTest.postMessage({ type: "snapshot", name });
        const results = [];
        const check = async (name, body) => {
          try { await body(); results.push({ name, ok: true }); }
          catch (error) { results.push({ name, ok: false, error: String(error && error.message || error) }); }
        };
        """
}

// MARK: - App

final class AppDelegate: NSObject, NSApplicationDelegate, NSWindowDelegate, WKNavigationDelegate,
    WKUIDelegate, WKScriptMessageHandler, WKScriptMessageHandlerWithReply, WKDownloadDelegate
{
    var window: NSWindow!
    var webView: WKWebView!
    let startup = StartupView()
    var backend: Backend!
    var workspace: URL!
    let test = UITest.fromEnvironment()
    var testStarted = false

    func applicationDidFinishLaunching(_ notification: Notification) {
        guard let root = Bundle.main.object(forInfoDictionaryKey: "InstinctRoot") as? String else {
            fatalError("Info.plist has no InstinctRoot; build the app with macos/build.sh")
        }
        backend = Backend(root: URL(fileURLWithPath: root))
        workspace = test?.workspace ?? Self.savedWorkspace()

        let configuration = WKWebViewConfiguration()
        configuration.userContentController.add(self, name: "instinct")
        if test != nil {
            configuration.userContentController.addScriptMessageHandler(self, contentWorld: .page, name: "instinctTest")
        }
        configuration.applicationNameForUserAgent = "InstinctApp/0.1"
        configuration.preferences.setValue(true, forKey: "developerExtrasEnabled")
        webView = WKWebView(frame: .zero, configuration: configuration)
        webView.navigationDelegate = self
        webView.uiDelegate = self
        webView.underPageBackgroundColor = background
        webView.setValue(false, forKey: "drawsBackground")
        webView.isHidden = true

        window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 1320, height: 860),
                          styleMask: [.titled, .closable, .miniaturizable, .resizable, .fullSizeContentView],
                          backing: .buffered, defer: false)
        window.title = "Instinct"
        window.titleVisibility = .hidden
        window.titlebarAppearsTransparent = true
        // An empty unified toolbar gives a 52-point title bar with the window
        // buttons centred in it, matching the page's header.
        window.toolbar = NSToolbar(identifier: "main")
        window.toolbarStyle = .unified
        window.titlebarSeparatorStyle = .none
        window.backgroundColor = .clear
        window.isOpaque = false
        window.minSize = NSSize(width: 760, height: 520)
        window.delegate = self
        let content = NSView()
        // macOS's sidebar material, showing through the page's translucent sidebar.
        let vibrancy = NSVisualEffectView()
        vibrancy.material = .sidebar
        vibrancy.blendingMode = .behindWindow
        vibrancy.state = .followsWindowActiveState
        for view in [vibrancy, startup, webView!] as [NSView] {
            view.translatesAutoresizingMaskIntoConstraints = false
            content.addSubview(view)
            NSLayoutConstraint.activate([
                view.leadingAnchor.constraint(equalTo: content.leadingAnchor),
                view.trailingAnchor.constraint(equalTo: content.trailingAnchor),
                view.topAnchor.constraint(equalTo: content.topAnchor),
                view.bottomAnchor.constraint(equalTo: content.bottomAnchor),
            ])
        }
        window.contentView = content
        window.center()
        window.setFrameAutosaveName("Main")
        if test != nil {
            window.alphaValue = 0
            window.ignoresMouseEvents = true
            window.orderFrontRegardless()
        } else {
            window.makeKeyAndOrderFront(nil)
        }

        startup.onRetry = { [weak self] in self?.launch() }
        backend.onReady = { [weak self] port in self?.load(port: port) }
        backend.onExit = { [weak self] message in
            self?.webView.isHidden = true
            self?.startup.isHidden = false
            self?.startup.show(message, failed: true)
        }
        NSApp.mainMenu = makeMenu()
        launch()
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }
    func applicationWillTerminate(_ notification: Notification) { backend.stop() }

    // MARK: Workspace

    static func savedWorkspace() -> URL {
        if let path = UserDefaults.standard.string(forKey: "workspace"),
           FileManager.default.fileExists(atPath: path) {
            return URL(fileURLWithPath: path, isDirectory: true)
        }
        // A folder of its own, so the agent never starts out in your files.
        let home = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Instinct", isDirectory: true)
        try? FileManager.default.createDirectory(at: home, withIntermediateDirectories: true)
        return home
    }

    func launch() {
        backend.stop()
        webView.isHidden = true
        startup.isHidden = false
        window.subtitle = workspace.path
        startup.show(backend.modelsReady ? "Starting in \(workspace.lastPathComponent)…" : "Models not found; starting in demo mode…")
        do {
            try backend.start(workspace: workspace, demo: test.map { !$0.models } ?? false,
                              state: test?.output.appendingPathComponent("state"))
        } catch {
            startup.show(error.localizedDescription, failed: true)
        }
    }

    func load(port: Int) {
        startup.show("Loading…")
        let url = URL(string: "http://127.0.0.1:\(port)/?app=mac#token=\(backend.token)")!
        webView.load(URLRequest(url: url))
    }

    @objc func openFolder(_ sender: Any?) {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.canCreateDirectories = true
        panel.prompt = "Open"
        panel.message = "Instinct can read, create and change files inside the folder you choose."
        panel.directoryURL = workspace
        panel.beginSheetModal(for: window) { [weak self] response in
            guard response == .OK, let url = panel.url, let self else { return }
            self.workspace = url
            UserDefaults.standard.set(url.path, forKey: "workspace")
            self.launch()
        }
    }

    @objc func revealWorkspace(_ sender: Any?) {
        NSWorkspace.shared.activateFileViewerSelecting([workspace])
    }

    // MARK: Menu

    func run(_ command: String) {
        webView.evaluateJavaScript("window.instinct && window.instinct.command(\(Self.jsString(command)))")
    }

    @objc func menuCommand(_ sender: NSMenuItem) {
        if let command = sender.representedObject as? String { run(command) }
    }

    @objc func reload(_ sender: Any?) { webView.reload() }

    func makeMenu() -> NSMenu {
        let main = NSMenu()
        func submenu(_ title: String, _ items: [NSMenuItem]) -> NSMenu {
            let menu = NSMenu(title: title)
            items.forEach(menu.addItem)
            let holder = NSMenuItem()
            holder.submenu = menu
            main.addItem(holder)
            return menu
        }
        func item(_ title: String, _ action: Selector?, _ key: String = "", _ modifiers: NSEvent.ModifierFlags = .command, target: AnyObject? = nil) -> NSMenuItem {
            let entry = NSMenuItem(title: title, action: action, keyEquivalent: key)
            entry.keyEquivalentModifierMask = modifiers
            entry.target = target
            return entry
        }
        func command(_ title: String, _ name: String, _ key: String = "", _ modifiers: NSEvent.ModifierFlags = .command) -> NSMenuItem {
            let entry = item(title, #selector(menuCommand(_:)), key, modifiers, target: self)
            entry.representedObject = name
            return entry
        }
        _ = submenu("Instinct", [
            item("About Instinct", #selector(NSApplication.orderFrontStandardAboutPanel(_:))),
            .separator(),
            command("Settings…", "settings", ","),
            .separator(),
            item("Hide Instinct", #selector(NSApplication.hide(_:)), "h"),
            item("Hide Others", #selector(NSApplication.hideOtherApplications(_:)), "h", [.command, .option]),
            item("Show All", #selector(NSApplication.unhideAllApplications(_:))),
            .separator(),
            item("Quit Instinct", #selector(NSApplication.terminate(_:)), "q"),
        ])
        _ = submenu("File", [
            command("New Chat", "newChat", "n"),
            item("Open Folder…", #selector(openFolder(_:)), "o", target: self),
            item("Show Workspace in Finder", #selector(revealWorkspace(_:)), "r", [.command, .shift], target: self),
            .separator(),
            item("Close Window", #selector(NSWindow.performClose(_:)), "w"),
        ])
        _ = submenu("Edit", [
            item("Undo", Selector(("undo:")), "z"),
            item("Redo", Selector(("redo:")), "z", [.command, .shift]),
            .separator(),
            item("Cut", #selector(NSText.cut(_:)), "x"),
            item("Copy", #selector(NSText.copy(_:)), "c"),
            item("Paste", #selector(NSText.paste(_:)), "v"),
            item("Select All", #selector(NSText.selectAll(_:)), "a"),
            .separator(),
            command("Search Chats", "searchChats", "f", [.command, .shift]),
        ])
        _ = submenu("View", [
            command("Toggle Sidebar", "toggleSidebar", "s", [.command, .control]),
            command("Toggle Workspace Panel", "toggleWorkspace", "d", [.command, .shift]),
            command("Toggle Terminal", "toggleTerminal", "`", [.control]),
            .separator(),
            item("Reload", #selector(reload(_:)), "r", target: self),
            item("Enter Full Screen", #selector(NSWindow.toggleFullScreen(_:)), "f", [.command, .control]),
        ])
        let windows = submenu("Window", [
            item("Minimize", #selector(NSWindow.performMiniaturize(_:)), "m"),
            item("Zoom", #selector(NSWindow.performZoom(_:))),
            .separator(),
            item("Bring All to Front", #selector(NSApplication.arrangeInFront(_:))),
        ])
        NSApp.windowsMenu = windows
        return main
    }

    static func jsString(_ value: String) -> String {
        let data = try! JSONSerialization.data(withJSONObject: [value])
        return String(data: data, encoding: .utf8)!.dropFirst().dropLast().description
    }

    // MARK: Messages from the page

    func userContentController(_ controller: WKUserContentController, didReceive message: WKScriptMessage,
                               replyHandler: @escaping (Any?, String?) -> Void) {
        guard let test, let body = message.body as? [String: Any] else { return replyHandler(nil, "unknown test message") }
        if body["type"] as? String == "hit", let x = body["x"] as? Double, let y = body["y"] as? Double,
           let frame = window.contentView?.superview {
            let view = frame.hitTest(NSPoint(x: x, y: frame.bounds.height - y))
            return replyHandler(view.map { String(describing: type(of: $0)) } ?? "none", nil)
        }
        guard body["type"] as? String == "snapshot", let name = body["name"] as? String else {
            return replyHandler(nil, "unknown test message")
        }
        snapshotWindow { data in
            let file = test.output.appendingPathComponent("\(name).png")
            if let data, (try? data.write(to: file)) != nil { replyHandler(file.path, nil) }
            else { replyHandler(nil, "snapshot failed") }
        }
    }

    func userContentController(_ controller: WKUserContentController, didReceive message: WKScriptMessage) {
        guard let body = message.body as? [String: Any], let type = body["type"] as? String else { return }
        switch type {
        case "drag":
            if let event = NSApp.currentEvent, event.type == .leftMouseDown { window.performDrag(with: event) }
        case "zoom":
            window.performZoom(nil)
        case "openFolder":
            openFolder(nil)
        case "reveal":
            let path = (body["path"] as? String).map { workspace.appendingPathComponent($0) } ?? workspace!
            NSWorkspace.shared.activateFileViewerSelecting([path])
        case "openExternal":
            if let link = body["url"] as? String, let url = URL(string: link),
               ["http", "https", "mailto"].contains(url.scheme ?? "") {
                NSWorkspace.shared.open(url)
            }
        default:
            break
        }
    }

    // MARK: Navigation, downloads and panels

    func isLocal(_ url: URL?) -> Bool {
        guard let url else { return false }
        if ["about", "blob", "data"].contains(url.scheme ?? "") { return true }
        return url.host == "127.0.0.1" && url.port == backend.port
    }

    func webView(_ webView: WKWebView, decidePolicyFor action: WKNavigationAction,
                 decisionHandler: @escaping (WKNavigationActionPolicy) -> Void) {
        if action.shouldPerformDownload { return decisionHandler(.download) }
        if isLocal(action.request.url) { return decisionHandler(.allow) }
        if let url = action.request.url { NSWorkspace.shared.open(url) }
        decisionHandler(.cancel)
    }

    func webView(_ webView: WKWebView, decidePolicyFor response: WKNavigationResponse,
                 decisionHandler: @escaping (WKNavigationResponsePolicy) -> Void) {
        decisionHandler(response.canShowMIMEType ? .allow : .download)
    }

    func webView(_ webView: WKWebView, navigationAction: WKNavigationAction, didBecome download: WKDownload) {
        download.delegate = self
    }

    func webView(_ webView: WKWebView, navigationResponse: WKNavigationResponse, didBecome download: WKDownload) {
        download.delegate = self
    }

    func download(_ download: WKDownload, decideDestinationUsing response: URLResponse,
                  suggestedFilename: String, completionHandler: @escaping (URL?) -> Void) {
        let folder = FileManager.default.urls(for: .downloadsDirectory, in: .userDomainMask)[0]
        var target = folder.appendingPathComponent(suggestedFilename)
        let stem = target.deletingPathExtension().lastPathComponent, ext = target.pathExtension
        var copy = 1
        while FileManager.default.fileExists(atPath: target.path) {
            copy += 1
            target = folder.appendingPathComponent(ext.isEmpty ? "\(stem) \(copy)" : "\(stem) \(copy).\(ext)")
        }
        completionHandler(target)
    }

    func downloadDidFinish(_ download: WKDownload) {
        if let url = download.progress.fileURL { NSWorkspace.shared.activateFileViewerSelecting([url]) }
    }

    func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
        webView.isHidden = false
        startup.isHidden = true  // Otherwise it would cover the sidebar material.
        window.makeFirstResponder(webView)
        if let test, !testStarted {
            testStarted = true
            runTest(test)
        }
    }

    func runTest(_ test: UITest) {
        let body = UITest.prelude + test.script + "\nreturn JSON.stringify(results);"
        webView.callAsyncJavaScript(body, arguments: [:], in: nil, in: .page) { [weak self] outcome in
            let report: String
            switch outcome {
            case .success(let value): report = value as? String ?? "[]"
            case .failure(let error):
                report = "[{\"name\": \"script\", \"ok\": false, \"error\": \(Self.jsString(error.localizedDescription))}]"
            }
            try? report.write(to: test.output.appendingPathComponent("result.json"), atomically: true, encoding: .utf8)
            self?.backend.stop()
            exit(report.contains("\"ok\":false") ? 1 : 0)
        }
    }

    /// The page as WebKit draws it, with the window's title bar buttons on top.
    func snapshotWindow(_ completion: @escaping (Data?) -> Void) {
        webView.takeSnapshot(with: nil) { [weak self] page, _ in
            guard let self, let page else { return completion(nil) }
            let size = self.window.frame.size
            let image = NSImage(size: size)
            image.lockFocus()
            background.setFill()
            NSRect(origin: .zero, size: size).fill()
            page.draw(in: self.webView.frame)
            if let close = self.window.standardWindowButton(.closeButton),
               let bar = close.superview?.superview,
               let rep = bar.bitmapImageRepForCachingDisplay(in: bar.bounds) {
                bar.cacheDisplay(in: bar.bounds, to: rep)
                let buttons = NSImage(size: bar.bounds.size)
                buttons.addRepresentation(rep)
                buttons.draw(in: bar.frame)
            }
            image.unlockFocus()
            let png = image.tiffRepresentation
                .flatMap(NSBitmapImageRep.init(data:))?
                .representation(using: .png, properties: [:])
            completion(png)
        }
    }

    func webViewWebContentProcessDidTerminate(_ webView: WKWebView) { webView.reload() }

    func webView(_ webView: WKWebView, createWebViewWith configuration: WKWebViewConfiguration,
                 for action: WKNavigationAction, windowFeatures: WKWindowFeatures) -> WKWebView? {
        if let url = action.request.url { NSWorkspace.shared.open(url) }
        return nil
    }

    func webView(_ webView: WKWebView, runOpenPanelWith parameters: WKOpenPanelParameters,
                 initiatedByFrame frame: WKFrameInfo, completionHandler: @escaping ([URL]?) -> Void) {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = parameters.allowsMultipleSelection
        panel.canChooseDirectories = parameters.allowsDirectories
        panel.beginSheetModal(for: window) { completionHandler($0 == .OK ? panel.urls : nil) }
    }

    func webView(_ webView: WKWebView, runJavaScriptAlertPanelWithMessage message: String,
                 initiatedByFrame frame: WKFrameInfo, completionHandler: @escaping () -> Void) {
        let alert = NSAlert()
        alert.messageText = message
        alert.beginSheetModal(for: window) { _ in completionHandler() }
    }

    func webView(_ webView: WKWebView, runJavaScriptConfirmPanelWithMessage message: String,
                 initiatedByFrame frame: WKFrameInfo, completionHandler: @escaping (Bool) -> Void) {
        let alert = NSAlert()
        alert.messageText = message
        alert.addButton(withTitle: "OK")
        alert.addButton(withTitle: "Cancel")
        alert.beginSheetModal(for: window) { completionHandler($0 == .alertFirstButtonReturn) }
    }
}

let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate
if delegate.test == nil {
    app.setActivationPolicy(.regular)
    app.activate(ignoringOtherApps: true)
} else {
    app.setActivationPolicy(.accessory)  // No Dock icon or focus change during tests.
}
app.run()
