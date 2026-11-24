// Observe the packaged app from the window server without screen recording.
import Cocoa
import CoreGraphics
import Foundation

let app = Process()
app.executableURL = URL(fileURLWithPath: CommandLine.arguments[1])
app.arguments = ["gui"]
try app.run()

func windowAppeared() -> Bool {
    let deadline = Date().addingTimeInterval(30)
    while app.isRunning && Date() < deadline {
        let windows = CGWindowListCopyWindowInfo([.optionOnScreenOnly, .excludeDesktopElements], kCGNullWindowID) as? [[String: Any]] ?? []
        for window in windows {
            guard let pid = window[kCGWindowOwnerPID as String] as? Int32, pid == app.processIdentifier,
                  let layer = window[kCGWindowLayer as String] as? Int, layer == 0,
                  let bounds = window[kCGWindowBounds as String] as? [String: CGFloat],
                  let width = bounds["Width"], width > 200,
                  let height = bounds["Height"], height > 200 else { continue }
            return true
        }
        Thread.sleep(forTimeInterval: 0.2)
    }
    return false
}

let visible = windowAppeared()
if app.isRunning {
    app.terminate()
    app.waitUntilExit()
}
guard visible else {
    FileHandle.standardError.write(Data("Packaged GUI did not display a window.\n".utf8))
    exit(1)
}
print("Packaged GUI displayed a window.")
