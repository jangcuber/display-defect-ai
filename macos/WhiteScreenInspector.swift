import AppKit
import Combine
import Foundation
import SwiftUI
import UniformTypeIdentifiers

struct Mark: Decodable, Identifiable, Sendable {
    var id: String { "\(kind)-\(box)" }
    static let kinds = ["dark_dot", "color_dot", "horizontal_line", "vertical_line", "dark_region"]
    let kind: String
    let score: Double
    let box: [Double]
    let polygon: [[Double]]
    let crop: String
    let crop_box: [Int]
    let detail: String
    var title: String {
        switch kind {
        case "dark_dot": "어두운 점"
        case "color_dot": "색 이상점"
        case "horizontal_line": "가로선"
        case "vertical_line": "세로선"
        case "dark_region": "밝기·색 영역 저하"
        default: kind
        }
    }
    var defaultZoom: Int { kind == "dark_region" ? 1 : kind.hasSuffix("_line") ? 2 : 4 }
    var color: Color {
        switch kind {
        case "dark_dot": .teal
        case "color_dot": .pink
        case "horizontal_line": .orange
        case "vertical_line": .blue
        default: .purple
        }
    }
}

struct InspectionPayload: Decodable, Sendable {
    let decision: String
    let message: String
    let model_name: String
    let width: Int
    let height: Int
    let detections: [Mark]
    let preview: String
    let mask_preview: String?
    let seconds: Double
    let note: String
}
struct WorkerEvent: Decodable, Sendable {
    let event: String
    let id: String
    let done: Int?
    let total: Int?
    let message: String?
    let result: InspectionPayload?
}

final class InspectionWorker: @unchecked Sendable {
    private var process: Process?
    private var input: FileHandle?
    private var log: FileHandle?

    func send(project: URL, photo: URL, output: URL, id: String,
              receive: @escaping @Sendable (WorkerEvent) -> Void) throws {
        if process?.isRunning != true {
            let bundled = Bundle.main.resourceURL?.appendingPathComponent("InspectorWorker/inspector-worker")
            let executable: URL
            let arguments: [String]
            if let bundled, FileManager.default.isExecutableFile(atPath: bundled.path) {
                executable = bundled
                arguments = ["--server", "--resource-root", project.path]
            } else {
                executable = project.appendingPathComponent("artifacts/yolo-env/bin/python")
                arguments = [project.appendingPathComponent("scripts/inspect_native_photo.py").path, "--server"]
            }
            guard FileManager.default.isExecutableFile(atPath: executable.path) else {
                throw NSError(domain: "검사 환경을 찾을 수 없습니다.", code: 1)
            }
            let p = Process(), stdin = Pipe(), stdout = Pipe()
            p.executableURL = executable
            p.arguments = arguments
            p.currentDirectoryURL = project
            var env = ProcessInfo.processInfo.environment
            env["YOLO_CONFIG_DIR"] = project.appendingPathComponent("artifacts/ultralytics").path
            env["MPLCONFIGDIR"] = project.appendingPathComponent("artifacts/matplotlib").path
            env["YOLO_AUTOINSTALL"] = "false"
            env["OMP_NUM_THREADS"] = "4"
            p.environment = env
            p.standardInput = stdin
            p.standardOutput = stdout
            let logURL = FileManager.default.temporaryDirectory.appendingPathComponent("native-inspector-worker.log")
            FileManager.default.createFile(atPath: logURL.path, contents: nil)
            log = try FileHandle(forWritingTo: logURL)
            p.standardError = log
            try p.run()
            process = p
            input = stdin.fileHandleForWriting
            let reader = stdout.fileHandleForReading
            DispatchQueue.global(qos: .userInitiated).async {
                var buffer = Data()
                while true {
                    let bytes = reader.availableData
                    if bytes.isEmpty { break }
                    buffer.append(bytes)
                    while let newline = buffer.firstIndex(of: 10) {
                        let line = buffer[..<newline]
                        buffer.removeSubrange(...newline)
                        if let value = try? JSONDecoder().decode(WorkerEvent.self, from: line) {
                            receive(value)
                        }
                    }
                }
                receive(WorkerEvent(event: "closed", id: "", done: nil, total: nil, message: nil, result: nil))
            }
        }
        let request = ["id": id, "photo": photo.path, "output": output.path]
        var bytes = try JSONSerialization.data(withJSONObject: request)
        bytes.append(10)
        try input?.write(contentsOf: bytes)
    }
    var isRunning: Bool { process?.isRunning == true }
    func stop() {
        process?.terminate()
        try? input?.close()
        try? log?.close()
        process = nil; input = nil; log = nil
    }
    deinit { stop() }
}

@MainActor
final class InspectorState: ObservableObject {
    @Published var preview: NSImage?
    @Published var mask: NSImage?
    @Published var result: InspectionPayload?
    @Published var error: String?
    @Published var name = "사진을 선택해 주세요."
    @Published var busy = false
    @Published var progress = ""
    @Published var selected = 0
    @Published var boxes = false
    @Published var masks = false
    @Published var zoom = 4
    private var identifier = ""
    private var work: URL?
    private var launched = false
    private var worker = InspectionWorker()
    private var workerGeneration = UUID().uuidString
    private var project: URL {
        if let bundled = Bundle.main.resourceURL?.appendingPathComponent("InspectorData"),
           FileManager.default.fileExists(atPath: bundled.path) {
            return bundled
        }
        return Bundle.main.bundleURL.deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    }
    func openLaunchPhoto() {
        guard !launched else { return }; launched = true
        if let path = CommandLine.arguments.dropFirst().first, !path.hasPrefix("-") { start(URL(fileURLWithPath: path)) }
    }
    func start(_ photo: URL) {
        guard !busy else { return }
        if let old = work { try? FileManager.default.removeItem(at: old) }
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent("oled-native-\(UUID().uuidString)")
        work = folder
        identifier = UUID().uuidString
        name = photo.lastPathComponent; result = nil; error = nil; preview = nil; mask = nil
        boxes = false; masks = false; selected = 0; busy = true; progress = "모델을 준비하고 있습니다…"
        do {
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            let current = identifier
            let generation = workerGeneration
            try worker.send(project: project, photo: photo, output: folder, id: current) { [weak self] event in
                Task { @MainActor in self?.receive(event, workerID: generation) }
            }
        } catch { self.error = error.localizedDescription; busy = false }
    }
    private func receive(_ event: WorkerEvent, workerID: String) {
        // A resident worker serves later requests too; event IDs, rather than the initial callback ID, identify jobs.
        if event.event == "closed" {
            if busy && workerID == workerGeneration && !worker.isRunning { error = "검사 프로세스가 종료됐습니다. 사진을 다시 선택해 주세요."; busy = false }
            return
        }
        guard event.id == identifier else { return }
        switch event.event {
        case "loading": progress = "모델을 불러오고 있습니다…"
        case "progress": progress = "원본 해상도 검사 \(event.done ?? 0) / \(event.total ?? 0) 타일"
        case "error": error = event.message; busy = false
        case "result":
            if let value = event.result {
                result = value
                zoom = value.detections.first?.defaultZoom ?? 4
                preview = NSImage(contentsOfFile: value.preview)
                mask = value.mask_preview.flatMap { NSImage(contentsOfFile: $0) }
            }
            busy = false
        default: break
        }
    }
    func openExample(_ kind: String) {
        result = nil; preview = nil; mask = nil; error = nil; name = "예시 사진"
        let folder = project.appendingPathComponent("data/oled-white-paired-v2")
        do {
            let text = try String(contentsOf: folder.appendingPathComponent("manifest.jsonl"), encoding: .utf8)
            for line in text.split(separator: "\n") {
                guard let bytes = String(line).data(using: .utf8),
                      let row = try JSONSerialization.jsonObject(with: bytes) as? [String: Any],
                      row["split"] as? String == "test", row["kind"] as? String == kind,
                      let path = row["image"] as? String else { continue }
                start(folder.appendingPathComponent(path)); return
            }
            error = "해당 예시 사진을 찾을 수 없습니다."
        } catch { self.error = "예시 목록을 열 수 없습니다. 프로젝트의 데이터 폴더를 확인해 주세요." }
    }
    func cancel() {
        identifier = UUID().uuidString
        workerGeneration = UUID().uuidString
        worker.stop(); worker = InspectionWorker()
        busy = false; progress = ""; error = "검사를 취소했습니다."
    }
    func shutdown() {
        worker.stop()
        if let work { try? FileManager.default.removeItem(at: work) }
    }
    var selectedMark: Mark? {
        guard let list = result?.detections, list.indices.contains(selected) else { return nil }
        return list[selected]
    }
}

@main
struct WhiteScreenInspector: App {
    @StateObject private var state = InspectorState()
    var body: some Scene {
        WindowGroup("흰 화면 검사") { InspectorView(state: state) }
            .defaultSize(width: 1260, height: 860)
            .windowResizability(.contentMinSize)
    }
}

struct InspectorView: View {
    @ObservedObject var state: InspectorState
    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text("OLED 흰 화면 검사").font(.largeTitle.weight(.semibold))
                    Text(state.name).foregroundStyle(.secondary).lineLimit(1)
                }
                Spacer()
                if state.busy { Button("취소") { state.cancel() } }
                Menu("예시 사진") {
                    Button("정상") { state.openExample("normal") }
                    Button("어두운 점") { state.openExample("dark_dot") }
                    Button("색 이상점") { state.openExample("color_dot") }
                    Button("가로선") { state.openExample("horizontal_line") }
                    Button("세로선") { state.openExample("vertical_line") }
                    Button("밝기·색 영역 저하") { state.openExample("dark_region") }
                    Button("복합 불량") { state.openExample("mixed") }
                }.disabled(state.busy).fixedSize()
                Button("사진 열기…") {
                    let picker = NSOpenPanel(); picker.allowedContentTypes = [.png, .jpeg]
                    picker.canChooseDirectories = false; picker.allowsMultipleSelection = false
                    if picker.runModal() == .OK, let photo = picker.url { state.start(photo) }
                }.disabled(state.busy).keyboardShortcut("o")
            }
            if state.busy {
                HStack { ProgressView().controlSize(.small); Text(state.progress) }.frame(height: 36)
            } else if let result = state.result {
                HStack {
                    Text(decision(result.decision)).font(.title2.bold())
                        .foregroundStyle(result.decision == "normal" ? .green : result.decision == "defect" ? .red : .orange)
                    Text("\(result.detections.count)개 · \(result.model_name)")
                    Spacer()
                    Text(String(format: "%d × %d px · %.2f초", result.width, result.height, result.seconds)).foregroundStyle(.secondary)
                }.frame(height: 36)
            } else if let error = state.error {
                HStack {
                    Text("검사 불가").font(.title2.bold()).foregroundStyle(.orange)
                    Text(error).font(.callout).foregroundStyle(.secondary)
                }.frame(height: 36)
            } else { Text("원본 해상도로 점·선·영역을 검사합니다.").foregroundStyle(.secondary).frame(height: 36) }
            if let result = state.result, !result.detections.isEmpty {
                HStack(spacing: 16) {
                    ForEach(Mark.kinds, id: \.self) { kind in
                        if let mark = result.detections.first(where: { $0.kind == kind }) {
                            HStack(spacing: 5) {
                                Circle().fill(mark.color).frame(width: 7, height: 7)
                                Text("\(mark.title) \(result.detections.filter { $0.kind == kind }.count)")
                            }
                        }
                    }
                }.font(.callout)
            }
            HStack {
                Text("원본 사진").font(.headline)
                Spacer()
                Toggle("검출 위치", isOn: $state.boxes).toggleStyle(.checkbox)
                Toggle("픽셀 출력", isOn: $state.masks).toggleStyle(.checkbox).disabled(state.mask == nil)
            }
            HSplitView {
                PhotoPane(image: state.masks ? state.mask : state.preview,
                          marks: state.boxes ? state.result?.detections ?? [] : [], selected: state.selectedMark?.id)
                    .frame(minWidth: 520, maxWidth: .infinity, maxHeight: .infinity)
                VStack(alignment: .leading, spacing: 12) {
                    Text("검출 목록").font(.headline).fixedSize(horizontal: false, vertical: true)
                    ScrollView {
                        VStack(alignment: .leading, spacing: 7) {
                            ForEach(Array((state.result?.detections ?? []).enumerated()), id: \.element.id) { index, mark in
                                Button { state.selected = index; state.zoom = mark.defaultZoom } label: {
                                    HStack(alignment: .top) {
                                        Circle().fill(mark.color).frame(width: 8, height: 8).padding(.top, 5)
                                        VStack(alignment: .leading, spacing: 3) {
                                            Text(mark.title).fontWeight(.medium)
                                            Text(mark.detail).font(.caption).foregroundStyle(.secondary)
                                        }
                                        Spacer()
                                    }.padding(8).background(state.selected == index ? Color.accentColor.opacity(0.12) : .clear).cornerRadius(6)
                                }.buttonStyle(.plain)
                            }
                            if state.result?.detections.isEmpty == true { Text("검출된 불량 없음").foregroundStyle(.secondary).padding(8) }
                        }
                    }.frame(minHeight: 60, maxHeight: 120)
                    if let mark = state.selectedMark, let image = NSImage(contentsOfFile: mark.crop) {
                        HStack {
                            Text("원본 세부 확대").font(.headline)
                            Spacer()
                            Picker("배율", selection: $state.zoom) { Text("1×").tag(1); Text("2×").tag(2); Text("4×").tag(4); Text("8×").tag(8) }.labelsHidden().frame(width: 80)
                        }
                        ScrollView([.horizontal, .vertical]) {
                            Image(nsImage: image).resizable().interpolation(.none)
                                .frame(width: image.size.width * CGFloat(state.zoom), height: image.size.height * CGFloat(state.zoom))
                        }.defaultScrollAnchor(.center).frame(minHeight: 100, maxHeight: .infinity).background(Color(nsColor: .underPageBackgroundColor))
                        Text("표시 없는 원본 부분 · 긴 선/큰 영역은 일부분\n좌표 (\(mark.crop_box[0]), \(mark.crop_box[1]))부터").font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                    } else { Spacer() }
                }.padding(.leading, 14).frame(minWidth: 300, idealWidth: 350, maxWidth: 440)
            }
            if let result = state.result {
                Text(result.message).font(.callout)
                Text(result.note + (result.mask_preview == nil ? "" : " · 픽셀 출력은 임계값을 넘은 모델 응답입니다.")).font(.caption).foregroundStyle(.secondary)
            }
        }.padding(22).frame(minWidth: 940, minHeight: 690).task { state.openLaunchPhoto() }
            .onReceive(NotificationCenter.default.publisher(for: NSApplication.willTerminateNotification)) { _ in state.shutdown() }
    }
    private func decision(_ value: String) -> String {
        switch value { case "normal": "정상"; case "defect": "불량"; case "review": "확인 필요"; default: "검사 불가" }
    }
}

struct PhotoPane: View {
    let image: NSImage?
    let marks: [Mark]
    let selected: String?
    var body: some View {
        GeometryReader { geometry in
            if let image {
                let scale = min(geometry.size.width / image.size.width, geometry.size.height / image.size.height)
                let size = CGSize(width: image.size.width * scale, height: image.size.height * scale)
                ZStack(alignment: .topLeading) {
                    Image(nsImage: image).resizable().interpolation(.high).frame(width: size.width, height: size.height)
                    Canvas { context, _ in
                        for mark in marks where mark.polygon.count == 4 {
                            var path = Path()
                            for (index, point) in mark.polygon.enumerated() {
                                let p = CGPoint(x: point[0] * size.width, y: point[1] * size.height)
                                if index == 0 { path.move(to: p) } else { path.addLine(to: p) }
                            }
                            path.closeSubpath()
                            context.stroke(path, with: .color(mark.color), lineWidth: mark.id == selected ? 1.2 : 0.7)
                        }
                    }.frame(width: size.width, height: size.height)
                }.frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                ContentUnavailableView("검사할 사진", systemImage: "photo", description: Text("PNG 또는 JPEG 사진을 열어 주세요."))
            }
        }.background(Color(nsColor: .underPageBackgroundColor)).clipShape(RoundedRectangle(cornerRadius: 8))
    }
}
