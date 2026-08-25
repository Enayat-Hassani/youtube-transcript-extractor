// Emits one TSV row per recognised region: x, y, width, height, confidence, text.
// Coordinates are normalised (0-1) with the origin bottom-left, as Vision reports them.
import Foundation
import Vision
import AppKit

guard CommandLine.arguments.count > 1,
      let image = NSImage(contentsOfFile: CommandLine.arguments[1]),
      let cg = image.cgImage(forProposedRect: nil, context: nil, hints: nil)
else { FileHandle.standardError.write("cannot read image\n".data(using: .utf8)!); exit(1) }

let request = VNRecognizeTextRequest { request, _ in
    guard let observations = request.results as? [VNRecognizedTextObservation] else { return }
    for o in observations {
        guard let best = o.topCandidates(1).first else { continue }
        let b = o.boundingBox
        let text = best.string.replacingOccurrences(of: "\t", with: " ")
        print(String(format: "%.5f\t%.5f\t%.5f\t%.5f\t%.3f\t%@",
                     b.minX, b.minY, b.width, b.height, best.confidence, text))
    }
}
request.recognitionLevel = .accurate
request.usesLanguageCorrection = false   // UI text and numbers, not prose
try? VNImageRequestHandler(cgImage: cg, options: [:]).perform([request])
