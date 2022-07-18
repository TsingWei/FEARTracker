/*
See LICENSE folder for this sample’s licensing information.

Abstract:
Contains the object recognition view controller for the Breakfast Finder.
*/

import UIKit
import AVFoundation
import Vision

class FEARTrackingViewController: ViewController {

    enum Constants {
        static let rect = CGRect(x: 0.25, y: 0.25, width: 0.5, height: 0.5)
        static let imageWidth: CGFloat = 480
        static let imageHeight: CGFloat = 640
    }

    /// button to start processing
    lazy var startButton: UIButton = {
        let button = UIButton()
        button.backgroundColor = .red
        button.translatesAutoresizingMaskIntoConstraints = false
        return button
    }()

    /// red rectangle used to enter object location
    lazy var rectangle: UIView = {
        let width = Constants.imageWidth * previewLayer.bounds.height / Constants.imageHeight
        let rectFrame = Constants.rect.applying(
            CGAffineTransform(translationX: -(width - previewLayer.bounds.width) / 2, y: 0)
                .scaledBy(x: width, y: previewView.frame.height)
        )
        let rect = UIView(frame: rectFrame)
        rect.layer.borderWidth = 8
        rect.layer.borderColor = UIColor.red.cgColor
        return rect
    }()

    /// container layer that has all the renderings of the observations
    lazy var detectionOverlay: CALayer! = {
        let detectionOverlay = CALayer()
        detectionOverlay.name = "DetectionOverlay"
        return detectionOverlay
    }()

    private var processingEnabled: Bool = false
    private let tracker = VOTracker()

    override func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard processingEnabled, let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return
        }

        do {
            if !tracker.isInitialized {
                let width = CVPixelBufferGetWidth(pixelBuffer)
                let height = CVPixelBufferGetHeight(pixelBuffer)
                let rect = Constants.rect.applying(
                    CGAffineTransform(scaleX: CGFloat(width), y: CGFloat(height))
                )
                try tracker.initialize(image: pixelBuffer, rect: rect)
            } else {
                let detection = try tracker.track(image: pixelBuffer)
                drawVisionRequestResults(detections: [detection])
            }
        } catch {
            print(error)
            DispatchQueue.main.async { [weak self] in
                self?.drawVisionRequestResults(detections: [])
                self?.trackingFailed()
            }
        }
    }

    override func setupAVCapture() {
        super.setupAVCapture()

        updateLayerGeometry()

        // start the capture
        startCaptureSession()
    }

    override func viewDidLoad() {
        super.viewDidLoad()

        detectionOverlay.bounds = CGRect(x: 0.0, y: 0.0, width: bufferSize.width, height: bufferSize.height)
        detectionOverlay.position = CGPoint(x: rootLayer.bounds.midX, y: rootLayer.bounds.midY)
        rootLayer.addSublayer(detectionOverlay)

        previewView.addSubview(startButton)

        startButton.centerXAnchor.constraint(equalTo: previewView.centerXAnchor).isActive = true
        startButton.bottomAnchor.constraint(equalTo: previewView.bottomAnchor, constant: -25).isActive = true
        startButton.heightAnchor.constraint(equalToConstant: 50).isActive = true
        startButton.widthAnchor.constraint(equalToConstant: 50).isActive = true
        startButton.addTarget(self, action: #selector(nextButtonPressed), for: .touchUpInside)
        previewView.addSubview(rectangle)
    }

    @objc func nextButtonPressed() {
        startButton.isHidden = true
        rectangle.isHidden = true
        processingEnabled = true
    }

    func trackingFailed() {
        startButton.isHidden = false
        rectangle.isHidden = false
        processingEnabled = false
        tracker.clear()
    }
}

// MARK: - visualizations
private extension FEARTrackingViewController {

    func drawVisionRequestResults(detections: [VOTrackerResult]) {
        CATransaction.begin()
        CATransaction.setValue(kCFBooleanTrue, forKey: kCATransactionDisableActions)
        detectionOverlay.sublayers = nil // remove all the old recognized objects

        for detection in detections {
            let shapeLayer = self.createRoundedRectLayerWithBounds(detection.bbox)
            let textLayer = self.createTextSubLayerInBounds(detection.bbox, confidence: detection.confidence)
            shapeLayer.addSublayer(textLayer)
            detectionOverlay.addSublayer(shapeLayer)
        }
        self.updateLayerGeometry()
        CATransaction.commit()
    }
    
    func updateLayerGeometry() {
        let bounds = rootLayer.bounds
        var scale: CGFloat
        
        let xScale: CGFloat = bounds.size.width / bufferSize.width
        let yScale: CGFloat = bounds.size.height / bufferSize.height

        scale = fmax(xScale, yScale)
        if scale.isInfinite {
            scale = 1.0
        }
        CATransaction.begin()
        CATransaction.setValue(kCFBooleanTrue, forKey: kCATransactionDisableActions)

        detectionOverlay.setAffineTransform(.identity.scaledBy(x: scale, y: scale))
        // center the layer
        detectionOverlay.position = CGPoint(x: bounds.midX, y: bounds.midY)
        
        CATransaction.commit()
        
    }
    
    func createTextSubLayerInBounds(_ bounds: CGRect, confidence: VNConfidence) -> CATextLayer {
        let textLayer = CATextLayer()
        textLayer.name = "Object Label"
        let formattedString = NSMutableAttributedString(string: String(format: "Confidence:  %.2f", confidence))
        textLayer.string = formattedString
        textLayer.bounds = CGRect(x: 0, y: 0, width: bounds.size.height - 10, height: bounds.size.width - 10)
        textLayer.position = CGPoint(x: bounds.midX, y: bounds.midY)
        textLayer.shadowOpacity = 0.7
        textLayer.shadowOffset = CGSize(width: 2, height: 2)
        textLayer.foregroundColor = CGColor(colorSpace: CGColorSpaceCreateDeviceRGB(), components: [0.0, 0.0, 0.0, 1.0])
        textLayer.contentsScale = 2.0 // retina rendering
        return textLayer
    }
    
    func createRoundedRectLayerWithBounds(_ bounds: CGRect) -> CALayer {
        let shapeLayer = CALayer()
        shapeLayer.bounds = bounds
        shapeLayer.position = CGPoint(x: bounds.midX, y: bounds.midY)
        shapeLayer.name = "Found Object"
        shapeLayer.backgroundColor = CGColor(colorSpace: CGColorSpaceCreateDeviceRGB(), components: [1.0, 1.0, 0.2, 0.4])
        shapeLayer.cornerRadius = 7
        return shapeLayer
    }
    
}
