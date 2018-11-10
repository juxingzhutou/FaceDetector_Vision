import Foundation
import UIKit
import Vision

class BoundingBox {
  let shapeLayer: CAShapeLayer

  init() {
    shapeLayer = CAShapeLayer()
    shapeLayer.fillColor = UIColor.clear.cgColor
    shapeLayer.lineWidth = 4
    shapeLayer.isHidden = true
    
    shapeLayer.fillColor = nil // No fill to show boxed object
    shapeLayer.shadowOpacity = 0
    shapeLayer.shadowRadius = 0
    shapeLayer.borderWidth = 2
    
    // Vary the line color according to input.
    shapeLayer.borderColor = UIColor.clear.cgColor
    
    // Locate the layer.
    shapeLayer.anchorPoint = .zero
    shapeLayer.masksToBounds = true
    
    // Transform the layer to have same coordinate system as the imageView underneath it.
    shapeLayer.transform = CATransform3DMakeScale(1, -1, 1)
  }

  func addToLayer(_ parent: CALayer) {
    parent.addSublayer(shapeLayer)
  }
  
  fileprivate func boundingBox(forRegionOfInterest: CGRect, withinImageBounds bounds: CGRect) -> CGRect {
    
    let imageWidth = bounds.width
    let imageHeight = bounds.height
    
    // Begin with input rect.
    var rect = forRegionOfInterest
    
    // Reposition origin.
    rect.origin.x *= imageWidth
    rect.origin.x += bounds.origin.x
    rect.origin.y = (1 - rect.origin.y) * imageHeight + bounds.origin.y
    
    // Rescale normalized coordinates.
    rect.size.width *= imageWidth
    rect.size.height *= imageHeight
    
    return rect
  }
    
  func draw(face: VNFaceObservation, onImageWithBounds bounds: CGRect) {
    let faceBox = boundingBox(forRegionOfInterest: face.boundingBox, withinImageBounds: bounds)
    shapeLayer.frame = faceBox
    shapeLayer.isHidden = false
  }
  
  // Facial landmarks are GREEN.
  func drawFeatures(onFaces faceObservation: VNFaceObservation, onImageWithBounds bounds: CGRect) {
    
    let faceBounds = boundingBox(forRegionOfInterest: faceObservation.boundingBox, withinImageBounds: bounds)
    guard let landmarks = faceObservation.landmarks else {
      return
    }
    
    // Iterate through landmarks detected on the current face.
    let landmarkLayer = shapeLayer
    let landmarkPath = CGMutablePath()
    let affineTransform = CGAffineTransform(scaleX: faceBounds.size.width, y: faceBounds.size.height)
    
    // Treat eyebrows and lines as open-ended regions when drawing paths.
    let openLandmarkRegions: [VNFaceLandmarkRegion2D?] = [
      landmarks.leftEyebrow,
      landmarks.rightEyebrow,
      landmarks.faceContour,
      landmarks.noseCrest,
      landmarks.medianLine
    ]
    
    // Draw eyes, lips, and nose as closed regions.
    let closedLandmarkRegions = [
      landmarks.leftEye,
      landmarks.rightEye,
      landmarks.outerLips,
      landmarks.innerLips,
      landmarks.nose
      ].flatMap { $0 } // Filter out missing regions.
    
    // Draw paths for the open regions.
    for openLandmarkRegion in openLandmarkRegions where openLandmarkRegion != nil {
      landmarkPath.addPoints(in: openLandmarkRegion!,
                             applying: affineTransform,
                             closingWhenComplete: false)
    }
    
    // Draw paths for the closed regions.
    for closedLandmarkRegion in closedLandmarkRegions {
      landmarkPath.addPoints(in: closedLandmarkRegion,
                             applying: affineTransform,
                             closingWhenComplete: true)
    }
    
    // Format the path's appearance: color, thickness, shadow.
    landmarkLayer.path = landmarkPath
    landmarkLayer.lineWidth = 2
    landmarkLayer.strokeColor = UIColor.green.cgColor
    landmarkLayer.fillColor = nil
    landmarkLayer.shadowOpacity = 0.75
    landmarkLayer.shadowRadius = 4
    
    // Locate the path in the parent coordinate system.
    landmarkLayer.anchorPoint = .zero
    landmarkLayer.frame = faceBounds
    landmarkLayer.transform = CATransform3DMakeScale(1, -1, 1)
    
    shapeLayer.isHidden = false
  }

  func hide() {
    shapeLayer.isHidden = true
  }
  
}

private extension CGMutablePath {
  // Helper function to add lines to a path.
  func addPoints(in landmarkRegion: VNFaceLandmarkRegion2D,
                 applying affineTransform: CGAffineTransform,
                 closingWhenComplete closePath: Bool) {
    let pointCount = landmarkRegion.pointCount
    
    // Draw line if and only if path contains multiple points.
    guard pointCount > 1 else {
      return
    }
    self.addLines(between: landmarkRegion.normalizedPoints, transform: affineTransform)
    
    if closePath {
      self.closeSubpath()
    }
  }
}
