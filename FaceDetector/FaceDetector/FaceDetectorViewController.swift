//
//  FaceDetectorViewController.swift
//
//  Created by Riceberg on 2018-11-08.
//  Copyright © 2018 Riceberg. All rights reserved.
//

import UIKit
import CoreML
import Vision
import AVFoundation
import Accelerate

class FaceDetectorViewController: UIViewController, AVCaptureVideoDataOutputSampleBufferDelegate {
  @IBOutlet weak var cameraView: UIView!
  @IBOutlet weak var frameLabel: UILabel!
  
  let semaphore = DispatchSemaphore(value: 1)
  let semaphore2 = DispatchSemaphore(value: 1)
  
  var lastExecution = Date()
  var screenHeight: Double?
  var screenWidth: Double?
  
  
  private lazy var cameraLayer: AVCaptureVideoPreviewLayer = AVCaptureVideoPreviewLayer(session: self.captureSession)
  private lazy var captureSession: AVCaptureSession = {
    let session = AVCaptureSession()
    session.sessionPreset = AVCaptureSession.Preset.hd1280x720
    
    guard
      let backCamera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
      let input = try? AVCaptureDeviceInput(device: backCamera)
      else { return session }
    session.addInput(input)
    return session
  }()
  
  let numBoxes = 100
  var boundingBoxes: [BoundingBox] = []
  let multiClass = true
  
  override func viewDidLoad() {
    super.viewDidLoad()
    self.cameraView?.layer.addSublayer(self.cameraLayer)
    self.cameraView?.bringSubview(toFront: self.frameLabel)
    self.frameLabel.textAlignment = .left
    let videoOutput = AVCaptureVideoDataOutput()
    videoOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "MyQueue"))
    self.captureSession.addOutput(videoOutput)
    self.captureSession.startRunning()
    
    setupBoxes()
    
    screenWidth = Double(view.frame.width)
    screenHeight = Double(view.frame.height)
    
  }
  
  override func viewDidAppear(_ animated: Bool) {
    super.viewDidAppear(animated)
    cameraLayer.frame = cameraView.layer.bounds
  }
  
  func setupBoxes() {
    // Create shape layers for the bounding boxes.
    for _ in 0..<numBoxes {
      let box = BoundingBox()
      box.addToLayer(view.layer)
      self.boundingBoxes.append(box)
    }
  }
  
  override func viewDidLayoutSubviews() {
    super.viewDidLayoutSubviews()
    self.cameraLayer.frame = self.cameraView?.bounds ?? .zero
  }
  
  override func didReceiveMemoryWarning() {
    super.didReceiveMemoryWarning()
    // Dispose of any resources that can be recreated.
  }
  
  func presentAlert(_ title: String, error: NSError) {
    // Always present alert on main thread.
    DispatchQueue.main.async {
      let alertController = UIAlertController(title: title,
                                              message: error.localizedDescription,
                                              preferredStyle: .alert)
      let okAction = UIAlertAction(title: "OK",
                                   style: .default) { _ in
                                    // Do nothing -- simply dismiss alert.
      }
      alertController.addAction(okAction)
      self.present(alertController, animated: true, completion: nil)
    }
  }
  
  
  lazy var faceDetectionRequest = VNDetectFaceRectanglesRequest(completionHandler: self.handleDetectedFaces)
  
  lazy var faceLandmarkRequest = VNDetectFaceLandmarksRequest(completionHandler: self.handleDetectedFaceLandmarks)
  
  fileprivate func handleDetectedFaces(request: VNRequest?, error: Error?) {
    if let nsError = error as NSError? {
      self.presentAlert("Face Detection Error", error: nsError)
      return
    }
    
    // Perform drawing on the main thread.
    DispatchQueue.main.async {
      
      guard let results = request?.results as? [VNFaceObservation] else {
        return
      }
      
      CATransaction.begin()
      CATransaction.setDisableActions(true)
      
      for (index, prediction) in results.enumerated() {
        self.boundingBoxes[index].draw(face: prediction, onImageWithBounds: self.cameraView.frame)
      }
      
      for index in results.count..<self.numBoxes {
        self.boundingBoxes[index].hide()
      }
      
      CATransaction.commit()
      
      self.semaphore.signal()
    }
  }
  
  fileprivate func handleDetectedFaceLandmarks(request: VNRequest?, error: Error?) {
    if let nsError = error as NSError? {
      self.presentAlert("Face Landmark Detection Error", error: nsError)
      return
    }
    
    let thisExecution = Date()
    let executionTime = thisExecution.timeIntervalSince(lastExecution)
    let framesPerSecond:Double = 1/executionTime
    lastExecution = thisExecution
    
    // Perform drawing on the main thread.
    DispatchQueue.main.async {
      guard let results = request?.results as? [VNFaceObservation] else {
        return
      }
      
      self.frameLabel.text = "FPS: \(framesPerSecond.format(f: ".3"))"
      
      CATransaction.begin()
      CATransaction.setDisableActions(true)
      
      for (index, prediction) in results.enumerated() {
        self.boundingBoxes[index].drawFeatures(onFaces: prediction, onImageWithBounds: self.cameraView.frame)
      }
      
      for index in results.count..<self.numBoxes {
        self.boundingBoxes[index].hide()
      }
      
      CATransaction.commit()
      
      self.semaphore2.signal()
    }
  }
  
  func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
    guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
      return
    }
    
    var requestOptions:[VNImageOption : Any] = [:]
    if let cameraIntrinsicData = CMGetAttachment(sampleBuffer, kCMSampleBufferAttachmentKey_CameraIntrinsicMatrix, nil) {
      requestOptions = [.cameraIntrinsics:cameraIntrinsicData]
    }
    let orientation = CGImagePropertyOrientation(rawValue: UInt32(EXIFOrientation.rightTop.rawValue))
    
    self.semaphore.wait()
    self.semaphore2.wait()
    do {
      let imageRequestHandler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: orientation!, options: requestOptions)
      try imageRequestHandler.perform([self.faceDetectionRequest, self.faceLandmarkRequest])
    } catch {
      print(error)
      self.semaphore.signal()
      self.semaphore2.signal()
    }
  }
  
  func sigmoid(_ val:Double) -> Double {
    return 1.0/(1.0 + exp(-val))
  }
  
  func softmax(_ values:[Double]) -> [Double] {
    if values.count == 1 { return [1.0]}
    guard let maxValue = values.max() else {
      fatalError("Softmax error")
    }
    let expValues = values.map { exp($0 - maxValue)}
    let expSum = expValues.reduce(0, +)
    return expValues.map({$0/expSum})
  }
  
  public static func softmax2(_ x: [Double]) -> [Double] {
    var x:[Float] = x.flatMap{Float($0)}
    let len = vDSP_Length(x.count)
    
    // Find the maximum value in the input array.
    var max: Float = 0
    vDSP_maxv(x, 1, &max, len)
    
    // Subtract the maximum from all the elements in the array.
    // Now the highest value in the array is 0.
    max = -max
    vDSP_vsadd(x, 1, &max, &x, 1, len)
    
    // Exponentiate all the elements in the array.
    var count = Int32(x.count)
    vvexpf(&x, x, &count)
    
    // Compute the sum of all exponentiated values.
    var sum: Float = 0
    vDSP_sve(x, 1, &sum, len)
    
    // Divide each element by the sum. This normalizes the array contents
    // so that they all add up to 1.
    vDSP_vsdiv(x, 1, &sum, &x, 1, len)
    
    let y:[Double] = x.flatMap{Double($0)}
    return y
  }
  
  enum EXIFOrientation : Int32 {
    case topLeft = 1
    case topRight
    case bottomRight
    case bottomLeft
    case leftTop
    case rightTop
    case rightBottom
    case leftBottom
    
    var isReflect:Bool {
      switch self {
      case .topLeft,.bottomRight,.rightTop,.leftBottom: return false
      default: return true
      }
    }
  }
  
  func compensatingEXIFOrientation(deviceOrientation:UIDeviceOrientation) -> EXIFOrientation
  {
    switch (deviceOrientation) {
    case (.landscapeRight): return .bottomRight
    case (.landscapeLeft): return .topLeft
    case (.portrait): return .rightTop
    case (.portraitUpsideDown): return .leftBottom
      
    case (.faceUp): return .rightTop
    case (.faceDown): return .rightTop
    case (_): fallthrough
    default:
      NSLog("Called in unrecognized orientation")
      return .rightTop
    }
  }
}


