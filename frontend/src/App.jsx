import React, { useState } from 'react';
import { Upload, Image, CheckCircle, AlertCircle, Loader2, XCircle } from 'lucide-react';

export default function App() {
  const [uploadedImage, setUploadedImage] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [annotatedImage, setAnnotatedImage] = useState(null);

  // Use environment variable for API URL
  const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000';

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      processImage(file);
    }
  };

  const handleFileInput = (e) => {
    const file = e.target.files[0];
    if (file) {
      processImage(file);
    }
  };

  const processImage = async (file) => {
    // Reset states
    setError(null);
    setResults(null);
    setAnnotatedImage(null);
    setIsProcessing(true);

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
      setUploadedImage(e.target.result);
    };
    reader.readAsDataURL(file);

    // Prepare form data
    const formData = new FormData();
    formData.append('image', file);

    try {
      // Call the full detection API with parameters
      const response = await fetch(
        `${API_BASE_URL}/detect-full?plate_conf=0.25&char_conf=0.15&return_image=true`,
        {
          method: 'POST',
          body: formData,
        }
      );

      if (!response.ok) {
        throw new Error(`API Error: ${response.status}`);
      }

      const data = await response.json();

      if (data.success && data.results && data.results.length > 0) {
        setResults(data);
        
        // Set annotated image if available
        if (data.annotated_image) {
          setAnnotatedImage(`data:image/jpeg;base64,${data.annotated_image}`);
        }
      } else {
        setError('No license plates detected in the image. Please try another image.');
      }
    } catch (err) {
      console.error('Error processing image:', err);
      setError(`Failed to process image: ${err.message}. Make sure the API is running on ${API_BASE_URL}`);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleReset = () => {
    setUploadedImage(null);
    setResults(null);
    setError(null);
    setAnnotatedImage(null);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-blue-600 rounded-lg flex items-center justify-center">
              <Image className="w-6 h-6 text-white" />
            </div>
            <h1 className="text-2xl font-bold text-gray-900">
              Egyptian Plate Recognition System
            </h1>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          
          {/* Upload Section */}
          <div className="space-y-6">
            <div className="bg-white rounded-2xl shadow-lg p-8">
              <h2 className="text-xl font-semibold text-gray-900 mb-6">
                Upload Vehicle Image
              </h2>
              
              {!uploadedImage ? (
                <div
                  onDragOver={handleDragOver}
                  onDragLeave={handleDragLeave}
                  onDrop={handleDrop}
                  className={`border-3 border-dashed rounded-xl p-12 text-center transition-all ${
                    isDragging
                      ? 'border-blue-500 bg-blue-50'
                      : 'border-gray-300 bg-gray-50 hover:bg-gray-100'
                  }`}
                >
                  <Upload className={`w-16 h-16 mx-auto mb-4 ${
                    isDragging ? 'text-blue-500' : 'text-gray-400'
                  }`} />
                  <p className="text-lg font-medium text-gray-700 mb-2">
                    Drag and drop an image here
                  </p>
                  <p className="text-sm text-gray-500 mb-6">
                    or click the button below to browse
                  </p>
                  <label className="inline-block">
                    <input
                      type="file"
                      accept="image/*"
                      onChange={handleFileInput}
                      className="hidden"
                    />
                    <span className="bg-blue-600 text-white px-8 py-3 rounded-lg font-medium cursor-pointer hover:bg-blue-700 transition-colors inline-block">
                      Choose Image
                    </span>
                  </label>
                </div>
              ) : (
                <div className="space-y-4">
                  <div className="relative rounded-xl overflow-hidden border-2 border-gray-200">
                    <img
                      src={annotatedImage || uploadedImage}
                      alt="Uploaded car"
                      className="w-full h-auto"
                    />
                    {isProcessing && (
                      <div className="absolute inset-0 bg-black bg-opacity-50 flex items-center justify-center">
                        <div className="text-center">
                          <Loader2 className="w-12 h-12 text-white animate-spin mx-auto mb-2" />
                          <p className="text-white font-medium">Processing...</p>
                        </div>
                      </div>
                    )}
                  </div>
                  <button
                    onClick={handleReset}
                    disabled={isProcessing}
                    className="w-full bg-gray-100 text-gray-700 px-6 py-3 rounded-lg font-medium hover:bg-gray-200 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    Upload Different Image
                  </button>
                </div>
              )}
            </div>

            {/* Sample Info Card */}
            <div className="bg-blue-50 rounded-xl p-6 border border-blue-200">
              <div className="flex items-start gap-3">
                <AlertCircle className="w-5 h-5 text-blue-600 mt-0.5 flex-shrink-0" />
                <div>
                  <h3 className="font-semibold text-blue-900 mb-1">
                    Image Requirements
                  </h3>
                  <ul className="text-sm text-blue-800 space-y-1">
                    <li>• Clear view of license plate</li>
                    <li>• Good lighting conditions</li>
                    <li>• Supports JPG, PNG formats</li>
                    <li>• Maximum file size: 10MB</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>

          {/* Results Section */}
          <div className="space-y-6">
            <div className="bg-white rounded-2xl shadow-lg p-8">
              <h2 className="text-xl font-semibold text-gray-900 mb-6">
                Recognition Results
              </h2>
              
              {isProcessing ? (
                <div className="text-center py-16">
                  <Loader2 className="w-16 h-16 text-blue-600 animate-spin mx-auto mb-4" />
                  <p className="text-gray-700 text-lg font-medium">
                    Analyzing image...
                  </p>
                  <p className="text-gray-500 text-sm mt-2">
                    This may take a few seconds
                  </p>
                </div>
              ) : error ? (
                <div className="text-center py-16">
                  <div className="w-20 h-20 bg-red-100 rounded-full flex items-center justify-center mx-auto mb-4">
                    <XCircle className="w-10 h-10 text-red-600" />
                  </div>
                  <p className="text-red-600 text-lg font-medium mb-2">
                    Detection Failed
                  </p>
                  <p className="text-gray-600 text-sm max-w-md mx-auto">
                    {error}
                  </p>
                </div>
              ) : !results ? (
                <div className="text-center py-16">
                  <div className="w-20 h-20 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-4">
                    <Image className="w-10 h-10 text-gray-400" />
                  </div>
                  <p className="text-gray-500 text-lg">
                    Upload an image to see results
                  </p>
                </div>
              ) : (
                <div className="space-y-6">
                  {results.results.map((plate, index) => (
                    <div key={index}>
                      {/* Detected Plate */}
                      <div className="bg-gradient-to-br from-green-50 to-emerald-50 rounded-xl p-6 border-2 border-green-200 mb-4">
                        <div className="flex items-center gap-2 mb-3">
                          <CheckCircle className="w-5 h-5 text-green-600" />
                          <span className="text-sm font-semibold text-green-900">
                            PLATE {plate.plate_number} DETECTED
                          </span>
                        </div>
                        <div className="bg-white rounded-lg p-4 mb-4 shadow-sm">
                          <div className="text-center">
                            <p className="text-4xl font-bold text-gray-900 tracking-wider" dir="rtl">
                              {plate.arabic_text}
                            </p>
                          </div>
                        </div>
                        <div className="space-y-1 text-sm text-gray-600">
                          <div>
                            <span className="font-medium">Arabic:</span> {plate.arabic_text}
                          </div>
                          <div>
                            <span className="font-medium">English:</span> {plate.detected_text}
                          </div>
                        </div>
                      </div>

                      {/* Confidence Score */}
                      <div className="bg-gray-50 rounded-xl p-6 border border-gray-200 mb-4">
                        <h3 className="text-sm font-semibold text-gray-700 mb-3">
                          Confidence Score
                        </h3>
                        <div className="flex items-end gap-3 mb-2">
                          <span className="text-5xl font-bold text-blue-600">
                            {Math.round(plate.plate_confidence * 100)}
                          </span>
                          <span className="text-2xl font-semibold text-gray-500 mb-1">%</span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
                          <div 
                            className="bg-gradient-to-r from-blue-500 to-blue-600 h-full rounded-full transition-all duration-1000" 
                            style={{ width: `${plate.plate_confidence * 100}%` }}
                          ></div>
                        </div>
                      </div>

                      {/* Character Details */}
                      {plate.characters && plate.characters.length > 0 && (
                        <div className="bg-gray-50 rounded-xl p-6 border border-gray-200 mb-4">
                          <h3 className="text-sm font-semibold text-gray-700 mb-3">
                            Detected Characters ({plate.characters.length})
                          </h3>
                          <div className="grid grid-cols-4 sm:grid-cols-6 gap-2">
                            {plate.characters.map((char, charIndex) => (
                              <div 
                                key={charIndex}
                                className="bg-white rounded-lg p-3 text-center border border-gray-200"
                              >
                                <div className="text-2xl font-bold text-gray-900 mb-1">
                                  {char.class}
                                </div>
                                <div className="text-xs text-gray-500">
                                  {Math.round(char.confidence * 100)}%
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  ))}

                  {/* Summary Info */}
                  <div className="grid grid-cols-2 gap-4">
                    <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
                      <p className="text-xs text-gray-500 mb-1">Plates Found</p>
                      <p className="text-lg font-semibold text-gray-900">
                        {results.plates_found}
                      </p>
                    </div>
                    <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
                      <p className="text-xs text-gray-500 mb-1">Plate Type</p>
                      <p className="text-lg font-semibold text-gray-900">Egyptian</p>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* API Status Section */}
        <div className="mt-12 bg-white rounded-2xl shadow-lg p-8">
          <h2 className="text-xl font-semibold text-gray-900 mb-6">
            System Information
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="bg-blue-50 rounded-lg p-4 border border-blue-200">
              <p className="text-sm text-blue-700 mb-1">API Endpoint</p>
              <p className="text-sm font-mono text-blue-900 break-all">
                {API_BASE_URL}
              </p>
            </div>
            <div className="bg-green-50 rounded-lg p-4 border border-green-200">
              <p className="text-sm text-green-700 mb-1">Detection Model</p>
              <p className="text-sm font-semibold text-green-900">
                YOLO v8
              </p>
            </div>
            <div className="bg-purple-50 rounded-lg p-4 border border-purple-200">
              <p className="text-sm text-purple-700 mb-1">Supported Plates</p>
              <p className="text-sm font-semibold text-purple-900">
                Egyptian Arabic
              </p>
            </div>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-gray-200 mt-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <p className="text-center text-sm text-gray-500">
            Egyptian Plate Recognition System - Connected to Flask API
          </p>
        </div>
      </footer>
    </div>
  );
}
