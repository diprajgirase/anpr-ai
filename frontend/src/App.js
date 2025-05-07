import React, { useState, useRef } from 'react';
// Assuming ImageUpload.js exists in ./components/
// import ImageUpload from './components/ImageUpload';
import './App.css';

function App() {
  // State
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null); // Stores { numberPlate: '...', confidence: ... }
  const [error, setError] = useState(null);   // Stores error messages
  const [previewUrl, setPreviewUrl] = useState(null); // Stores the uploaded image preview URL
  const fileInputRef = useRef(null); // Ref to access the file input

  // Trigger backend call and preview update
  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (!file) {
      // Clear previous results if no file is selected
      setPreviewUrl(null);
      setResult(null);
      setError(null);
      return;
    }
    
    // Show preview immediately
    // Revoke previous object URL to prevent memory leaks
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setPreviewUrl(URL.createObjectURL(file));

    // Call the backend
    uploadAndDetect(file);
  };

  // Function to handle the API call
  const uploadAndDetect = async (file) => {
    setIsLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append('image', file);

    try {
      const response = await fetch('http://localhost:5000/detect', {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();
        if (data.numberPlate) {
           setResult(data);
        } else {
          setError("Received success status but no plate data.");
        }
      } else {
        const errorData = await response.json().catch(() => ({}));
        const errorMessage = errorData.error || `Request failed with status: ${response.status}`;
        setError(errorMessage); // Display specific error from backend or status
      }
    } catch (err) {
      console.error("Fetch error:", err);
      setError(err instanceof Error ? err.message : 'An unexpected error occurred during fetch.');
    } finally {
      setIsLoading(false);
    }
  };

  // Function to trigger the hidden file input
  const handleUploadButtonClick = () => {
    fileInputRef.current?.click();
  };

  return (
    <div className="app">
      {/* Header like reference */}
      <header className="app-header">
        <h1>ANPR Dashboard</h1>
        {/* Removed subtitle */}
      </header>

      {/* Main content card */}
      <main className="app-main">
        {/* Upload Section */}
        <div className="upload-section">
          <input 
            ref={fileInputRef}
            type="file" 
            accept="image/*" 
            onChange={handleFileChange} 
            style={{ display: 'none' }} // Keep the actual input hidden
          />
          <button 
            type="button" 
            className="upload-button-styled" 
            onClick={handleUploadButtonClick}
            disabled={isLoading} // Disable button while loading
          >
            {isLoading ? 'Processing...' : 'Upload Image'}
          </button>
        </div>

        {/* Display Section - Two Columns */}
        <div className="display-section">
          {/* Left Column: Image Preview */}
          <div className="image-preview-container">
            <h2>Uploaded Image</h2>
            <div className="image-box">
              {previewUrl ? (
                <img src={previewUrl} alt="Uploaded preview" />
              ) : (
                <p>No image uploaded yet.</p>
              )}
            </div>
          </div>

          {/* Right Column: Detection Result */}
          <div className="detection-result-container">
            <h2>Detected Plate</h2>
            <div className="result-box">
              {isLoading && (
                <div className="loading">
                  <div className="spinner"></div>
                </div>
              )}
              {error && (
                 <div className="error">
                    <p>{error}</p> 
                 </div>
              )} 
              {result && (
                <div className="result-content">
                  <p className="number-plate">{result.numberPlate}</p>
                  <p className="confidence">
                    Confidence: {(result.confidence * 100).toFixed(1)}%
                  </p>
                </div>
              )}
              {!isLoading && !error && !result && (
                 <p className="no-result">N/A</p> // Show N/A initially or if no result/error/loading
              )}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
