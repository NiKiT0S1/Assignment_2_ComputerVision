import React, { useState, useRef, useEffect } from 'react';

const App = () => {
  const [image, setImage] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [variants, setVariants] = useState([]);
  const [loading, setLoading] = useState(false);
  const [stats, setStats] = useState(null);
  const [activeTab, setActiveTab] = useState('upload'); // upload, draw, variants

  const canvasRef = useRef(null);
  const [isDrawing, setIsDrawing] = useState(false);

  // Fetch stats on mount
  useEffect(() => {
    fetch('/api/stats')
      .then(res => res.json())
      .then(data => setStats(data))
      .catch(err => console.error('Error fetching stats:', err));
  }, []);

  // Drawing functions
  const startDrawing = (e) => {
    setIsDrawing(true);
    draw(e);
  };

  const stopDrawing = () => {
    setIsDrawing(false);
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    ctx.beginPath();
  };

  const draw = (e) => {
    if (!isDrawing) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    ctx.lineWidth = 20;
    ctx.lineCap = 'round';
    ctx.strokeStyle = 'black';

    ctx.lineTo(x, y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x, y);
  };

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    setPrediction(null);
    setVariants([]);
  };

  const getCanvasImage = () => {
    const canvas = canvasRef.current;
    return canvas.toDataURL('image/png');
  };

  // Handle file upload
  const handleUpload = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      setImage(event.target.result);
      setPrediction(null);
      setVariants([]);
    };
    reader.readAsDataURL(file);
  };

  // Predict
  const handlePredict = async () => {
    let imageData = activeTab === 'draw' ? getCanvasImage() : image;
    if (!imageData) return;

    setLoading(true);
    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: imageData })
      });

      const data = await response.json();
      setPrediction(data);
    } catch (err) {
      console.error('Prediction error:', err);
      alert('Error making prediction');
    }
    setLoading(false);
  };

  // Generate variants
  const handleGenerateVariants = async () => {
    let imageData = activeTab === 'draw' ? getCanvasImage() : image;
    if (!imageData) return;

    setLoading(true);
    try {
      const response = await fetch('/api/generate_variants', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: imageData })
      });

      const data = await response.json();
      setVariants(data.variants);
      setActiveTab('variants');
    } catch (err) {
      console.error('Variant generation error:', err);
      alert('Error generating variants');
    }
    setLoading(false);
  };

  return (
    <div style={{ minHeight: '100vh', background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)' }}>
      {/* Header */}
      <div style={{ background: 'rgba(0,0,0,0.3)', padding: '20px', color: 'white' }}>
        <div style={{ maxWidth: '1200px', margin: '0 auto' }}>
          <h1 style={{ margin: 0, fontSize: '28px', fontWeight: 'bold' }}>
            🎯 Handwritten Digits Recognition
          </h1>
          <p style={{ margin: '5px 0 0 0', opacity: 0.9 }}>
            Assignment 2 - Computer Vision | Astana IT University
          </p>
        </div>
      </div>

      {/* Main Content */}
      <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '30px 20px' }}>

        {/* Stats Cards */}
        {stats && (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '15px', marginBottom: '30px' }}>
            <div style={{ background: 'white', borderRadius: '10px', padding: '20px', boxShadow: '0 4px 6px rgba(0,0,0,0.1)' }}>
              <div style={{ fontSize: '14px', color: '#666', marginBottom: '5px' }}>Baseline Accuracy</div>
              <div style={{ fontSize: '32px', fontWeight: 'bold', color: '#e74c3c' }}>{(stats.baseline_accuracy * 100).toFixed(1)}%</div>
              <div style={{ fontSize: '12px', color: '#999' }}>SVM (A1)</div>
            </div>

            <div style={{ background: 'white', borderRadius: '10px', padding: '20px', boxShadow: '0 4px 6px rgba(0,0,0,0.1)' }}>
              <div style={{ fontSize: '14px', color: '#666', marginBottom: '5px' }}>Current Accuracy</div>
              <div style={{ fontSize: '32px', fontWeight: 'bold', color: '#27ae60' }}>{(stats.improved_accuracy * 100).toFixed(1)}%</div>
              <div style={{ fontSize: '12px', color: '#999' }}>LightGBM + ArtAug</div>
            </div>

            <div style={{ background: 'white', borderRadius: '10px', padding: '20px', boxShadow: '0 4px 6px rgba(0,0,0,0.1)' }}>
              <div style={{ fontSize: '14px', color: '#666', marginBottom: '5px' }}>Improvement</div>
              <div style={{ fontSize: '32px', fontWeight: 'bold', color: '#3498db' }}>{stats.improvement}</div>
              <div style={{ fontSize: '12px', color: '#999' }}>vs Baseline</div>
            </div>

            <div style={{ background: 'white', borderRadius: '10px', padding: '20px', boxShadow: '0 4px 6px rgba(0,0,0,0.1)' }}>
              <div style={{ fontSize: '14px', color: '#666', marginBottom: '5px' }}>Training Data</div>
              <div style={{ fontSize: '32px', fontWeight: 'bold', color: '#9b59b6' }}>{stats.training_samples.total}</div>
              <div style={{ fontSize: '12px', color: '#999' }}>{stats.training_samples.synthetic} synthetic</div>
            </div>
          </div>
        )}

        {/* Main Card */}
        <div style={{ background: 'white', borderRadius: '15px', padding: '30px', boxShadow: '0 10px 30px rgba(0,0,0,0.2)' }}>

          {/* Tabs */}
          <div style={{ display: 'flex', gap: '10px', marginBottom: '30px', borderBottom: '2px solid #eee', paddingBottom: '10px' }}>
            <button
              onClick={() => setActiveTab('upload')}
              style={{
                padding: '10px 20px',
                border: 'none',
                background: activeTab === 'upload' ? '#667eea' : 'transparent',
                color: activeTab === 'upload' ? 'white' : '#666',
                borderRadius: '8px',
                cursor: 'pointer',
                fontWeight: 'bold',
                transition: 'all 0.3s'
              }}
            >
              📤 Upload Image
            </button>

            <button
              onClick={() => setActiveTab('draw')}
              style={{
                padding: '10px 20px',
                border: 'none',
                background: activeTab === 'draw' ? '#667eea' : 'transparent',
                color: activeTab === 'draw' ? 'white' : '#666',
                borderRadius: '8px',
                cursor: 'pointer',
                fontWeight: 'bold',
                transition: 'all 0.3s'
              }}
            >
              ✏️ Draw Digit
            </button>

            <button
              onClick={() => setActiveTab('variants')}
              style={{
                padding: '10px 20px',
                border: 'none',
                background: activeTab === 'variants' ? '#667eea' : 'transparent',
                color: activeTab === 'variants' ? 'white' : '#666',
                borderRadius: '8px',
                cursor: 'pointer',
                fontWeight: 'bold',
                transition: 'all 0.3s'
              }}
              disabled={variants.length === 0}
            >
              🎨 ArtAug Variants ({variants.length})
            </button>
          </div>

          {/* Upload Tab */}
          {activeTab === 'upload' && (
            <div>
              <div style={{
                border: '3px dashed #ddd',
                borderRadius: '10px',
                padding: '40px',
                textAlign: 'center',
                marginBottom: '20px'
              }}>
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleUpload}
                  style={{ display: 'none' }}
                  id="file-input"
                />
                <label
                  htmlFor="file-input"
                  style={{
                    cursor: 'pointer',
                    display: 'inline-block',
                    padding: '15px 30px',
                    background: '#667eea',
                    color: 'white',
                    borderRadius: '8px',
                    fontWeight: 'bold'
                  }}
                >
                  Choose Image
                </label>
                <p style={{ marginTop: '15px', color: '#666' }}>
                  Upload a handwritten digit (0-9)
                </p>
              </div>

              {image && (
                <div style={{ textAlign: 'center', marginBottom: '20px' }}>
                  <img
                    src={image}
                    alt="Uploaded"
                    style={{
                      maxWidth: '300px',
                      maxHeight: '300px',
                      border: '2px solid #ddd',
                      borderRadius: '8px'
                    }}
                  />
                </div>
              )}
            </div>
          )}

          {/* Draw Tab */}
          {activeTab === 'draw' && (
            <div style={{ textAlign: 'center' }}>
              <canvas
                ref={canvasRef}
                width={400}
                height={400}
                onMouseDown={startDrawing}
                onMouseUp={stopDrawing}
                onMouseMove={draw}
                onMouseLeave={stopDrawing}
                style={{
                  border: '3px solid #667eea',
                  borderRadius: '10px',
                  cursor: 'crosshair',
                  background: 'white',
                  touchAction: 'none'
                }}
              />
              <div style={{ marginTop: '15px' }}>
                <button
                  onClick={clearCanvas}
                  style={{
                    padding: '10px 20px',
                    background: '#e74c3c',
                    color: 'white',
                    border: 'none',
                    borderRadius: '8px',
                    cursor: 'pointer',
                    fontWeight: 'bold'
                  }}
                >
                  🗑️ Clear Canvas
                </button>
              </div>
            </div>
          )}

          {/* Variants Tab */}
          {activeTab === 'variants' && variants.length > 0 && (
            <div>
              <h3 style={{ marginBottom: '20px', color: '#333' }}>
                🎨 ArtAug Variants (12 Synthesis Methods)
              </h3>
              <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))',
                gap: '15px'
              }}>
                {variants.map((variant, idx) => (
                  <div
                    key={idx}
                    style={{
                      border: '2px solid #eee',
                      borderRadius: '8px',
                      padding: '10px',
                      textAlign: 'center',
                      background: '#f9f9f9'
                    }}
                  >
                    <img
                      src={variant.image}
                      alt={variant.name}
                      style={{
                        width: '100%',
                        height: '120px',
                        objectFit: 'contain',
                        marginBottom: '8px',
                        background: 'white',
                        borderRadius: '5px'
                      }}
                    />
                    <div style={{ fontSize: '12px', fontWeight: 'bold', color: '#333' }}>
                      {variant.description}
                    </div>
                    <div style={{ fontSize: '11px', color: '#666', marginTop: '5px' }}>
                      Pred: {variant.prediction} ({(variant.confidence * 100).toFixed(0)}%)
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Action Buttons */}
          {(activeTab === 'upload' || activeTab === 'draw') && (
            <div style={{ display: 'flex', gap: '15px', justifyContent: 'center', marginTop: '30px' }}>
              <button
                onClick={handlePredict}
                disabled={loading || (!image && activeTab === 'upload')}
                style={{
                  padding: '15px 30px',
                  background: loading ? '#95a5a6' : '#27ae60',
                  color: 'white',
                  border: 'none',
                  borderRadius: '8px',
                  cursor: loading ? 'not-allowed' : 'pointer',
                  fontWeight: 'bold',
                  fontSize: '16px',
                  transition: 'all 0.3s'
                }}
              >
                {loading ? '⏳ Processing...' : '🎯 Predict Digit'}
              </button>

              <button
                onClick={handleGenerateVariants}
                disabled={loading || (!image && activeTab === 'upload')}
                style={{
                  padding: '15px 30px',
                  background: loading ? '#95a5a6' : '#9b59b6',
                  color: 'white',
                  border: 'none',
                  borderRadius: '8px',
                  cursor: loading ? 'not-allowed' : 'pointer',
                  fontWeight: 'bold',
                  fontSize: '16px',
                  transition: 'all 0.3s'
                }}
              >
                {loading ? '⏳ Generating...' : '🎨 Generate 12 Variants'}
              </button>
            </div>
          )}

          {/* Prediction Results */}
          {prediction && (
            <div style={{
              marginTop: '30px',
              padding: '25px',
              background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
              borderRadius: '10px',
              color: 'white'
            }}>
              <h2 style={{ margin: '0 0 15px 0' }}>
                Prediction Result: <span style={{ fontSize: '48px' }}>{prediction.prediction}</span>
              </h2>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', marginTop: '20px' }}>
                <div>
                  <div style={{ fontSize: '14px', opacity: 0.8, marginBottom: '5px' }}>Confidence</div>
                  <div style={{ fontSize: '24px', fontWeight: 'bold' }}>
                    {(prediction.confidence * 100).toFixed(2)}%
                  </div>
                </div>
                <div>
                  <div style={{ fontSize: '14px', opacity: 0.8, marginBottom: '5px' }}>Inference Time</div>
                  <div style={{ fontSize: '24px', fontWeight: 'bold' }}>
                    {prediction.inference_time_ms} ms
                  </div>
                </div>
              </div>

              {/* Probability Distribution */}
              <div style={{ marginTop: '20px' }}>
                <div style={{ fontSize: '14px', opacity: 0.8, marginBottom: '10px' }}>
                  Probability Distribution
                </div>
                {Object.entries(prediction.probabilities).map(([digit, prob]) => (
                  <div key={digit} style={{ marginBottom: '8px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', marginBottom: '3px' }}>
                      <span>Digit {digit}</span>
                      <span>{(prob * 100).toFixed(1)}%</span>
                    </div>
                    <div style={{
                      width: '100%',
                      height: '6px',
                      background: 'rgba(255,255,255,0.3)',
                      borderRadius: '3px',
                      overflow: 'hidden'
                    }}>
                      <div style={{
                        width: `${prob * 100}%`,
                        height: '100%',
                        background: 'white',
                        transition: 'width 0.3s'
                      }} />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div style={{
          textAlign: 'center',
          marginTop: '30px',
          color: 'white',
          opacity: 0.8
        }}>
          <p style={{ margin: '5px 0' }}>
            🎓 Assignment 2 - Computer Vision Course
          </p>
          <p style={{ margin: '5px 0', fontSize: '14px' }}>
            SOTA Method: ArtAug (Artistic Augmentation) | Model: LightGBM
          </p>
        </div>
      </div>
    </div>
  );
};

export default App;