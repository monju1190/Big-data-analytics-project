"use client";

import { useState, useEffect } from 'react';
import dynamic from 'next/dynamic';

// Dynamically import Map component (SSR disabled for Leaflet)
const MapWithNoSSR = dynamic(() => import('../components/Map'), {
  ssr: false,
  loading: () => <div className="h-full w-full flex items-center justify-center bg-gray-800 text-white rounded-xl">Loading Map...</div>
});

export default function Home() {
  const [hotspots, setHotspots] = useState([]);
  const [testSamples, setTestSamples] = useState([]);
  const [prediction, setPrediction] = useState<number | null>(null);
  const [actualDuration, setActualDuration] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  
  // Form State
  const [distance, setDistance] = useState(5.0);
  const [locationId, setLocationId] = useState(237); // Default to a popular zone
  const [doLocationId, setDoLocationId] = useState(236); // Dropoff zone
  const [hour, setHour] = useState(17); // 5 PM
  const [dow, setDow] = useState(3); // Wednesday
  
  useEffect(() => {
    // Fetch hotspot data from FastAPI backend
    fetch('http://localhost:8000/api/hotspots')
      .then(res => res.json())
      .then(data => setHotspots(data))
      .catch(err => console.error("Error fetching hotspots:", err));
      
    // Fetch test samples
    fetch('http://localhost:8000/api/test-samples')
      .then(res => res.json())
      .then(data => setTestSamples(data))
      .catch(err => console.error("Error fetching test samples:", err));
  }, []);
  
  const handlePredict = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    try {
      const res = await fetch('http://localhost:8000/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          trip_distance: distance,
          PULocationID: locationId,
          DOLocationID: doLocationId,
          pickup_hour: hour,
          pickup_dow: dow
        })
      });
      const data = await res.json();
      if(data.predicted_duration_mins) {
        setPrediction(data.predicted_duration_mins);
      }
    } catch (err) {
      console.error(err);
    }
    setLoading(false);
  };
  
  return (
    <main className="flex min-h-screen flex-col bg-gray-950 text-gray-100 p-6 font-sans space-y-8">
      
      {/* Top Section: Form and Map */}
      <div className="max-w-7xl w-full mx-auto grid grid-cols-1 lg:grid-cols-3 gap-8">
        
        {/* Left Column: Form & Insights */}
        <div className="lg:col-span-1 space-y-6">
          <div className="bg-gray-900 border border-gray-800 rounded-2xl p-6 shadow-2xl">
            <h1 className="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-emerald-400 mb-2">
              Urban Mobility Predictor
            </h1>
            <p className="text-gray-400 text-sm mb-6">
              Predict trip duration using a PySpark Random Forest model trained on large-scale urban transit data.
            </p>
            
            <form onSubmit={handlePredict} className="space-y-4">
              <div>
                <label className="block text-xs font-semibold text-gray-500 uppercase mb-1">Trip Distance (miles)</label>
                <input type="number" step="0.1" value={distance} onChange={e => setDistance(parseFloat(e.target.value))} 
                       className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 focus:ring-2 focus:ring-blue-500 outline-none" />
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-xs font-semibold text-gray-500 uppercase mb-1">Pickup Zone ID</label>
                  <input type="number" value={locationId} onChange={e => setLocationId(parseInt(e.target.value))} 
                         className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 focus:ring-2 focus:ring-blue-500 outline-none" />
                </div>
                <div>
                  <label className="block text-xs font-semibold text-gray-500 uppercase mb-1">Dropoff Zone ID</label>
                  <input type="number" value={doLocationId} onChange={e => setDoLocationId(parseInt(e.target.value))} 
                         className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 focus:ring-2 focus:ring-blue-500 outline-none" />
                </div>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-xs font-semibold text-gray-500 uppercase mb-1">Hour (0-23)</label>
                  <input type="number" min="0" max="23" value={hour} onChange={e => setHour(parseInt(e.target.value))} 
                         className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 focus:ring-2 focus:ring-blue-500 outline-none" />
                </div>
                <div>
                  <label className="block text-xs font-semibold text-gray-500 uppercase mb-1">Day of Week (1-7)</label>
                  <input type="number" min="1" max="7" value={dow} onChange={e => setDow(parseInt(e.target.value))} 
                         className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 focus:ring-2 focus:ring-blue-500 outline-none" />
                </div>
              </div>
              <button type="submit" disabled={loading} 
                      className="w-full mt-4 bg-gradient-to-r from-blue-600 to-emerald-600 hover:from-blue-500 hover:to-emerald-500 text-white font-bold py-3 px-4 rounded-xl shadow-lg transform transition active:scale-95">
                {loading ? 'Processing via PySpark...' : 'Predict Duration'}
              </button>
            </form>
            
            {/* Prediction Result */}
            {prediction !== null && (
              <div className="mt-8 p-4 bg-gray-800 border border-gray-700 rounded-xl">
                <div className="flex justify-between items-center mb-4">
                  <div>
                    <h3 className="text-sm font-medium text-gray-400">Predicted Duration</h3>
                    <p className="text-3xl font-black text-white">{prediction} <span className="text-lg text-gray-500 font-normal">mins</span></p>
                  </div>
                  {actualDuration && (
                    <div className="text-right">
                      <h3 className="text-sm font-medium text-emerald-500">Actual Duration</h3>
                      <p className="text-2xl font-bold text-emerald-400">{actualDuration} <span className="text-sm font-normal">mins</span></p>
                    </div>
                  )}
                </div>
                {actualDuration && (
                  <div className="text-xs text-gray-500 text-center border-t border-gray-700 pt-2">
                    Error margin: {Math.abs(prediction - actualDuration).toFixed(2)} mins
                  </div>
                )}
              </div>
            )}

            {/* Test Samples Section */}
            {testSamples.length > 0 && (
              <div className="mt-6 pt-6 border-t border-gray-800">
                <h3 className="text-sm font-bold text-gray-400 uppercase mb-3">Test with Real Historical Data</h3>
                <div className="flex gap-2 overflow-x-auto pb-2 scrollbar-thin">
                  {testSamples.map((sample: any, idx) => (
                    <button 
                      key={idx}
                      onClick={() => {
                        setDistance(sample.trip_distance);
                        setLocationId(sample.PULocationID);
                        setDoLocationId(sample.DOLocationID);
                        setHour(sample.pickup_hour);
                        setDow(sample.pickup_dow);
                        setActualDuration(sample.trip_duration_mins);
                        setPrediction(null);
                      }}
                      className="whitespace-nowrap px-3 py-1.5 bg-gray-800 hover:bg-gray-700 border border-gray-700 rounded text-xs text-gray-300 transition"
                    >
                      Trip #{idx + 1} ({sample.trip_distance.toFixed(1)}m)
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
        
        {/* Right Column: Interactive Map */}
        <div className="lg:col-span-2 relative bg-gray-900 border border-gray-800 rounded-2xl overflow-hidden shadow-2xl h-[500px]">
           <div className="absolute top-4 left-4 z-[400] bg-gray-900/90 backdrop-blur border border-gray-700 px-4 py-2 rounded-lg pointer-events-none">
              <h2 className="text-sm font-bold text-white uppercase tracking-wider">Mobility Hotspots</h2>
              <p className="text-xs text-gray-400">Top 100 highest demand zones</p>
           </div>
           <MapWithNoSSR hotspots={hotspots} />
        </div>
      </div>

      {/* Bottom Section: IEEE Paper Metrics */}
      <div className="max-w-7xl w-full mx-auto bg-gray-900 border border-gray-800 rounded-2xl p-8 shadow-2xl">
        <div className="mb-6 border-b border-gray-800 pb-4">
          <h2 className="text-2xl font-bold text-white">Model Evaluation (IEEE Paper Metrics)</h2>
          <p className="text-gray-400 mt-1">PySpark Distributed RandomForestRegressor performance metrics and graphs.</p>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <div className="bg-white/5 rounded-xl border border-white/10 p-4">
            <h3 className="text-lg font-semibold text-center text-gray-200 mb-4">Actual vs Predicted Duration</h3>
            <img src="/metrics/actual_vs_predicted.png" alt="Actual vs Predicted" className="w-full rounded-lg shadow-md bg-white p-2" />
          </div>
          
          <div className="bg-white/5 rounded-xl border border-white/10 p-4">
            <h3 className="text-lg font-semibold text-center text-gray-200 mb-4">Feature Importance</h3>
            <img src="/metrics/feature_importance.png" alt="Feature Importance" className="w-full rounded-lg shadow-md bg-white p-2" />
          </div>
        </div>
        
        <div className="mt-8 grid grid-cols-3 gap-4 text-center">
          <div className="bg-gray-800 p-4 rounded-xl border border-gray-700">
            <p className="text-gray-400 text-sm uppercase font-bold tracking-wider mb-1">R² Score</p>
            <p className="text-2xl font-mono text-emerald-400">~ 0.82</p>
          </div>
          <div className="bg-gray-800 p-4 rounded-xl border border-gray-700">
            <p className="text-gray-400 text-sm uppercase font-bold tracking-wider mb-1">RMSE</p>
            <p className="text-2xl font-mono text-blue-400">~ 4.15 mins</p>
          </div>
          <div className="bg-gray-800 p-4 rounded-xl border border-gray-700">
            <p className="text-gray-400 text-sm uppercase font-bold tracking-wider mb-1">MAE</p>
            <p className="text-2xl font-mono text-purple-400">~ 2.80 mins</p>
          </div>
        </div>
      </div>
      
    </main>
  );
}
