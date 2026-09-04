"use client";

import { useEffect } from 'react';
import { MapContainer, TileLayer, CircleMarker, Popup } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';

// Fix for default Leaflet icon in Next.js
const DefaultIcon = L.icon({
    iconUrl: 'https://unpkg.com/leaflet@1.7.1/dist/images/marker-icon.png',
    shadowUrl: 'https://unpkg.com/leaflet@1.7.1/dist/images/marker-shadow.png',
    iconSize: [25, 41],
    iconAnchor: [12, 41]
});
L.Marker.prototype.options.icon = DefaultIcon;

export default function Map({ hotspots }: { hotspots: any[] }) {
    return (
        <MapContainer center={[40.7128, -74.0060]} zoom={11} style={{ height: '100%', width: '100%' }}>
            {/* Esri Dark Gray Theme Tile Layer - completely free and no API key required */}
            <TileLayer
                url="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}"
                attribution='Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ'
            />
            {hotspots.map((hotspot, idx) => {
                if (!hotspot.latitude || !hotspot.longitude) return null;
                
                // Calculate dynamic radius (clamped between 5 and 35) to prevent overwhelming the map
                const maxExpectedTrips = 3000;
                const dynamicRadius = Math.min(Math.max((hotspot.trip_count / maxExpectedTrips) * 35, 5), 35);
                
                // Determine color based on intensity
                const isHighDemand = hotspot.trip_count > 1500;
                
                return (
                    <CircleMarker
                        key={idx}
                        center={[hotspot.latitude, hotspot.longitude]}
                        radius={dynamicRadius}
                        fillColor={isHighDemand ? "#ef4444" : "#06b6d4"} // Red for very high demand, Cyan for others
                        color={isHighDemand ? "#f87171" : "#22d3ee"}
                        weight={2}
                        opacity={0.9}
                        fillOpacity={isHighDemand ? 0.6 : 0.4}
                    >
                        <Popup className="custom-popup">
                            <div className="font-sans">
                                <div className="text-xs text-gray-500 uppercase tracking-wider mb-1 font-bold">Zone ID: {hotspot.LocationID}</div>
                                <div className="text-sm font-bold text-gray-800 mb-1">{hotspot.zone}</div>
                                <div className="text-xs bg-blue-100 text-blue-800 px-2 py-1 rounded inline-block font-medium">
                                    {hotspot.trip_count.toLocaleString()} Trips
                                </div>
                            </div>
                        </Popup>
                    </CircleMarker>
                );
            })}
        </MapContainer>
    );
}
