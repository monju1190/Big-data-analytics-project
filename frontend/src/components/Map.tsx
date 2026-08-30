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
            <TileLayer
                url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            />
            {hotspots.map((hotspot, idx) => (
                hotspot.latitude && hotspot.longitude && (
                    <CircleMarker
                        key={idx}
                        center={[hotspot.latitude, hotspot.longitude]}
                        radius={Math.max(hotspot.trip_count / 10, 5)}
                        fillColor="#facc15"
                        color="#facc15"
                        weight={1}
                        opacity={0.8}
                        fillOpacity={0.4}
                    >
                        <Popup>
                            <strong>{hotspot.zone}</strong><br/>
                            Pickups: {hotspot.trip_count}
                        </Popup>
                    </CircleMarker>
                )
            ))}
        </MapContainer>
    );
}
