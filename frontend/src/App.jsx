import React, { useState, useMemo, useEffect, useCallback, useRef } from 'react';
import Map, { Marker, Source, Layer, NavigationControl } from 'react-map-gl';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import axios from 'axios';

// Import Mapbox GL CSS
import 'mapbox-gl/dist/mapbox-gl.css';
import './App.css';

const DepotPin = () => (
  <svg height="25" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style={{ transform: 'translate(-50%, -100%)' }}>
    <path d="M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7zm0 9.5c-1.38 0-2.5-1.12-2.5-2.5s1.12-2.5 2.5-2.5 2.5 1.12 2.5 2.5-1.12 2.5-2.5 2.5z" fill="#4CAF50"/>
  </svg>
);

const LocationPin = () => (
  <svg height="25" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style={{ transform: 'translate(-50%, -100%)' }}>
    <path d="M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7zm0 9.5c-1.38 0-2.5-1.12-2.5-2.5s1.12-2.5 2.5-2.5 2.5 1.12 2.5 2.5-1.12 2.5-2.5 2.5z" fill="#007aff"/>
  </svg>
);

const ROUTE_COLORS = ['#007aff', '#ff3b30', '#34c759', '#ff9500', '#af52de', '#5856d6'];

const isTauriEnvironment = () => {
  if (typeof window === 'undefined') {
    return false;
  }
  return '__TAURI__' in window || '__TAURI_IPC__' in window || '__TAURI_INTERNALS__' in window;
};

const routeLayerStyle = {
  id: 'route',
  type: 'line',
  layout: { 'line-join': 'round', 'line-cap': 'round' },
  paint: { 'line-color': ['get', 'color'], 'line-width': 5, 'line-opacity': 0.8 }
};

function App() {
  // const [showTraffic, setShowTraffic] = useState(false);
  const mapRef = useRef(null);
  const [viewState, setViewState] = useState({
    longitude: 174.77557,
    latitude: -41.28664,
    zoom: 14,
    pitch: 60,
  });
  const [depot, setDepot] = useState(null);
  const [locations, setLocations] = useState([]);
  const [solution, setSolution] = useState(null);
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [routeGeometry, setRouteGeometry] = useState(null);
  const [vehicles, setVehicles] = useState([{ id: 'vehicle_1', capacity: 100 }]);
  const [config, setConfig] = useState({
    iterations: 5000,
    initial_temperature: 50,
    minimum_temperature: 0.1,
    cooling_factor: 0.95,
  });
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [progress, setProgress] = useState({ current: 0, total: 0 });
  const [loggingEnabled, setLoggingEnabled] = useState(false);
  const [logPath, setLogPath] = useState('');
  const [logFlushInterval, setLogFlushInterval] = useState(100);
  const [logFormat, setLogFormat] = useState('jsonl');
  const [lastLogPath, setLastLogPath] = useState('');
  const [mlGuidanceEnabled, setMlGuidanceEnabled] = useState(false);

  const onMapLoad = useCallback((evt) => {
    mapRef.current = evt.target;
    const map = evt.target;
    map.setFog({});
    if (!map.getSource('mapbox-dem')) {
      map.addSource('mapbox-dem', {
        'type': 'raster-dem', 'url': 'mapbox://mapbox.mapbox-terrain-dem-v1',
        'tileSize': 512, 'maxzoom': 14
      });
    }
    map.setTerrain({ source: 'mapbox-dem', exaggeration: 1.5 });
    map.on('style.load', () => {
        if (!map.getSource('mapbox-traffic')) {
            map.addSource('mapbox-traffic', {
                type: 'vector', url: 'mapbox://mapbox.mapbox-traffic-v1',
            });
        }
        if (!map.getLayer('traffic-layer')) {
            map.addLayer({
                id: 'traffic-layer', type: 'line', source: 'mapbox-traffic',
                'source-layer': 'traffic',
                paint: {
                    'line-width': 2,
                    'line-color': [
                        'case',
                        ['==', ['get', 'congestion'], 'low'], '#4CAF50',
                        ['==', ['get', 'congestion'], 'moderate'], '#FFC107',
                        ['==', ['get', 'congestion'], 'heavy'], '#F44336',
                        ['==', ['get', 'congestion'], 'severe'], '#B71C1C',
                        '#E0E0E0'
                    ]
                }
            });
        }
    });
  }, []);

  // useEffect(() => {
  //   if (mapRef.current && mapRef.current.getLayer('traffic-layer')) {
  //     mapRef.current.setLayoutProperty(
  //       'traffic-layer',
  //       'visibility',
  //       showTraffic ? 'visible' : 'none'
  //     );
  //   }
  // }, [showTraffic]);

  useEffect(() => {
    if (!isTauriEnvironment()) {
      return () => {};
    }

    let unlistenProgress;
    let unlistenDone;
    let unlistenCancelled;
    let unlistenLogPath;

    const setupListeners = async () => {
      try {
        unlistenProgress = await listen('optimization-progress', (event) => {
          const { current, total } = event.payload || { current: 0, total: 0 };
          setProgress({ current, total });
        });

        unlistenDone = await listen('optimization-done', () => {
          setIsLoading(false);
          setProgress((prev) => ({ ...prev, current: prev.total }));
        });

        unlistenCancelled = await listen('optimization-cancelled', () => {
          setIsLoading(false);
          setError('Optimization cancelled');
        });

        unlistenLogPath = await listen('optimization-log-path', (event) => {
          if (event?.payload) {
            setLastLogPath(String(event.payload));
          }
        });
      } catch (err) {
        console.error('Failed to register Tauri listeners', err);
      }
    };

    setupListeners();

    return () => {
      if (unlistenProgress) unlistenProgress();
      if (unlistenDone) unlistenDone();
      if (unlistenCancelled) unlistenCancelled();
      if (unlistenLogPath) unlistenLogPath();
    };
  }, []);

  const handleMapClick = (event) => {
    if (isLoading) return;
    const { lng, lat } = event.lngLat;
    if (!depot) {
      setDepot({ id: 'depot', x: lng, y: lat, demand: 0 });
    } else {
      const newLocation = { id: `loc_${locations.length + 1}`, x: lng, y: lat, demand: 10 };
      setLocations([...locations, newLocation]);
    }
  };
  
  const handleOptimize = async () => {
    if (!depot || locations.length === 0) return;
    setError('');
    setSolution(null);
    setRouteGeometry(null);
    setIsLoading(true);
    setProgress({ current: 0, total: config.iterations });
    setLastLogPath('');
    const problemData = { depot, locations, vehicles };
    const configPayload = {
      ...config,
      iterations: Math.max(1, Math.floor(config.iterations)),
    };
    if (!isTauriEnvironment()) {
      setError('Optimization is only available in the desktop app.');
      setIsLoading(false);
      return;
    }
    const loggingPayload = loggingEnabled
      ? {
          enabled: true,
          path: (logPath || '').trim() ? logPath.trim() : null,
          flush_interval: Math.max(1, Math.floor(Number(logFlushInterval) || 100)),
          format: logFormat,
        }
      : { enabled: false };
    try {
      const result = await invoke('optimize', {
        payload: { problem: problemData, config: configPayload, logging: loggingPayload },
      });
      setSolution(result);
    } catch (e) {
      setError(`Optimization failed: ${e}`);
      setIsLoading(false);
    }
  };

  const handleClear = () => {
    setDepot(null); setLocations([]); setSolution(null);
    setRouteGeometry(null); setError('');
    setProgress({ current: 0, total: 0 });
    setLastLogPath('');
  };

  const handleAddVehicle = () => {
    const newId = `vehicle_${vehicles.length + 1}`;
    setVehicles([...vehicles, { id: newId, capacity: 100 }]);
  };

  const handleRemoveVehicle = () => {
    if (vehicles.length > 1) {
      setVehicles(vehicles.slice(0, -1));
    }
  };

  const handleIterationsChange = (event) => {
    const parsed = Number.parseInt(event.target.value, 10);
    setConfig((prev) => ({
      ...prev,
      iterations: Number.isFinite(parsed) && parsed > 0 ? parsed : prev.iterations,
    }));
  };

  const handleFloatChange = (field, min = null, max = null) => (event) => {
    const raw = Number(event.target.value);
    if (!Number.isFinite(raw)) return;
    let value = raw;
    if (min !== null && min !== undefined) {
      value = Math.max(min, value);
    }
    if (max !== null && max !== undefined) {
      value = Math.min(max, value);
    }
    setConfig((prev) => ({ ...prev, [field]: value }));
  };

  const handleCancel = async () => {
    if (!isTauriEnvironment()) {
      return;
    }
    try {
      await invoke('cancel_optimization');
    } catch (e) {
      console.error('Failed to cancel optimization', e);
    }
  };

  const handleLogFlushChange = (event) => {
    const parsed = Number.parseInt(event.target.value, 10);
    if (Number.isFinite(parsed) && parsed > 0) {
      setLogFlushInterval(parsed);
    }
  };

  const progressPercent = useMemo(() => {
    if (!progress.total) return 0;
    return Math.min(100, Math.round((progress.current / progress.total) * 100));
  }, [progress]);

  useEffect(() => {
    if (!solution || !solution.routes || solution.routes.length === 0) {
      setRouteGeometry(null);
      return;
    }

    const fetchAllRouteGeometries = async () => {
      const allPoints = [depot, ...locations];
      const accessToken = "pk.eyJ1IjoiamFja3dlZWtseSIsImEiOiJjbWc0aHR1cjExbGR0MmxuMGVkNnJ3bzBxIn0.ay9ucOZV_GVfgr7ZKLMS4w";

      const promises = solution.routes.map(route => {
        if (route.locations.length < 2) return Promise.resolve(null);
        const routeCoords = route.locations.map(id => allPoints.find(p => p.id === id)).filter(Boolean);
        if (routeCoords.length < 2) return Promise.resolve(null);
        const coordsString = routeCoords.map(p => `${p.x},${p.y}`).join(';');
        const url = `https://api.mapbox.com/directions/v5/mapbox/driving-traffic/${coordsString}?geometries=geojson&access_token=${accessToken}`;
        return axios.get(url).then(res => res.data.routes[0]?.geometry).catch(() => null);
      });

      const geometries = await Promise.all(promises);
      setRouteGeometry(geometries.filter(Boolean));
    };

    fetchAllRouteGeometries();
  }, [solution, depot, locations]);

  const routeGeoJson = useMemo(() => {
    if (!routeGeometry || routeGeometry.length === 0) return null;
    const features = routeGeometry.map((geom, index) => ({
      type: 'Feature',
      properties: { 
        id: index,
        color: ROUTE_COLORS[index % ROUTE_COLORS.length]
      }, 
      geometry: geom
    }));
    return { type: 'FeatureCollection', features: features };
  }, [routeGeometry]);

  return (
    <div className="app-container">
      <div className="map-container">
        <Map
          {...viewState}
          onMove={evt => setViewState(evt.viewState)}
          onClick={handleMapClick}
          onLoad={onMapLoad}
          mapboxAccessToken="pk.eyJ1IjoiamFja3dlZWtseSIsImEiOiJjbWc0aHR1cjExbGR0MmxuMGVkNnJ3bzBxIn0.ay9ucOZV_GVfgr7ZKLMS4w"
          mapStyle="mapbox://styles/jackweekly/cmg4jeocw004p01ps99zy2olk"
        >
          <NavigationControl position="top-left" />
          {depot && (<Marker longitude={depot.x} latitude={depot.y}><DepotPin /></Marker>)}
          {locations.map((loc) => (<Marker key={loc.id} longitude={loc.x} latitude={loc.y}><LocationPin /></Marker>))}
          {routeGeoJson && (<Source id="route-source" type="geojson" data={routeGeoJson}><Layer {...routeLayerStyle} /></Source>)}
        </Map>
      </div>
      <div className="control-panel">
        <h1>VRP Solver</h1>
        <p>
          { !depot ? 'Click to place Depot (Green).' : 'Click to add locations (Blue).' }
        </p>
        <div className="button-group">
          <button onClick={handleOptimize} disabled={isLoading || !depot || locations.length === 0}>
            {isLoading ? 'Optimizing...' : 'Optimize Routes'}
          </button>
          <button onClick={handleClear} className="clear-button" disabled={isLoading}>Clear</button>
        </div>
        <div className="advanced-settings">
          <button
            type="button"
            className="secondary-button"
            onClick={() => setShowAdvanced((prev) => !prev)}
            disabled={isLoading}
          >
            {showAdvanced ? 'Hide Advanced Settings' : 'Show Advanced Settings'}
          </button>
          {showAdvanced && (
            <div className="advanced-content">
              <label>
                Max Iterations
                <input
                  type="number"
                  min="1"
                  step="100"
                  value={config.iterations}
                  onChange={handleIterationsChange}
                  disabled={isLoading}
                />
              </label>
              <label>
                Initial Temperature
                <input
                  type="number"
                  min="0.1"
                  step="1"
                  value={config.initial_temperature}
                  onChange={handleFloatChange('initial_temperature', 0.1, null)}
                  disabled={isLoading}
                />
              </label>
              <label>
                Minimum Temperature
                <input
                  type="number"
                  min="0.01"
                  step="0.1"
                  value={config.minimum_temperature}
                  onChange={handleFloatChange('minimum_temperature', 0.01, null)}
                  disabled={isLoading}
                />
              </label>
              <label>
                Cooling Factor
                <input
                  type="number"
                  min="0.5"
                  max="0.999"
                  step="0.01"
                  value={config.cooling_factor}
                  onChange={handleFloatChange('cooling_factor', 0.5, 0.999)}
                  disabled={isLoading}
                />
              </label>
              <label className="checkbox-inline">
                <input
                  type="checkbox"
                  checked={loggingEnabled}
                  onChange={(event) => setLoggingEnabled(event.target.checked)}
                  disabled={isLoading}
                />
                Enable Move Logging
              </label>
              {loggingEnabled && (
                <div className="logging-config">
                  <label>
                    Log Directory or File
                    <input
                      type="text"
                      value={logPath}
                      onChange={(event) => setLogPath(event.target.value)}
                      placeholder="logs"
                      disabled={isLoading}
                    />
                  </label>
                  <label>
                    Flush Interval
                    <input
                      type="number"
                      min="1"
                      step="50"
                      value={logFlushInterval}
                      onChange={handleLogFlushChange}
                      disabled={isLoading}
                    />
                  </label>
                  <label>
                    Format
                    <select
                      value={logFormat}
                      onChange={(event) => setLogFormat(event.target.value)}
                      disabled={isLoading}
                    >
                      <option value="jsonl">JSON Lines</option>
                    </select>
                  </label>
                </div>
              )}
              <label className="checkbox-inline" title="Learning-based guidance is under development">
                <input
                  type="checkbox"
                  checked={mlGuidanceEnabled}
                  onChange={(event) => setMlGuidanceEnabled(event.target.checked)}
                  disabled
                />
                Use ML Guidance (coming soon)
              </label>
            </div>
          )}
        </div>
        {isLoading && (
          <div className="progress-container">
            <p>
              Iteration {Math.min(progress.current, progress.total) || 0} of {progress.total || config.iterations}
            </p>
            <div className="progress-bar">
              <div className="progress-fill" style={{ width: `${progressPercent}%` }} />
            </div>
            <button type="button" className="cancel-button" onClick={handleCancel}>
              Cancel
            </button>
          </div>
        )}
        <div className="stats">
          <p><strong>Depot:</strong> {depot ? 'Set' : 'Not Set'}</p>
          <p><strong>Locations:</strong> {locations.length}</p>
          <p><strong>Vehicles:</strong> {vehicles.length}</p>
          {lastLogPath && (
            <p className="log-path"><strong>Last Log:</strong> {lastLogPath}</p>
          )}
          <div className="button-group">
             <button onClick={handleAddVehicle} disabled={isLoading}>Add Vehicle</button>
             <button onClick={handleRemoveVehicle} disabled={isLoading || vehicles.length <= 1}>Remove</button>
          </div>
          {solution && (
            <div className="route-manifest">
              <h3>Route Manifest</h3>
              <p>
                <strong>Total Duration:</strong>{' '}
                {Number.isFinite(solution.total_cost)
                  ? (solution.total_cost / 60).toFixed(0)
                  : '—'} mins
              </p>
              {solution.routes.map((route, index) => (
                <div key={route.vehicle_id} className="route-details">
                  <strong style={{ color: ROUTE_COLORS[index % ROUTE_COLORS.length] }}>
                    {route.vehicle_id.replace('_', ' ')}
                  </strong>
                  <p>Route: {route.locations.join(' → ')}</p>
                  <p>
                    Duration: {Number.isFinite(route.total_duration)
                      ? (route.total_duration / 60).toFixed(0)
                      : '—'} mins |
                    {' '}
                    Distance: {Number.isFinite(route.total_distance)
                      ? (route.total_distance / 1000).toFixed(2)
                      : '—'} km
                  </p>
                </div>
              ))}
              {solution.unassigned_locations && solution.unassigned_locations.length > 0 && (
                <div className="unassigned-details">
                  <strong>Unassigned Locations:</strong>
                  <p>{solution.unassigned_locations.join(', ')}</p>
                </div>
              )}
            </div>
          )}
        </div>
        {error && <p className="error-message">{error}</p>}
      </div>
    </div>
  );
}

export default App;
