"""
WebSocket Handler for Real-time Alerts

Handles WebSocket connections for real-time PIR monitoring and alerts.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Set, Dict, Any
from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manage WebSocket connections."""
    
    def __init__(self):
        """Initialize connection manager."""
        self.active_connections: Set[WebSocket] = set()
        self.connection_data: Dict[WebSocket, Dict[str, Any]] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str = None) -> None:
        """
        Accept and register a new connection.
        
        Args:
            websocket: WebSocket connection
            client_id: Unique client identifier
        """
        await websocket.accept()
        self.active_connections.add(websocket)
        self.connection_data[websocket] = {
            'client_id': client_id or f"client_{len(self.active_connections)}",
            'connected_at': datetime.now().isoformat(),
            'last_alert': None,
            'filters': {}
        }
        logger.info(
            f"Client {self.connection_data[websocket]['client_id']} connected. "
            f"Total connections: {len(self.active_connections)}"
        )
    
    async def disconnect(self, websocket: WebSocket) -> None:
        """
        Disconnect and unregister a connection.
        
        Args:
            websocket: WebSocket connection to disconnect
        """
        self.active_connections.discard(websocket)
        client_id = self.connection_data.pop(websocket, {}).get('client_id', 'unknown')
        logger.info(
            f"Client {client_id} disconnected. "
            f"Total connections: {len(self.active_connections)}"
        )
    
    async def send_personal(self, websocket: WebSocket, message: Dict) -> None:
        """
        Send message to a specific client.
        
        Args:
            websocket: Target WebSocket connection
            message: Message dictionary to send
        """
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Error sending personal message: {e}")
            await self.disconnect(websocket)
    
    async def broadcast(self, message: Dict, exclude: WebSocket = None) -> None:
        """
        Broadcast message to all connected clients.
        
        Args:
            message: Message dictionary to broadcast
            exclude: WebSocket connection to exclude (optional)
        """
        disconnected = []
        
        for connection in self.active_connections:
            if connection == exclude:
                continue
            
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting message: {e}")
                disconnected.append(connection)
        
        # Clean up disconnected connections
        for connection in disconnected:
            await self.disconnect(connection)
    
    async def broadcast_alert(self, alert: Dict) -> None:
        """
        Broadcast an alert to all connected clients.
        
        Args:
            alert: Alert dictionary containing:
                - risk_level: 'LOW', 'MEDIUM', 'HIGH'
                - pir_value: float (0-1)
                - timestamp: ISO format timestamp
                - pipeline_id: str
                - location: str
                - confidence: float
                - recommendation: str
        """
        alert['type'] = 'alert'
        alert['timestamp'] = alert.get('timestamp', datetime.now().isoformat())
        
        await self.broadcast(alert)
        
        logger.info(
            f"Alert broadcast: Risk={alert['risk_level']}, "
            f"PIR={alert['pir_value']:.3f}, "
            f"Recipients={len(self.active_connections)}"
        )
    
    async def send_status(self, websocket: WebSocket) -> None:
        """
        Send server status to a specific client.
        
        Args:
            websocket: Target WebSocket connection
        """
        status = {
            'type': 'status',
            'timestamp': datetime.now().isoformat(),
            'active_connections': len(self.active_connections),
            'server_status': 'running'
        }
        await self.send_personal(websocket, status)
    
    def get_client_info(self, websocket: WebSocket) -> Dict:
        """Get information about a connected client."""
        return self.connection_data.get(websocket, {})
    
    def set_client_filter(self, websocket: WebSocket, filter_config: Dict) -> None:
        """
        Set alert filters for a client.
        
        Args:
            websocket: Target WebSocket connection
            filter_config: Filter configuration dict
                - risk_levels: list of risk levels to receive ('LOW', 'MEDIUM', 'HIGH')
                - pipeline_ids: list of pipeline IDs to monitor
                - location: location filter
        """
        if websocket in self.connection_data:
            self.connection_data[websocket]['filters'] = filter_config
            logger.info(f"Filters updated for {self.connection_data[websocket]['client_id']}")
    
    def should_send_alert(self, websocket: WebSocket, alert: Dict) -> bool:
        """
        Check if alert should be sent to a client based on filters.
        
        Args:
            websocket: Target WebSocket connection
            alert: Alert dictionary
            
        Returns:
            True if alert matches client filters
        """
        filters = self.connection_data.get(websocket, {}).get('filters', {})
        
        # If no filters, send all alerts
        if not filters:
            return True
        
        # Check risk level filter
        if 'risk_levels' in filters:
            if alert.get('risk_level') not in filters['risk_levels']:
                return False
        
        # Check pipeline ID filter
        if 'pipeline_ids' in filters:
            if alert.get('pipeline_id') not in filters['pipeline_ids']:
                return False
        
        # Check location filter
        if 'location' in filters:
            if alert.get('location') != filters['location']:
                return False
        
        return True


class AlertStreamHandler:
    """Handle real-time alert streaming."""
    
    def __init__(self, manager: ConnectionManager):
        """
        Initialize alert stream handler.
        
        Args:
            manager: ConnectionManager instance
        """
        self.manager = manager
        self.alert_queue = asyncio.Queue()
        self.running = False
    
    async def start(self) -> None:
        """Start alert streaming."""
        self.running = True
        logger.info("Alert stream started")
    
    async def stop(self) -> None:
        """Stop alert streaming."""
        self.running = False
        logger.info("Alert stream stopped")
    
    async def enqueue_alert(self, alert: Dict) -> None:
        """
        Add alert to queue for broadcasting.
        
        Args:
            alert: Alert dictionary
        """
        await self.alert_queue.put(alert)
    
    async def process_alerts(self) -> None:
        """Process and broadcast queued alerts."""
        while self.running:
            try:
                # Wait for alert with timeout
                alert = await asyncio.wait_for(
                    self.alert_queue.get(),
                    timeout=1.0
                )
                
                # Broadcast to all connected clients
                await self.manager.broadcast_alert(alert)
                
            except asyncio.TimeoutError:
                # No alert in queue, continue
                continue
            except Exception as e:
                logger.error(f"Error processing alert: {e}")


class HealthCheckHandler:
    """Handle health check and monitoring."""
    
    def __init__(self, manager: ConnectionManager):
        """
        Initialize health check handler.
        
        Args:
            manager: ConnectionManager instance
        """
        self.manager = manager
        self.running = False
    
    async def start(self, interval: int = 30) -> None:
        """
        Start periodic health checks.
        
        Args:
            interval: Health check interval in seconds
        """
        self.running = True
        logger.info(f"Health check started (interval: {interval}s)")
        
        while self.running:
            await asyncio.sleep(interval)
            await self._send_health_check()
    
    async def stop(self) -> None:
        """Stop health checks."""
        self.running = False
        logger.info("Health check stopped")
    
    async def _send_health_check(self) -> None:
        """Send health check message to all clients."""
        health_check = {
            'type': 'health_check',
            'timestamp': datetime.now().isoformat(),
            'status': 'healthy',
            'connected_clients': len(self.manager.active_connections)
        }
        
        await self.manager.broadcast(health_check)


class MessageHandler:
    """Handle incoming WebSocket messages."""
    
    def __init__(self, manager: ConnectionManager):
        """
        Initialize message handler.
        
        Args:
            manager: ConnectionManager instance
        """
        self.manager = manager
    
    async def handle_message(self, websocket: WebSocket, data: Dict) -> None:
        """
        Handle incoming WebSocket message.
        
        Args:
            websocket: WebSocket connection
            data: Message data dictionary
        """
        message_type = data.get('type', 'unknown')
        
        handlers = {
            'filter': self._handle_filter,
            'status': self._handle_status,
            'subscribe': self._handle_subscribe,
            'unsubscribe': self._handle_unsubscribe,
        }
        
        handler = handlers.get(message_type, self._handle_unknown)
        await handler(websocket, data)
    
    async def _handle_filter(self, websocket: WebSocket, data: Dict) -> None:
        """Handle filter configuration message."""
        filter_config = data.get('filter_config', {})
        self.manager.set_client_filter(websocket, filter_config)
        
        response = {
            'type': 'filter_ack',
            'timestamp': datetime.now().isoformat(),
            'status': 'ok',
            'filters': filter_config
        }
        await self.manager.send_personal(websocket, response)
    
    async def _handle_status(self, websocket: WebSocket, data: Dict) -> None:
        """Handle status request message."""
        await self.manager.send_status(websocket)
    
    async def _handle_subscribe(self, websocket: WebSocket, data: Dict) -> None:
        """Handle subscription message."""
        pipeline_id = data.get('pipeline_id')
        
        response = {
            'type': 'subscribe_ack',
            'timestamp': datetime.now().isoformat(),
            'status': 'ok',
            'pipeline_id': pipeline_id
        }
        await self.manager.send_personal(websocket, response)
    
    async def _handle_unsubscribe(self, websocket: WebSocket, data: Dict) -> None:
        """Handle unsubscription message."""
        pipeline_id = data.get('pipeline_id')
        
        response = {
            'type': 'unsubscribe_ack',
            'timestamp': datetime.now().isoformat(),
            'status': 'ok',
            'pipeline_id': pipeline_id
        }
        await self.manager.send_personal(websocket, response)
    
    async def _handle_unknown(self, websocket: WebSocket, data: Dict) -> None:
        """Handle unknown message type."""
        logger.warning(f"Unknown message type: {data.get('type')}")
        
        response = {
            'type': 'error',
            'timestamp': datetime.now().isoformat(),
            'error': 'Unknown message type',
            'received_type': data.get('type')
        }
        await self.manager.send_personal(websocket, response)
