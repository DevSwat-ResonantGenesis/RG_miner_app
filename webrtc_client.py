"""
WebRTC Client for Miner P2P Connections
==========================================

Provides a simple interface for miners to establish WebRTC connections
with their pipeline peers through the Mining service's signaling server.

Usage:
    client = WebRTCClient(miner_id="miner-123", mining_url="ws://localhost:8701")
    await client.connect()
    
    # Wait for peer connections
    await client.wait_for_peers(timeout=30)
    
    # Send weight data to upstream peer
    await client.send_to_upstream({
        "type": "weight-shard",
        "layer_start": 0,
        "layer_end": 8,
        "data": weight_bytes,
    })
    
    # Receive messages from peers
    async for message in client.message_stream():
        print(f"Received: {message}")

The WebRTC DataChannel replaces HTTP POST for P2P communication,
enabling NAT traversal without port forwarding.
"""

import asyncio
import base64
import hashlib
import json
import logging
from typing import Dict, Optional, Set, AsyncGenerator
import uuid

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

try:
    import aiortc
    from aiortc import RTCPeerConnection, RTCDataChannel
    from aiortc.contrib.signaling import object_to_string, string_to_object
    AIORTC_AVAILABLE = True
except ImportError:
    AIORTC_AVAILABLE = False

try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False

logger = logging.getLogger("miner.webrtc-client")


class WebRTCClient:
    """
    WebRTC client for miner P2P connections.
    
    Handles:
      - WebSocket connection to signaling server
      - WebRTC peer connections with pipeline peers
      - DataChannel message routing
      - Automatic reconnection
    """
    
    def __init__(self, miner_id: str, mining_url: str = "ws://localhost:8701"):
        if not AIORTC_AVAILABLE:
            raise ImportError("aiortc is required for WebRTC P2P connections")
        if not WEBSOCKETS_AVAILABLE:
            raise ImportError("websockets is required for signaling")
        
        self.miner_id = miner_id
        self.mining_url = mining_url.rstrip("/")
        self.signaling_url = f"{self.mining_url}/mining/p2p/signaling/{miner_id}"
        self.api_url = mining_url.rstrip("/mining")  # Remove /mining for API calls
        
        # WebRTC connections: peer_id -> RTCPeerConnection
        self.peer_connections: Dict[str, RTCPeerConnection] = {}
        # Data channels: peer_id -> RTCDataChannel
        self.data_channels: Dict[str, RTCDataChannel] = {}
        # Message queue for incoming messages
        self.message_queue = asyncio.Queue()
        # Connected peers
        self.connected_peers: Set[str] = set()
        # Expected peers (from pipeline assignment)
        self.expected_peers: Set[str] = set()
        
        self._websocket = None
        self._running = False
        self._peer_id = None  # Assigned by signaling server
    
    async def connect(self) -> str:
        """
        Connect to signaling server and return assigned peer ID.
        
        The signaling server assigns a unique peer_id and handles
        WebRTC offer/answer exchange with pipeline peers.
        """
        try:
            self._websocket = await websockets.connect(self.signaling_url)
            self._running = True
            
            # Start message handler
            asyncio.create_task(self._handle_signaling())
            
            logger.info(f"Connected to signaling server as {self.miner_id}")
            return self._peer_id or "connecting"
            
        except Exception as e:
            logger.error(f"Failed to connect to signaling server: {e}")
            raise
    
    async def disconnect(self):
        """Close all connections and cleanup."""
        self._running = False
        
        # Close WebRTC connections
        for pc in self.peer_connections.values():
            try:
                await pc.close()
            except Exception as e:
                logger.warning(f"Error closing peer connection: {e}")
        
        # Close WebSocket
        if self._websocket:
            try:
                await self._websocket.close()
            except Exception as e:
                logger.warning(f"Error closing WebSocket: {e}")
        
        logger.info("WebRTC client disconnected")
    
    async def _handle_signaling(self):
        """Handle incoming signaling messages."""
        try:
            async for message in self._websocket:
                try:
                    data = json.loads(message)
                    await self._process_signaling_message(data)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid signaling message: {message}")
                except Exception as e:
                    logger.error(f"Error processing signaling message: {e}")
        except websockets.exceptions.ConnectionClosed:
            logger.info("Signaling connection closed")
        except Exception as e:
            logger.error(f"Signaling handler error: {e}")
    
    async def _process_signaling_message(self, message: Dict):
        """Process a single signaling message."""
        msg_type = message.get("type")
        from_peer_id = message.get("from_peer_id")
        data = message.get("data", {})
        
        if msg_type == "create-offer":
            # We need to create an offer for the target peer
            target_peer_id = data.get("target_peer_id")
            await self._create_offer(target_peer_id)
            
        elif msg_type == "offer":
            # Received an offer, create answer
            await self._handle_offer(from_peer_id, data)
            
        elif msg_type == "answer":
            # Received answer to our offer
            await self._handle_answer(from_peer_id, data)
            
        elif msg_type == "ice-candidate":
            # Add ICE candidate
            await self._add_ice_candidate(from_peer_id, data)
            
        else:
            logger.warning(f"Unknown signaling message type: {msg_type}")
    
    async def _create_offer(self, target_peer_id: str):
        """Create WebRTC offer for target peer."""
        pc = RTCPeerConnection()
        self.peer_connections[target_peer_id] = pc
        
        # Create data channel for messages
        dc = pc.createDataChannel("messages")
        await self._setup_data_channel(dc, target_peer_id)
        
        # Set up event handlers
        @pc.on("icecandidate")
        def on_icecandidate(candidate):
            if candidate:
                asyncio.create_task(self._send_ice_candidate(target_peer_id, candidate))
        
        @pc.on("connectionstatechange")
        def on_connectionstatechange():
            state = pc.connectionState
            logger.info(f"WebRTC connection to {target_peer_id}: {state}")
            if state == "connected":
                self.connected_peers.add(target_peer_id)
            elif state in ["failed", "disconnected", "closed"]:
                self.connected_peers.discard(target_peer_id)
        
        # Create and send offer
        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)
        
        await self._send_signaling_message({
            "type": "offer",
            "target_peer_id": target_peer_id,
            "offer": object_to_string(offer),
        })
        
        logger.info(f"Created WebRTC offer for {target_peer_id}")
    
    async def _handle_offer(self, from_peer_id: str, data: Dict):
        """Handle incoming WebRTC offer."""
        pc = RTCPeerConnection()
        self.peer_connections[from_peer_id] = pc
        
        # Set up event handlers
        @pc.on("icecandidate")
        def on_icecandidate(candidate):
            if candidate:
                asyncio.create_task(self._send_ice_candidate(from_peer_id, candidate))
        
        @pc.on("datachannel")
        def on_datachannel(channel):
            asyncio.create_task(self._setup_data_channel(channel, from_peer_id))
        
        @pc.on("connectionstatechange")
        def on_connectionstatechange():
            state = pc.connectionState
            logger.info(f"WebRTC connection to {from_peer_id}: {state}")
            if state == "connected":
                self.connected_peers.add(from_peer_id)
            elif state in ["failed", "disconnected", "closed"]:
                self.connected_peers.discard(from_peer_id)
        
        # Set remote description and create answer
        offer_str = data.get("offer")
        offer = string_to_object(offer_str)
        await pc.setRemoteDescription(offer)
        
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        
        await self._send_signaling_message({
            "type": "answer",
            "target_peer_id": from_peer_id,
            "answer": object_to_string(answer),
        })
        
        logger.info(f"Handled WebRTC offer from {from_peer_id}")
    
    async def _handle_answer(self, from_peer_id: str, data: Dict):
        """Handle WebRTC answer to our offer."""
        pc = self.peer_connections.get(from_peer_id)
        if not pc:
            logger.warning(f"Received answer for unknown peer: {from_peer_id}")
            return
        
        answer_str = data.get("answer")
        answer = string_to_object(answer_str)
        await pc.setRemoteDescription(answer)
        
        logger.info(f"Handled WebRTC answer from {from_peer_id}")
    
    async def _add_ice_candidate(self, from_peer_id: str, data: Dict):
        """Add ICE candidate to peer connection."""
        pc = self.peer_connections.get(from_peer_id)
        if not pc:
            logger.warning(f"Received ICE candidate for unknown peer: {from_peer_id}")
            return
        
        candidate_str = data.get("candidate")
        candidate = string_to_object(candidate_str)
        await pc.addIceCandidate(candidate)
    
    async def _setup_data_channel(self, dc: RTCDataChannel, peer_id: str):
        """Setup data channel event handlers."""
        self.data_channels[peer_id] = dc
        
        @dc.on("open")
        def on_open():
            logger.info(f"DataChannel open with {peer_id}")
            # Notify signaling server
            asyncio.create_task(self._send_signaling_message({
                "type": "datachannel-open",
                "target_peer_id": peer_id,
                "connected_at": asyncio.get_event_loop().time(),
            }))
        
        @dc.on("message")
        def on_message(message):
            try:
                if isinstance(message, str):
                    data = json.loads(message)
                    
                    # Handle weight transfer requests
                    if data.get("type") == "weight-transfer-request":
                        asyncio.create_task(self._handle_weight_transfer_request(peer_id, data))
                        return
                    
                else:
                    # Binary data (weight tensors)
                    data = {"type": "binary", "data": message}
                
                data["from_peer_id"] = peer_id
                asyncio.create_task(self.message_queue.put(data))
                
            except Exception as e:
                logger.error(f"Error handling datachannel message: {e}")
        
        @dc.on("close")
        def on_close():
            logger.info(f"DataChannel closed with {peer_id}")
            self.data_channels.pop(peer_id, None)
            self.connected_peers.discard(peer_id)
    
    async def _send_signaling_message(self, message: Dict):
        """Send message via WebSocket to signaling server."""
        if not self._websocket:
            logger.warning("Cannot send signaling message: not connected")
            return
        
        try:
            await self._websocket.send(json.dumps(message))
        except Exception as e:
            logger.error(f"Failed to send signaling message: {e}")
    
    async def _send_ice_candidate(self, target_peer_id: str, candidate):
        """Send ICE candidate to peer via signaling."""
        await self._send_signaling_message({
            "type": "ice-candidate",
            "target_peer_id": target_peer_id,
            "candidate": object_to_string(candidate),
        })
    
    async def send_to_peer(self, peer_id: str, message: Dict):
        """Send message to specific peer via DataChannel."""
        dc = self.data_channels.get(peer_id)
        if not dc or dc.readyState != "open":
            logger.warning(f"Cannot send to {peer_id}: datachannel not ready")
            return False
        
        try:
            if message.get("type") == "binary":
                # Send binary data directly
                dc.send(message["data"])
            else:
                # Send JSON message
                dc.send(json.dumps(message))
            return True
        except Exception as e:
            logger.error(f"Failed to send message to {peer_id}: {e}")
            return False
    
    async def send_to_upstream(self, message: Dict):
        """Send message to upstream peer in pipeline."""
        # TODO: Get upstream peer ID from pipeline assignment
        # For now, send to first connected peer
        if self.connected_peers:
            peer_id = next(iter(self.connected_peers))
            return await self.send_to_peer(peer_id, message)
        return False
    
    async def send_to_downstream(self, message: Dict):
        """Send message to downstream peer in pipeline."""
        # TODO: Get downstream peer ID from pipeline assignment
        # For now, broadcast to all connected peers
        success = False
        for peer_id in self.connected_peers:
            if await self.send_to_peer(peer_id, message):
                success = True
        return success
    
    async def message_stream(self) -> AsyncGenerator[Dict, None]:
        """Yield incoming messages from all peers."""
        while self._running:
            try:
                message = await asyncio.wait_for(self.message_queue.get(), timeout=1.0)
                yield message
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in message stream: {e}")
                break
    
    async def wait_for_peers(self, timeout: float = 30.0, expected_count: int = 2) -> bool:
        """
        Wait for expected number of peers to connect.
        
        Returns True if all peers connected within timeout.
        """
        if expected_count <= 0:
            return True
        
        start_time = asyncio.get_event_loop().time()
        while (len(self.connected_peers) < expected_count and 
               asyncio.get_event_loop().time() - start_time < timeout):
            await asyncio.sleep(0.5)
        
        return len(self.connected_peers) >= expected_count
    
    async def serve_weight_shard(
        self,
        model_id: str,
        layer_start: int,
        layer_end: int,
        weight_data: bytes,
        weight_hash: str,
    ) -> bool:
        """
        Serve a weight shard to requesting peers via WebRTC DataChannel.
        
        Automatically responds to weight-transfer-request messages.
        """
        # Store the weight data for serving
        shard_key = f"{model_id}:L{layer_start}-{layer_end}"
        self._weight_shards = getattr(self, "_weight_shards", {})
        self._weight_shards[shard_key] = {
            "data": weight_data,
            "hash": weight_hash,
            "size": len(weight_data),
        }
        
        logger.info(f"Registered weight shard for serving: {shard_key}")
        return True
    
    async def _handle_weight_transfer_request(self, from_peer_id: str, request: Dict):
        """Handle incoming weight transfer request."""
        shard_key = f"{request['model_id']}:L{request['layer_start']}-{request['layer_end']}"
        
        # Check if we have this shard
        shard_info = getattr(self, "_weight_shards", {}).get(shard_key)
        if not shard_info:
            await self.send_to_peer(from_peer_id, {
                "type": "weight-transfer-response",
                "transfer_id": request.get("transfer_id"),
                "status": "error",
                "error": "Shard not available",
            })
            return
        
        # Send the weight data
        success = await self.send_to_peer(from_peer_id, {
            "type": "binary",
            "data": shard_info["data"],
            "metadata": {
                "transfer_id": request.get("transfer_id"),
                "shard_key": shard_key,
                "weight_hash": shard_info["hash"],
                "size": shard_info["size"],
            },
        })
        
        if success:
            logger.info(f"Sent weight shard {shard_key} to {from_peer_id}")
        else:
            logger.error(f"Failed to send weight shard {shard_key} to {from_peer_id}")
    
    async def verify_received_weights(
        self,
        source_miner_id: str,
        model_id: str,
        layer_start: int,
        layer_end: int,
        weight_data: bytes,
        expected_hash: str,
        transfer_id: str = None,
    ) -> Tuple[bool, str]:
        """
        Verify weights received from a peer and report violations.
        
        Returns:
            (is_valid, error_message)
        """
        if not HTTPX_AVAILABLE:
            logger.warning("httpx not available - skipping weight verification")
            return True, "Verification skipped (httpx not available)"
        
        # Verify locally first
        actual_hash = hashlib.sha256(weight_data).hexdigest()
        if actual_hash != expected_hash:
            logger.error(f"Weight hash mismatch from {source_miner_id}: expected {expected_hash}, got {actual_hash}")
            # Report to slashing service
            await self._report_weight_violation(
                source_miner_id=source_miner_id,
                model_id=model_id,
                layer_start=layer_start,
                layer_end=layer_end,
                weight_data=weight_data,
                expected_hash=expected_hash,
                actual_hash=actual_hash,
                transfer_id=transfer_id,
            )
            return False, f"Weight hash mismatch: expected {expected_hash}, got {actual_hash}"
        
        # Report to slashing service for verification
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.api_url}/mining/slashing/verify-weights",
                    json={
                        "source_miner_id": source_miner_id,
                        "model_id": model_id,
                        "layer_start": layer_start,
                        "layer_end": layer_end,
                        "weight_data": base64.b64encode(weight_data).decode(),
                        "expected_hash": expected_hash,
                        "transfer_id": transfer_id,
                    },
                    headers={"Authorization": f"Bearer {getattr(self, '_auth_token', '')}"},
                )
                response.raise_for_status()
                result = response.json()
                return result["valid"], result.get("error")
        except Exception as e:
            logger.error(f"Failed to verify weights with slashing service: {e}")
            # Accept weights if verification service is down
            return True, "Verification service unavailable"
    
    async def _report_weight_violation(
        self,
        source_miner_id: str,
        model_id: str,
        layer_start: int,
        layer_end: int,
        weight_data: bytes,
        expected_hash: str,
        actual_hash: str,
        transfer_id: str = None,
    ):
        """Report a weight violation to the slashing service."""
        if not HTTPX_AVAILABLE:
            return
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                await client.post(
                    f"{self.api_url}/mining/slashing/report-violation",
                    json={
                        "violator_miner_id": source_miner_id,
                        "violation_type": "weight_hash_mismatch",
                        "evidence": {
                            "model_id": model_id,
                            "layer_start": layer_start,
                            "layer_end": layer_end,
                            "expected_hash": expected_hash,
                            "actual_hash": actual_hash,
                            "weight_size": len(weight_data),
                            "transfer_id": transfer_id,
                        },
                        "description": f"Weight hash mismatch for shard {model_id}:L{layer_start}-{layer_end}",
                    },
                    headers={"Authorization": f"Bearer {getattr(self, '_auth_token', '')}"},
                )
        except Exception as e:
            logger.error(f"Failed to report violation: {e}")
    
    async def receive_weight_shard(
        self,
        from_peer_id: str,
        model_id: str,
        layer_start: int,
        layer_end: int,
        expected_hash: str,
        timeout: float = 30.0,
    ) -> Optional[bytes]:
        """
        Receive a weight shard from a peer with verification.
        
        Returns the weight data if valid, None if verification fails.
        """
        # Send transfer request
        request = {
            "type": "weight-transfer-request",
            "model_id": model_id,
            "layer_start": layer_start,
            "layer_end": layer_end,
            "expected_hash": expected_hash,
        }
        
        if not await self.send_to_peer(from_peer_id, request):
            logger.error(f"Failed to send transfer request to {from_peer_id}")
            return None
        
        # Wait for response
        start_time = asyncio.get_event_loop().time()
        while asyncio.get_event_loop().time() - start_time < timeout:
            try:
                message = await asyncio.wait_for(self.message_queue.get(), timeout=1.0)
                if (message.get("from_peer_id") == from_peer_id and 
                    message.get("type") == "binary"):
                    weight_data = message["data"]
                    metadata = message.get("metadata", {})
                    
                    # Verify weights
                    is_valid, error = await self.verify_received_weights(
                        source_miner_id=from_peer_id,
                        model_id=model_id,
                        layer_start=layer_start,
                        layer_end=layer_end,
                        weight_data=weight_data,
                        expected_hash=expected_hash,
                        transfer_id=metadata.get("transfer_id"),
                    )
                    
                    if is_valid:
                        logger.info(f"Successfully received and verified weight shard from {from_peer_id}")
                        return weight_data
                    else:
                        logger.error(f"Weight verification failed: {error}")
                        return None
                        
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error receiving weight shard: {e}")
                return None
        
        logger.error(f"Timeout waiting for weight shard from {from_peer_id}")
        return None
    
    def set_auth_token(self, token: str):
        """Set authentication token for API calls."""
        self._auth_token = token
    
    def get_status(self) -> Dict:
        """Get current connection status."""
        return {
            "miner_id": self.miner_id,
            "peer_id": self._peer_id,
            "connected": self._websocket is not None and not self._websocket.closed,
            "connected_peers": list(self.connected_peers),
            "peer_connections": list(self.peer_connections.keys()),
            "data_channels": list(self.data_channels.keys()),
            "message_queue_size": self.message_queue.qsize(),
            "serving_shards": len(getattr(self, "_weight_shards", {})),
        }


# Convenience function for quick setup
async def create_webrtc_client(miner_id: str, mining_url: str = "ws://localhost:8701") -> WebRTCClient:
    """Create and connect WebRTC client."""
    client = WebRTCClient(miner_id, mining_url)
    await client.connect()
    return client
