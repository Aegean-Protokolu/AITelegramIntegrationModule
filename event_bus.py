# E:/CodingWorkspace/Projects/AITelegramIntegration/event_bus.py
from typing import Callable, Dict, List, Any
from dataclasses import dataclass
import time


@dataclass
class Event:
    """Standard event packet."""
    type: str
    data: Any = None
    source: str = "System"
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


class EventBus:
    """
    Central nervous system for the architecture.
    Decouples components by allowing them to publish/subscribe to events.
    """

    def __init__(self):
        self._subscribers: Dict[str, List[Callable[[Event], None]]] = {}

    def subscribe(self, event_type: str, callback: Callable[[Event], None]):
        """Register a callback for a specific event type."""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(callback)

    def publish(self, event_type: str, data: Any = None, source: str = "System"):
        """Broadcast an event to all subscribers."""
        event = Event(event_type, data, source)

        # Notify specific listeners
        if event_type in self._subscribers:
            for callback in self._subscribers[event_type]:
                try:
                    callback(event)
                except Exception as e:
                    print(f"❌ Event Bus Error processing '{event_type}': {e}")

        # Notify wildcard listeners (optional, good for logging)
        if "*" in self._subscribers:
            for callback in self._subscribers["*"]:
                try:
                    callback(event)
                except Exception:
                    pass